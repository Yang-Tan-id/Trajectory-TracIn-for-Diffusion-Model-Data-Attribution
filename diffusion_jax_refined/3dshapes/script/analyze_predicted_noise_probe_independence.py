#!/usr/bin/env python3
"""Audit the Gaussian output probes used by predicted-noise attribution.

This reconstructs the probes from the exact RNG-key rule used by TrajTracIn
and reports both within-term and persistent cross-term correlations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT / "script") not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT / "script"))

from run_predicted_noise_jvp_l2_squared import query_artifact_path


DOMAIN_TAG = 0x50524F42


def predicted_noise_probe_key(
    seed: int,
    checkpoint_index: int,
    timestep: int,
    snapshot_position: int,
    probe_index: int,
):
    """Mirror legacy_jax.traj_tracin.algorithm.predicted_noise_probe_key."""
    key = jax.random.PRNGKey(seed)
    for value in (DOMAIN_TAG, checkpoint_index, timestep, snapshot_position):
        key = jax.random.fold_in(key, int(value))
    # Probe zero intentionally preserves the historical key.
    if probe_index:
        key = jax.random.fold_in(key, int(probe_index))
    return key


def parse_shape(text: str) -> tuple[int, ...]:
    shape = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not shape or any(value <= 0 for value in shape):
        raise argparse.ArgumentTypeError(f"invalid positive shape: {text!r}")
    return shape


def load_terms(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    artifact = query_artifact_path(
        args.experiment,
        args.train_seed,
        args.epochs,
        args.query_id,
        num_probes=args.num_probes,
        probe_index=0,
        query_namespace_pattern=args.query_namespace_pattern,
    )
    if not artifact.is_file():
        raise FileNotFoundError(
            f"missing reference query artifact: {artifact}\n"
            "Run the independent-probe query pipeline first, or choose a query whose "
            "probe-0 artifact was retained."
        )
    with np.load(artifact, allow_pickle=False) as payload:
        ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
        timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        positions = np.asarray(payload["snapshot_positions"], dtype=np.int32)
        mode = str(np.asarray(payload.get("output_probe_mode", "independent_gaussian")).item())
    if mode != "independent_gaussian":
        raise ValueError(f"artifact probe mode is {mode!r}, not 'independent_gaussian'")
    if not (len(ckpts) == len(timesteps) == len(positions)):
        raise ValueError("query artifact term metadata lengths do not match")
    return ckpts, timesteps, positions, artifact


def off_diagonal(matrix: np.ndarray) -> np.ndarray:
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]


def save_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["probe"] + [f"p{index + 1}" for index in range(len(matrix))])
        for index, row in enumerate(matrix):
            writer.writerow([f"p{index + 1}"] + [f"{value:.10g}" for value in row])


def make_heatmap(path: Path, matrix: np.ndarray, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib unavailable; skipping heatmap", flush=True)
        return
    limit = max(0.01, float(np.max(np.abs(off_diagonal(matrix)))))
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit)
    labels = [str(index + 1) for index in range(len(matrix))]
    ax.set_xticks(range(len(matrix)), labels)
    ax.set_yticks(range(len(matrix)), labels)
    ax.set_xlabel("probe index")
    ax.set_ylabel("probe index")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-id", type=int, default=0)
    parser.add_argument("--num-probes", type=int, default=12)
    parser.add_argument("--output-shape", type=parse_shape, default=(1, 64, 64, 3))
    parser.add_argument(
        "--query-namespace-pattern",
        default="loss_direction_residual_rms_predicted_noise_probe4_r{probe_index}",
    )
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    ckpts, timesteps, positions, artifact = load_terms(args)
    term_count = len(ckpts)
    dimension = int(np.prod(args.output_shape))
    probe_count = int(args.num_probes)
    if probe_count < 2:
        raise ValueError("--num-probes must be at least 2")

    out_dir = args.out_dir or (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "eval"
        / f"probe{probe_count}_independence"
        / f"seed_{args.train_seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    global_gram = np.zeros((probe_count, probe_count), dtype=np.float64)
    sum_term_cos = np.zeros_like(global_gram)
    sum_sq_term_cos = np.zeros_like(global_gram)
    max_abs_term_cos = np.zeros_like(global_gram)
    norms = np.empty((term_count, probe_count), dtype=np.float64)
    coordinate_means = np.empty_like(norms)
    dc_z = np.empty_like(norms)
    adjacent_cos_sum = np.zeros(probe_count, dtype=np.float64)
    adjacent_count = 0
    previous_unit: np.ndarray | None = None
    generate_bank = jax.jit(
        jax.vmap(
            lambda key: jax.random.normal(
                key, args.output_shape, dtype=jnp.float32
            )
        )
    )

    for term_id, (ckpt, timestep, position) in enumerate(
        zip(ckpts, timesteps, positions)
    ):
        keys = jnp.stack(
            [
                predicted_noise_probe_key(
                    args.train_seed,
                    int(ckpt),
                    int(timestep),
                    int(position),
                    probe_index,
                )
                for probe_index in range(probe_count)
            ]
        )
        probes = np.asarray(generate_bank(keys), dtype=np.float64).reshape(
            probe_count, dimension
        )
        term_norms = np.linalg.norm(probes, axis=1)
        unit = probes / term_norms[:, None]
        cosine = unit @ unit.T

        global_gram += probes @ probes.T
        sum_term_cos += cosine
        sum_sq_term_cos += np.square(cosine)
        max_abs_term_cos = np.maximum(max_abs_term_cos, np.abs(cosine))
        norms[term_id] = term_norms
        coordinate_means[term_id] = probes.mean(axis=1)
        # sqrt(D) * cosine(probe, all-ones direction); N(0,1) under iid N(0,I).
        dc_z[term_id] = probes.sum(axis=1) / term_norms
        if previous_unit is not None:
            adjacent_cos_sum += np.sum(previous_unit * unit, axis=1)
            adjacent_count += 1
        previous_unit = unit

        if (term_id + 1) % 50 == 0 or term_id + 1 == term_count:
            print(f"[probe audit] terms={term_id + 1}/{term_count}", flush=True)

    global_norms = np.sqrt(np.diag(global_gram))
    global_cos = global_gram / global_norms[:, None] / global_norms[None, :]
    mean_term_cos = sum_term_cos / term_count
    rms_term_cos = np.sqrt(sum_sq_term_cos / term_count)
    adjacent_cos = adjacent_cos_sum / max(adjacent_count, 1)

    random_cos_sd = 1.0 / math.sqrt(dimension)
    persistent_cos_sd = 1.0 / math.sqrt(dimension * term_count)
    global_off = off_diagonal(global_cos)
    mean_off = off_diagonal(mean_term_cos)
    max_term_off = off_diagonal(max_abs_term_cos)

    per_probe_path = out_dir / "per_probe.csv"
    with per_probe_path.open("w", newline="") as handle:
        fieldnames = [
            "probe",
            "norm_mean",
            "norm_std",
            "coordinate_mean_mean",
            "coordinate_mean_std",
            "dc_z_mean",
            "dc_z_std",
            "dc_positive_fraction",
            "adjacent_term_cosine_mean",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for probe_index in range(probe_count):
            writer.writerow(
                {
                    "probe": probe_index + 1,
                    "norm_mean": norms[:, probe_index].mean(),
                    "norm_std": norms[:, probe_index].std(),
                    "coordinate_mean_mean": coordinate_means[:, probe_index].mean(),
                    "coordinate_mean_std": coordinate_means[:, probe_index].std(),
                    "dc_z_mean": dc_z[:, probe_index].mean(),
                    "dc_z_std": dc_z[:, probe_index].std(),
                    "dc_positive_fraction": np.mean(dc_z[:, probe_index] > 0),
                    "adjacent_term_cosine_mean": adjacent_cos[probe_index],
                }
            )

    save_matrix(out_dir / "global_probe_cosine.csv", global_cos)
    save_matrix(out_dir / "mean_within_term_cosine.csv", mean_term_cos)
    save_matrix(out_dir / "rms_within_term_cosine.csv", rms_term_cos)
    save_matrix(out_dir / "max_abs_within_term_cosine.csv", max_abs_term_cos)
    make_heatmap(
        out_dir / "global_probe_cosine.png",
        global_cos,
        f"Global cosine across {term_count} terms ({probe_count} probes)",
    )

    summary = {
        "artifact": str(artifact),
        "seed": args.train_seed,
        "num_probes": probe_count,
        "num_terms": term_count,
        "output_shape": list(args.output_shape),
        "output_dimension": dimension,
        "theory_random_single_term_cosine_sd": random_cos_sd,
        "theory_persistent_cosine_sd": persistent_cos_sd,
        "global_offdiag_cosine_mean": float(global_off.mean()),
        "global_offdiag_abs_cosine_mean": float(np.abs(global_off).mean()),
        "global_offdiag_abs_cosine_max": float(np.abs(global_off).max()),
        "global_offdiag_max_z": float(np.abs(global_off).max() / persistent_cos_sd),
        "mean_within_term_offdiag_abs_mean": float(np.abs(mean_off).mean()),
        "mean_within_term_offdiag_abs_max": float(np.abs(mean_off).max()),
        "max_single_term_offdiag_abs_cosine": float(max_term_off.max()),
        "norm_mean_all": float(norms.mean()),
        "norm_std_all": float(norms.std()),
        "theory_norm_approx_mean": math.sqrt(dimension),
        "dc_z_mean_all": float(dc_z.mean()),
        "dc_z_std_all": float(dc_z.std()),
        "dc_z_max_abs_probe_mean": float(np.max(np.abs(dc_z.mean(axis=0)))),
        "dc_positive_fraction_min": float(np.min(np.mean(dc_z > 0, axis=0))),
        "dc_positive_fraction_max": float(np.max(np.mean(dc_z > 0, axis=0))),
        "adjacent_same_probe_cosine_abs_max": float(np.max(np.abs(adjacent_cos))),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    print("\n12-PROBE INDEPENDENCE AUDIT" if probe_count == 12 else "\nPROBE INDEPENDENCE AUDIT")
    print(f"terms={term_count} dimension={dimension} source={artifact}")
    print(f"single-term random cosine SD       : {random_cos_sd:.6f}")
    print(f"500-term persistent cosine SD     : {persistent_cos_sd:.6f}")
    print(f"global offdiag |cos| mean / max   : {np.abs(global_off).mean():.6f} / {np.abs(global_off).max():.6f}")
    print(f"largest persistent-correlation z  : {summary['global_offdiag_max_z']:.2f}")
    print(f"largest one-term |cos|            : {max_term_off.max():.6f}")
    print(f"norm mean / std (theory sqrt(D))  : {norms.mean():.4f} / {norms.std():.4f} ({math.sqrt(dimension):.4f})")
    print(f"DC z mean / std                   : {dc_z.mean():.4f} / {dc_z.std():.4f}")
    print(f"largest |mean DC z| by probe      : {summary['dc_z_max_abs_probe_mean']:.4f}")
    print(f"DC-positive fraction range        : {summary['dc_positive_fraction_min']:.3f} .. {summary['dc_positive_fraction_max']:.3f}")
    print(f"same-probe adjacent-term |cos|max : {summary['adjacent_same_probe_cosine_abs_max']:.6f}")
    print("\nPROBE   NORM MEAN±STD       MEAN DC-Z  DC+ FRAC  ADJACENT COS")
    for probe_index in range(probe_count):
        print(
            f"{probe_index + 1:5d}   "
            f"{norms[:, probe_index].mean():8.3f}±{norms[:, probe_index].std():6.3f}   "
            f"{dc_z[:, probe_index].mean():+9.4f}  "
            f"{np.mean(dc_z[:, probe_index] > 0):8.3f}  "
            f"{adjacent_cos[probe_index]:+12.6f}"
        )
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
