#!/usr/bin/env python3
"""Rescore saved predicted-noise probes after orienting them with eps_theta."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import jax


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))
if str(SHAPES_ROOT / "script") not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT / "script"))

from analyze_predicted_noise_probe8_choose4 import (
    TARGETS,
    cache_group,
    load_target_data,
    spearman,
    write_csv,
)
from run_predicted_noise_jvp_l2_squared import (
    load_query_bank,
    query_artifact_path,
    train_part_dir,
)
from analyze_reference_probe_delta_alignment import (
    PROBE_SEEDS as TIMESTAMP_SHARED_PROBE_SEEDS,
    artifact_path as delta_geometry_artifact_path,
    probe_key as timestamp_shared_probe_key,
)


BANKS = {
    "1-4": (0, 1, 2, 3),
    "5-8": (4, 5, 6, 7),
    "9-12": (8, 9, 10, 11),
    "5-12": tuple(range(4, 12)),
    "1-8": tuple(range(8)),
    "1-12": tuple(range(12)),
}
SCHEMES = (
    "angle_sign",
    "angle_weighted",
    "angle_sign_delta_norm",
    "angle_weighted_delta_norm",
)
VARIANTS = ("raw", "query_l2", "train_l2", "both_l2")


def records() -> list[dict]:
    return json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]


def aggregate_oriented_queries(
    query_term: np.ndarray,
    scalar_term: np.ndarray,
    probe_indices: tuple[int, ...],
    scheme: str,
) -> np.ndarray:
    """Aggregate first, leaving query-L2 normalization to the score stage."""
    selected = np.asarray(probe_indices, dtype=np.int64)
    if scheme == "angle_sign":
        alpha = np.where(scalar_term[selected] >= 0.0, 1.0, -1.0)
    elif scheme == "angle_weighted":
        alpha = scalar_term[selected]
    elif scheme == "angle_sign_delta_norm":
        # Apply delta_norm after query normalization at the score stage.  Doing
        # it here would cancel identically in query_l2 and both_l2.
        alpha = np.where(scalar_term[selected] >= 0.0, 1.0, -1.0)
    elif scheme == "angle_weighted_delta_norm":
        alpha = scalar_term[selected]
    else:
        raise ValueError(f"unknown angle-orientation scheme {scheme!r}")
    return (
        np.einsum(
            "pq,pqd->qd",
            alpha.astype(np.float32),
            query_term[selected],
            optimize=True,
        )
        / float(len(selected))
    )


def load_alignments(args: argparse.Namespace, reference: dict[str, np.ndarray]):
    if args.alignment_source == "timestamp_shared_delta":
        return load_timestamp_shared_delta_alignments(args, reference)
    scalars = []
    cosines = []
    for query_id in range(10):
        path = query_artifact_path(
            args.experiment,
            args.train_seed,
            args.epochs,
            query_id,
            num_probes=args.num_probes,
            probe_index=0,
            query_namespace_pattern=args.alignment_namespace,
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            scalar = np.asarray(payload["probe_scalars"], dtype=np.float32)
            cosine = np.asarray(payload["probe_cosines"], dtype=np.float32)
            if scalar.shape != (args.num_probes, len(reference["ckpt_indices"])):
                raise ValueError(f"unexpected alignment shape in {path}: {scalar.shape}")
            for key in ("ckpt_indices", "timesteps", "term_weights"):
                if not np.allclose(np.asarray(payload[key]), reference[key]):
                    raise ValueError(f"alignment metadata mismatch for {key}: {path}")
        scalars.append(scalar)
        cosines.append(cosine)
    # probe, query, term
    scalar_array = np.stack(scalars, axis=1)
    cosine_array = np.stack(cosines, axis=1)
    # Legacy alignment artifacts do not carry delta norms.  Unit norms preserve
    # the two original schemes; norm-weighted schemes require timestamp_shared_delta.
    delta_norms = np.ones(scalar_array.shape[1:], dtype=np.float32)
    return scalar_array, cosine_array, delta_norms


def load_timestamp_shared_delta_alignments(
    args: argparse.Namespace,
    reference: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orient each saved J^T v toward the current next-checkpoint delta-eps."""
    seeds = [int(value) for value in args.expected_query_probe_seeds.split(",")]
    if len(seeds) != args.num_probes:
        raise ValueError("timestamp-shared delta orientation needs one seed per probe")
    scalars = np.empty((args.num_probes, 10, len(reference["ckpt_indices"])), dtype=np.float32)
    cosines = np.empty_like(scalars)
    all_delta_norms = np.empty((10, len(reference["ckpt_indices"])), dtype=np.float32)
    geometry_args = argparse.Namespace(
        experiment=args.experiment,
        train_seed=args.train_seed,
        epochs=args.epochs,
        geometry_namespace=args.geometry_namespace,
    )
    for query_id in range(10):
        path = delta_geometry_artifact_path(geometry_args, query_id)
        with np.load(path, allow_pickle=False) as payload:
            deltas = np.asarray(
                payload["checkpoint_next_predicted_noise_deltas"], dtype=np.float32
            )
        checkpoint_count, timestamp_count = deltas.shape[:2]
        if checkpoint_count != 49 or timestamp_count != 10:
            raise ValueError(f"unexpected delta shape in {path}: {deltas.shape}")
        timesteps = np.asarray(reference["timesteps"][:timestamp_count], dtype=np.int32)
        delta_norms = np.linalg.norm(deltas.reshape(49, 10, -1), axis=2)
        expanded_delta_norms = np.empty((50, 10), dtype=np.float32)
        expanded_delta_norms[:49] = delta_norms
        expanded_delta_norms[49] = delta_norms[48]
        all_delta_norms[query_id] = expanded_delta_norms.reshape(-1)
        for probe, seed in enumerate(seeds):
            values = np.empty((50, 10), dtype=np.float32)
            angles = np.empty_like(values)
            for slot, timestep in enumerate(timesteps):
                v = np.asarray(
                    jax.random.normal(
                        timestamp_shared_probe_key(seed, int(timestep)),
                        deltas.shape[2:],
                    ),
                    dtype=np.float32,
                )
                dots = np.sum(deltas[:, slot] * v, axis=tuple(range(1, deltas[:, slot].ndim)))
                values[:49, slot] = dots
                angles[:49, slot] = dots / np.maximum(
                    np.linalg.norm(v) * delta_norms[:, slot], 1e-12
                )
                # No c->c+1 update exists at checkpoint 50.  Orient its term by
                # the update that arrived at checkpoint 50 (49->50).
                values[49, slot] = values[48, slot]
                angles[49, slot] = angles[48, slot]
            scalars[probe, query_id] = values.reshape(-1)
            cosines[probe, query_id] = angles.reshape(-1)
    return scalars, cosines, all_delta_norms


def query_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        experiment=args.experiment,
        train_seed=args.train_seed,
        epochs=args.epochs,
        num_probes=args.num_probes,
        query_namespace_pattern=args.query_namespace_pattern,
        query_namespace_patterns=args.query_namespace_patterns,
        expected_query_probe_mode=args.expected_query_probe_mode,
        expected_query_probe_seed=None,
        expected_query_probe_seeds=args.expected_query_probe_seeds,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num-probes", type=int, default=12)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prediction-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument(
        "--query-namespace-pattern",
        default="loss_direction_residual_rms_predicted_noise_probe4_r{probe_index}",
    )
    parser.add_argument("--query-namespace-patterns", default="")
    parser.add_argument("--expected-query-probe-mode", default="independent_gaussian")
    parser.add_argument("--expected-query-probe-seeds", default="")
    parser.add_argument(
        "--alignment-source",
        choices=("artifact", "timestamp_shared_delta"),
        default="artifact",
    )
    parser.add_argument(
        "--geometry-namespace",
        default="reference_probe_delta_geometry_collect_all",
    )
    parser.add_argument(
        "--alignment-namespace",
        default="predicted_noise_alignment_probe12",
    )
    args = parser.parse_args()
    if args.num_probes != 12:
        raise ValueError("this comparison currently expects the existing 12-probe bank")

    import jax
    import jax.numpy as jnp

    query, meta = load_query_bank(query_args(args))
    scalars, cosines, delta_norms = load_alignments(args, meta)
    term_lookup = {
        (int(ckpt), int(timestep)): index
        for index, (ckpt, timestep) in enumerate(
            zip(meta["ckpt_indices"], meta["timesteps"])
        )
    }
    method_keys = [(scheme, bank) for scheme in SCHEMES for bank in BANKS]
    sums = {
        variant: np.zeros((len(method_keys), 10, 5000), dtype=np.float64)
        for variant in VARIANTS
    }
    score_indices = None

    for ckpt_i in range(50):
        part = train_part_dir(args.experiment, args.train_seed) / f"ckpt_{ckpt_i:04d}.npz"
        with np.load(part, allow_pickle=False) as payload:
            train_terms = np.asarray(payload["train_features"], dtype=np.float32)
            indices = np.asarray(payload["score_indices"], dtype=np.int64)
            ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            weights = np.asarray(payload["term_weights"], dtype=np.float64)
        if score_indices is None:
            score_indices = indices
        elif not np.array_equal(score_indices, indices):
            raise ValueError(f"score indices mismatch in {part}")

        for local_term, (ckpt, timestep, term_weight) in enumerate(
            zip(ckpts, timesteps, weights)
        ):
            term = term_lookup[(int(ckpt), int(timestep))]
            query_term = query[:, :, term, :].astype(np.float32, copy=False)
            aggregated = []
            for scheme, bank_name in method_keys:
                aggregated.append(
                    aggregate_oriented_queries(
                        query_term,
                        scalars[:, :, term],
                        BANKS[bank_name],
                        scheme,
                    )
                )
            aggregated_np = np.stack(aggregated, axis=0)
            flat_query = jax.device_put(
                jnp.asarray(aggregated_np.reshape(len(method_keys) * 10, -1))
            )
            train = jax.device_put(jnp.asarray(train_terms[local_term]))
            directional = train @ flat_query.T
            directional = directional.reshape(5000, len(method_keys), 10)
            train_norm = jnp.linalg.norm(train, axis=1) + 1e-8
            query_norm = jnp.linalg.norm(flat_query, axis=1).reshape(
                len(method_keys), 10
            ) + 1e-8
            term_values = {
                "raw": directional,
                "query_l2": directional / query_norm[None, :, :],
                "train_l2": directional / train_norm[:, None, None],
                "both_l2": directional
                / train_norm[:, None, None]
                / query_norm[None, :, :],
            }
            post_normalization_scale = np.stack(
                [
                    delta_norms[:, term]
                    if scheme.endswith("_delta_norm")
                    else np.ones(10, dtype=np.float32)
                    for scheme, _ in method_keys
                ],
                axis=0,
            )
            for variant, values in term_values.items():
                values = values * jnp.asarray(post_normalization_scale)[None, :, :]
                # method, query, datapoint
                host = np.asarray(jax.device_get(values), dtype=np.float64).transpose(1, 2, 0)
                sums[variant] += float(term_weight) * host
        print(f"[angle score] checkpoint={ckpt_i + 1}/50", flush=True)

    assert score_indices is not None
    result_root = SHAPES_ROOT / "result" / args.experiment
    out_dir = (
        result_root
        / "eval"
        / "predicted_noise_angle_oriented_probe12"
        / f"run_{args.run_id}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "scores.npz",
        score_indices=score_indices,
        method_keys=np.asarray([f"{scheme}:{bank}" for scheme, bank in method_keys]),
        **{f"scores_{variant}": values for variant, values in sums.items()},
    )

    angle_rows = []
    for probe_index in range(args.num_probes):
        values = cosines[probe_index]
        angle_rows.append(
            {
                "probe": probe_index + 1,
                "mean_cosine": float(values.mean()),
                "mean_abs_cosine": float(np.abs(values).mean()),
                "positive_fraction": float(np.mean(values > 0)),
                "std_cosine": float(values.std()),
            }
        )
    write_csv(out_dir / "probe_angle_summary.csv", angle_rows)

    lds_rows = []
    query_records = records()
    for query_id, record in enumerate(query_records):
        prompt_tag = str(record["prompt"]).replace(",", "_")
        eval_root = (
            result_root
            / "eval"
            / "prompted_solo"
            / f"query_{prompt_tag}"
            / f"initial_seed_{int(record['initial_seed'])}"
        )
        incidence, true_values = load_target_data(cache_group(eval_root), score_indices)
        for method_index, (scheme, bank) in enumerate(method_keys):
            for variant in VARIANTS:
                prediction = (
                    args.prediction_sign
                    * sums[variant][method_index, query_id]
                    @ incidence.T
                )
                for target in TARGETS:
                    lds_rows.append(
                        {
                            "scheme": scheme,
                            "bank": bank,
                            "variant": variant,
                            "query": query_id,
                            "target": target,
                            "lds_percent": 100.0
                            * spearman(prediction, true_values[target]),
                        }
                    )
    write_csv(out_dir / "per_query_lds.csv", lds_rows)

    grouped = {}
    for row in lds_rows:
        key = (row["scheme"], row["bank"], row["variant"], row["target"])
        grouped.setdefault(key, []).append(float(row["lds_percent"]))
    summary_rows = [
        {
            "scheme": scheme,
            "bank": bank,
            "variant": variant,
            "target": target,
            "mean_lds_percent": statistics.mean(values),
            "std_lds_percent": statistics.stdev(values),
        }
        for (scheme, bank, variant, target), values in sorted(grouped.items())
    ]
    write_csv(out_dir / "summary_lds.csv", summary_rows)

    lookup = {
        (row["scheme"], row["bank"], row["variant"], row["target"]): float(
            row["mean_lds_percent"]
        )
        for row in summary_rows
    }
    for scheme in SCHEMES:
        print(f"\n{scheme.upper()} — BOTH-L2, 10-query mean")
        print(
            f"{'BANK':8s} {'ENDPOINT':>10s} {'TRAJ':>10s} {'CF JOINT':>10s} "
            f"{'NOISE':>10s} {'SIMPLE':>10s}"
        )
        print("-" * 72)
        for bank in BANKS:
            endpoint = lookup[(scheme, bank, "both_l2", "endpoint_contarfactual")]
            traj = lookup[(scheme, bank, "both_l2", "traj_contarfactual")]
            noise = lookup[(scheme, bank, "both_l2", "noise_trajectory")]
            simple = lookup[(scheme, bank, "both_l2", "simple_loss")]
            print(
                f"{bank:8s} {endpoint:9.3f}% {traj:9.3f}% "
                f"{statistics.mean((endpoint, traj)):9.3f}% "
                f"{noise:9.3f}% {simple:9.3f}%"
            )
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
