#!/usr/bin/env python3
"""Diagnose checkpoint-wise convergence of Q3 delta-weighted probe estimates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import numpy as np

from analyze_reference_probe_delta_alignment import (
    PROBE_SEEDS,
    artifact_path as geometry_artifact_path,
    probe_key,
)
from run_predicted_noise_jvp_l2_squared import query_artifact_path


SHAPES_ROOT = Path(__file__).resolve().parents[1]
BANKS = {"1-4": slice(0, 4), "5-8": slice(4, 8), "9-12": slice(8, 12)}


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.dot(left, right)
        / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12)
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query", type=int, default=3)
    parser.add_argument(
        "--geometry-namespace",
        default="reference_probe_delta_geometry_collect_all",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    geometry_args = argparse.Namespace(
        experiment=args.experiment, train_seed=args.train_seed,
        epochs=args.epochs, geometry_namespace=args.geometry_namespace,
    )
    geometry_path = geometry_artifact_path(geometry_args, args.query)
    with np.load(geometry_path, allow_pickle=False) as payload:
        deltas = np.asarray(
            payload["checkpoint_next_predicted_noise_deltas"], dtype=np.float32
        )

    features = []
    reference = None
    for seed in PROBE_SEEDS:
        namespace = (
            "loss_direction_predicted_noise_probe1_timestamp_shared_"
            f"reference_seed{seed}_r0"
        )
        path = query_artifact_path(
            args.experiment, args.train_seed, args.epochs, args.query,
            num_probes=1, probe_index=0, query_namespace_pattern=namespace,
        )
        with np.load(path, allow_pickle=False) as payload:
            features.append(np.asarray(payload["query_features"], dtype=np.float32))
            metadata = {
                "ckpt_indices": np.asarray(payload["ckpt_indices"], dtype=np.int32),
                "timesteps": np.asarray(payload["timesteps"], dtype=np.int32),
                "term_weights": np.asarray(payload["term_weights"], dtype=np.float64),
            }
        if reference is None:
            reference = metadata
        else:
            for key in metadata:
                if not np.allclose(metadata[key], reference[key]):
                    raise ValueError(f"metadata mismatch for {key}: {path}")
    assert reference is not None
    query_features = np.stack(features, axis=0)  # probe, term, parameter projection
    if query_features.shape[:2] != (12, 500):
        raise ValueError(f"unexpected query feature shape: {query_features.shape}")

    timesteps = reference["timesteps"].reshape(50, 10)
    weights = reference["term_weights"].reshape(50, 10)
    scalars = np.empty((12, 50, 10), dtype=np.float32)
    for probe, seed in enumerate(PROBE_SEEDS):
        for slot, timestep in enumerate(timesteps[0]):
            v = np.asarray(
                jax.random.normal(probe_key(seed, int(timestep)), deltas.shape[2:]),
                dtype=np.float32,
            )
            dots = np.sum(
                deltas[:, slot] * v,
                axis=tuple(range(1, deltas[:, slot].ndim)),
            )
            scalars[probe, :49, slot] = dots
            scalars[probe, 49, slot] = dots[-1]

    weighted = (
        scalars.reshape(12, 500, 1)
        * query_features
    )
    bank_terms = {
        name: np.mean(weighted[selection], axis=0).reshape(50, 10, -1)
        for name, selection in BANKS.items()
    }
    bank_terms["1-12"] = np.mean(weighted, axis=0).reshape(50, 10, -1)
    # A checkpoint's learning-rate multiplier is a common scalar across its
    # timestamp terms.  Remove that scalar before cosine calculations so late
    # cosine-schedule checkpoints do not collapse into the numerical epsilon.
    direction_weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-30)
    bank_checkpoints = {
        name: np.einsum("ct,ctd->cd", direction_weights, values, optimize=True)
        for name, values in bank_terms.items()
    }

    rows = []
    previous_full = None
    for checkpoint in range(50):
        vectors = {name: value[checkpoint] for name, value in bank_checkpoints.items()}
        full = vectors["1-12"]
        row = {
            "checkpoint": checkpoint + 1,
            "epoch": 4 * (checkpoint + 1),
            "cos_1_4__5_8": cosine(vectors["1-4"], vectors["5-8"]),
            "cos_1_4__9_12": cosine(vectors["1-4"], vectors["9-12"]),
            "cos_5_8__9_12": cosine(vectors["5-8"], vectors["9-12"]),
            "cos_1_4__full12": cosine(vectors["1-4"], full),
            "cos_5_8__full12": cosine(vectors["5-8"], full),
            "cos_9_12__full12": cosine(vectors["9-12"], full),
            "cos_full12_previous": (
                float("nan") if previous_full is None else cosine(previous_full, full)
            ),
            "norm_1_4": float(np.linalg.norm(vectors["1-4"])),
            "norm_5_8": float(np.linalg.norm(vectors["5-8"])),
            "norm_9_12": float(np.linalg.norm(vectors["9-12"])),
            "norm_full12": float(np.linalg.norm(full)),
        }
        rows.append(row)
        previous_full = full

    term_rows = []
    for checkpoint in range(50):
        for slot, timestep in enumerate(timesteps[checkpoint]):
            values = {name: bank_terms[name][checkpoint, slot] for name in BANKS}
            term_rows.append({
                "checkpoint": checkpoint + 1,
                "epoch": 4 * (checkpoint + 1),
                "timestep": int(timestep),
                "cos_1_4__5_8": cosine(values["1-4"], values["5-8"]),
                "cos_1_4__9_12": cosine(values["1-4"], values["9-12"]),
                "cos_5_8__9_12": cosine(values["5-8"], values["9-12"]),
            })

    write_csv(args.out_dir / "per_checkpoint.csv", rows)
    write_csv(args.out_dir / "per_checkpoint_timestamp.csv", term_rows)

    print(f"Q{args.query} DELTA-WEIGHTED PROBE ESTIMATOR CONVERGENCE BY CHECKPOINT")
    print(
        f"{'CKPT':>4s} {'EPOCH':>5s} {'1-4~5-8':>10s} {'1-4~9-12':>11s} "
        f"{'5-8~9-12':>12s} {'1-4~ALL':>10s} {'5-8~ALL':>10s} "
        f"{'9-12~ALL':>10s} {'ALL~PREV':>10s}"
    )
    print("-" * 102)
    for row in rows:
        print(
            f"{row['checkpoint']:4d} {row['epoch']:5d} "
            f"{row['cos_1_4__5_8']:+10.4f} {row['cos_1_4__9_12']:+11.4f} "
            f"{row['cos_5_8__9_12']:+12.4f} {row['cos_1_4__full12']:+10.4f} "
            f"{row['cos_5_8__full12']:+10.4f} {row['cos_9_12__full12']:+10.4f} "
            f"{row['cos_full12_previous']:+10.4f}"
        )

    pair_keys = ("cos_1_4__5_8", "cos_1_4__9_12", "cos_5_8__9_12")
    print("\nSUMMARY")
    for key in pair_keys:
        values = np.asarray([float(row[key]) for row in rows])
        print(
            f"{key}: mean={values.mean():+.4f} median={np.median(values):+.4f} "
            f"negative={np.mean(values < 0):.3f}"
        )
    continuity = np.asarray([float(row["cos_full12_previous"]) for row in rows[1:]])
    print(
        f"full12 checkpoint continuity: mean={continuity.mean():+.4f} "
        f"median={np.median(continuity):+.4f} negative={np.mean(continuity < 0):.3f}"
    )
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
