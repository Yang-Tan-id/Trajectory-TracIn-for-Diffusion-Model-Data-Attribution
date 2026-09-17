#!/usr/bin/env python3
"""Compare Q8 checkpoint-own and reference trajectory states by timestep."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from analyze_predicted_noise_probe24_output_alignment import artifact_path, write_csv
from analyze_q8_checkpoint_sign_predicted_noise_geometry import majority_signs


def cosine(left, right):
    return float(
        np.dot(left, right)
        / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12)
    )


def summarize(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for values, group in sorted(groups.items()):
        result = dict(zip(keys, values))
        result["terms"] = len(group)
        for metric in ("cosine", "centered_cosine", "rmse", "mean_absolute_error"):
            samples = np.asarray([row[metric] for row in group], dtype=np.float64)
            result[f"{metric}_mean"] = float(np.mean(samples))
            result[f"{metric}_std"] = float(np.std(samples))
        output.append(result)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-id", type=int, default=8)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--checkpoint-crossfit-dir", type=Path, required=True)
    parser.add_argument("--variant", default="query_train_l2")
    parser.add_argument("--method", default="five_bins")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    signs, stability, split_count = majority_signs(
        args.checkpoint_crossfit_dir / "per_split.csv",
        args.query_id,
        args.variant,
        args.method,
        "checkpoint",
    )
    path = artifact_path(
        args.experiment, args.train_seed, args.epochs, args.query_id, args.namespace
    )
    with np.load(path, allow_pickle=False) as payload:
        states = np.asarray(payload["checkpoint_own_trajectory_states"], dtype=np.float64)
        reference_states = np.asarray(
            payload["checkpoint_own_trajectory_reference_states"], dtype=np.float64
        )
        timesteps = np.asarray(
            payload["checkpoint_own_trajectory_state_timesteps"], dtype=np.int32
        )
        checkpoints = np.unique(np.asarray(payload["ckpt_indices"], dtype=np.int32))

    if states.shape[:2] != (len(checkpoints), len(timesteps)):
        raise ValueError(
            f"state shape mismatch: {states.shape} vs checkpoints={len(checkpoints)}, "
            f"timesteps={len(timesteps)}"
        )
    if reference_states.shape[0] != len(timesteps) or len(signs) != len(checkpoints):
        raise ValueError(
            f"reference/sign mismatch: reference={reference_states.shape}, signs={signs.shape}"
        )

    rows = []
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        for timestep_index, timestep in enumerate(timesteps):
            own = states[checkpoint_index, timestep_index].reshape(-1)
            reference = reference_states[timestep_index].reshape(-1)
            difference = own - reference
            own_centered = own - np.mean(own)
            reference_centered = reference - np.mean(reference)
            rows.append(
                {
                    "checkpoint": int(checkpoint) + 1,
                    "epoch": 4 * (int(checkpoint) + 1),
                    "checkpoint_bin": min(checkpoint_index // 10 + 1, 5),
                    "flip_sign": int(signs[checkpoint_index]),
                    "flip_stability": float(stability[int(checkpoint)]),
                    "timestep": int(timestep),
                    "cosine": cosine(own, reference),
                    "centered_cosine": cosine(own_centered, reference_centered),
                    "rmse": float(np.sqrt(np.mean(np.square(difference)))),
                    "mean_absolute_error": float(np.mean(np.abs(difference))),
                }
            )

    by_bin_timestep = summarize(rows, ("checkpoint_bin", "flip_sign", "timestep"))
    by_sign_timestep = summarize(rows, ("flip_sign", "timestep"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_checkpoint_timestep.csv", rows)
    write_csv(args.out_dir / "by_checkpoint_bin_timestep.csv", by_bin_timestep)
    write_csv(args.out_dir / "by_flip_sign_timestep.csv", by_sign_timestep)

    print(
        f"Q{args.query_id} CHECKPOINT-OWN STATES vs REFERENCE STATES; "
        f"signs from {split_count} splits"
    )
    print("\nBY CHECKPOINT BIN AND TIMESTAMP")
    print("BIN SIGN    T   N  CENTERED_COS     RMSE      MAE")
    for row in sorted(
        by_bin_timestep, key=lambda item: (item["checkpoint_bin"], -item["timestep"])
    ):
        print(
            f"{int(row['checkpoint_bin']):3d} {int(row['flip_sign']):+4d} "
            f"{int(row['timestep']):4d} {int(row['terms']):3d} "
            f"{row['centered_cosine_mean']:+13.5f} {row['rmse_mean']:8.5f} "
            f"{row['mean_absolute_error_mean']:8.5f}"
        )

    print("\nBIN 2 (+) MINUS BIN 3 (-), MATCHED BY TIMESTAMP")
    print("   T   DELTA CENTERED_COS   DELTA RMSE   DELTA MAE")
    lookup = {
        (int(row["checkpoint_bin"]), int(row["timestep"])): row
        for row in by_bin_timestep
    }
    for timestep in sorted(timesteps, reverse=True):
        bin2 = lookup[(2, int(timestep))]
        bin3 = lookup[(3, int(timestep))]
        print(
            f"{int(timestep):4d} "
            f"{bin2['centered_cosine_mean'] - bin3['centered_cosine_mean']:+20.5f} "
            f"{bin2['rmse_mean'] - bin3['rmse_mean']:+12.5f} "
            f"{bin2['mean_absolute_error_mean'] - bin3['mean_absolute_error_mean']:+11.5f}"
        )
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
