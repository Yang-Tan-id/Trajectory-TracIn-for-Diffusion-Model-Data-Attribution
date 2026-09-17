#!/usr/bin/env python3
"""Compare Q8 checkpoint-specific sampling endpoints with the reference endpoint."""

from __future__ import annotations

import argparse
import csv
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
    left = left.reshape(len(left), -1)
    right = np.broadcast_to(right, left.shape)
    return np.sum(left * right, axis=1) / np.maximum(
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-12
    )


def summarize(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    output = []
    for value, group in sorted(groups.items()):
        result = {key: value, "checkpoints": len(group)}
        for metric in ("cosine", "centered_cosine", "rmse", "mean_absolute_error"):
            values = np.asarray([row[metric] for row in group])
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_min"] = float(np.min(values))
            result[f"{metric}_max"] = float(np.max(values))
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
        endpoints = np.asarray(
            payload["checkpoint_own_trajectory_endpoints"], dtype=np.float64
        )
        reference = np.asarray(
            payload["checkpoint_own_trajectory_reference_endpoint"], dtype=np.float64
        )
        checkpoints = np.unique(np.asarray(payload["ckpt_indices"], dtype=np.int32))

    if len(endpoints) != len(checkpoints) or len(signs) != len(checkpoints):
        raise ValueError(
            f"shape mismatch endpoints={endpoints.shape} checkpoints={checkpoints.shape} signs={signs.shape}"
        )
    reference_flat = reference.reshape(1, -1)
    endpoint_flat = endpoints.reshape(len(endpoints), -1)
    raw_cosine = cosine(endpoint_flat, reference_flat)
    centered_endpoint = endpoint_flat - np.mean(endpoint_flat, axis=1, keepdims=True)
    centered_reference = reference_flat - np.mean(reference_flat, axis=1, keepdims=True)
    centered_cosine = cosine(centered_endpoint, centered_reference)
    difference = endpoint_flat - reference_flat
    rmse = np.sqrt(np.mean(np.square(difference), axis=1))
    mae = np.mean(np.abs(difference), axis=1)

    rows = []
    for index, checkpoint in enumerate(checkpoints):
        rows.append(
            {
                "checkpoint": int(checkpoint) + 1,
                "epoch": 4 * (int(checkpoint) + 1),
                "checkpoint_bin": min(index // 10 + 1, 5),
                "flip_sign": int(signs[index]),
                "flip_stability": float(stability[int(checkpoint)]),
                "cosine": float(raw_cosine[index]),
                "centered_cosine": float(centered_cosine[index]),
                "rmse": float(rmse[index]),
                "mean_absolute_error": float(mae[index]),
            }
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_checkpoint.csv", rows)
    by_bin = summarize(rows, "checkpoint_bin")
    by_sign = summarize(rows, "flip_sign")
    write_csv(args.out_dir / "by_checkpoint_bin.csv", by_bin)
    write_csv(args.out_dir / "by_flip_sign.csv", by_sign)

    print(
        f"Q{args.query_id} CHECKPOINT-OWN ENDPOINT vs REFERENCE ENDPOINT; "
        f"signs from {split_count} splits"
    )
    print("\nBY FIVE CHECKPOINT BINS")
    print("BIN SIGN N  COSINE CENTERED_COS   RMSE      MAE")
    for row in by_bin:
        first_index = (int(row["checkpoint_bin"]) - 1) * 10
        print(
            f"{int(row['checkpoint_bin']):3d} {int(signs[first_index]):+4d} "
            f"{int(row['checkpoints']):2d} {row['cosine_mean']:+8.5f} "
            f"{row['centered_cosine_mean']:+12.5f} {row['rmse_mean']:8.5f} "
            f"{row['mean_absolute_error_mean']:8.5f}"
        )
    print("\nBY FLIP SIGN")
    print("SIGN N  COSINE CENTERED_COS   RMSE      MAE")
    for row in by_sign:
        print(
            f"{int(row['flip_sign']):+4d} {int(row['checkpoints']):2d} "
            f"{row['cosine_mean']:+8.5f} {row['centered_cosine_mean']:+12.5f} "
            f"{row['rmse_mean']:8.5f} {row['mean_absolute_error_mean']:8.5f}"
        )
    print("\nPER CHECKPOINT")
    print("CKPT EPOCH BIN SIGN  COSINE CENTERED_COS   RMSE      MAE")
    for row in rows:
        print(
            f"{row['checkpoint']:4d} {row['epoch']:5d} {row['checkpoint_bin']:3d} "
            f"{row['flip_sign']:+4d} {row['cosine']:+8.5f} "
            f"{row['centered_cosine']:+12.5f} {row['rmse']:8.5f} "
            f"{row['mean_absolute_error']:8.5f}"
        )
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
