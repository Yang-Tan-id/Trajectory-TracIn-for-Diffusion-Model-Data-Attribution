#!/usr/bin/env python3
"""Measure held-out LDS of every Q8 checkpoint component before/after flip."""

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

from analyze_original_f_timestamp_sign_crossfit import TARGETS, lds, target_data


def parse_signs(text):
    result = {}
    for item in text.split(","):
        coordinate, sign = item.split(":", 1)
        result[int(coordinate)] = 1 if sign.strip() == "+" else -1
    return result


def majority_signs(path, query, variant, method):
    counts = defaultdict(list)
    with path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["query"]) == query
            and row["variant"] == variant
            and row["method"] == method
        ]
    if not rows or "selected_signs" not in rows[0]:
        raise ValueError("selected_signs missing from matching crossfit rows")
    for row in rows:
        for coordinate, sign in parse_signs(row["selected_signs"]).items():
            counts[coordinate].append(sign)
    signs = np.asarray(
        [1 if np.mean(counts[index]) >= 0 else -1 for index in sorted(counts)],
        dtype=np.int8,
    )
    return signs, len(rows)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--query-id", type=int, default=8)
    parser.add_argument("--variant", default="query_train_l2")
    parser.add_argument("--method", default="five_bins")
    parser.add_argument("--component-cache", type=Path, required=True)
    parser.add_argument("--per-split", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260916)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.component_cache, allow_pickle=False) as payload:
        query_ids = np.asarray(payload["query_ids"], dtype=np.int32)
        qslots = np.flatnonzero(query_ids == args.query_id)
        if len(qslots) != 1:
            raise ValueError(f"Q{args.query_id} absent from {query_ids.tolist()}")
        checkpoints = np.asarray(payload["checkpoints"], dtype=np.int32)
        score_indices = np.asarray(payload["score_indices"], dtype=np.int64)
        components = np.asarray(
            payload[args.variant][int(qslots[0])], dtype=np.float64
        )
    signs, sign_split_count = majority_signs(
        args.per_split, args.query_id, args.variant, args.method
    )
    if len(signs) != len(components):
        raise ValueError(f"sign/component mismatch: {len(signs)} vs {components.shape}")

    # Reuse the exact subset incidence and target loader used by the crossfit run.
    args.skip_predicted = True
    args.shard_index = 0
    args.shard_count = 1
    args.run_id = "q8_per_checkpoint_target_lds"
    incidence, true = target_data(args, args.query_id, score_indices)
    endpoint = true[TARGETS[0]]
    trajectory = true[TARGETS[1]]
    predictions = -components @ incidence.T

    rng = np.random.default_rng(args.random_seed)
    folds = []
    for _ in range(args.repeats):
        permutation = rng.permutation(len(endpoint))
        folds.extend((permutation[::2], permutation[1::2]))

    split_rows = []
    for fold_index, heldout in enumerate(folds):
        for checkpoint in range(len(checkpoints)):
            raw = lds(
                predictions[checkpoint, heldout],
                endpoint[heldout],
                trajectory[heldout],
            )
            split_rows.append(
                {
                    "fold": fold_index,
                    "checkpoint": int(checkpoints[checkpoint]) + 1,
                    "epoch": 4 * (int(checkpoints[checkpoint]) + 1),
                    "checkpoint_bin": min(checkpoint // 10 + 1, 5),
                    "flip_sign": int(signs[checkpoint]),
                    "raw_endpoint_percent": float(raw[0][0]),
                    "raw_trajectory_percent": float(raw[1][0]),
                    "raw_joint_percent": float(raw[2][0]),
                    "flipped_endpoint_percent": signs[checkpoint] * float(raw[0][0]),
                    "flipped_trajectory_percent": signs[checkpoint] * float(raw[1][0]),
                    "flipped_joint_percent": signs[checkpoint] * float(raw[2][0]),
                }
            )

    checkpoint_rows = []
    for checkpoint in range(len(checkpoints)):
        group = [row for row in split_rows if row["checkpoint"] == checkpoint + 1]
        raw = np.asarray([row["raw_joint_percent"] for row in group])
        flipped = np.asarray([row["flipped_joint_percent"] for row in group])
        checkpoint_rows.append(
            {
                "checkpoint": checkpoint + 1,
                "epoch": 4 * (checkpoint + 1),
                "checkpoint_bin": min(checkpoint // 10 + 1, 5),
                "flip_sign": int(signs[checkpoint]),
                "raw_joint_mean_percent": float(np.mean(raw)),
                "raw_joint_std_percent": float(np.std(raw)),
                "raw_positive_fraction": float(np.mean(raw > 0.0)),
                "flipped_joint_mean_percent": float(np.mean(flipped)),
                "flipped_positive_fraction": float(np.mean(flipped > 0.0)),
                "flip_matches_raw_lds_sign": int(
                    np.sign(np.mean(raw)) == signs[checkpoint]
                ),
            }
        )

    bin_rows = []
    for bin_index in range(1, 6):
        indices = np.flatnonzero(
            np.asarray([min(index // 10 + 1, 5) for index in range(len(checkpoints))])
            == bin_index
        )
        # Evaluate the whole bin as one component; this preserves covariance
        # between checkpoints and is not the average of individual LDS values.
        bin_prediction = predictions[indices].sum(axis=0)
        sign = int(signs[indices[0]])
        raw_values = []
        for heldout in folds:
            value = float(
                lds(bin_prediction[heldout], endpoint[heldout], trajectory[heldout])[2][0]
            )
            raw_values.append(value)
        raw_values = np.asarray(raw_values)
        bin_rows.append(
            {
                "checkpoint_bin": bin_index,
                "checkpoint_start": int(indices[0]) + 1,
                "checkpoint_end": int(indices[-1]) + 1,
                "flip_sign": sign,
                "raw_joint_mean_percent": float(np.mean(raw_values)),
                "raw_joint_std_percent": float(np.std(raw_values)),
                "raw_positive_fraction": float(np.mean(raw_values > 0.0)),
                "flipped_joint_mean_percent": float(np.mean(sign * raw_values)),
                "flipped_positive_fraction": float(np.mean(sign * raw_values > 0.0)),
                "flip_matches_raw_lds_sign": int(np.sign(np.mean(raw_values)) == sign),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_fold_checkpoint.csv", split_rows)
    write_csv(args.out_dir / "per_checkpoint_summary.csv", checkpoint_rows)
    write_csv(args.out_dir / "per_bin_summary.csv", bin_rows)

    print(
        f"Q{args.query_id} {args.variant} PER-CHECKPOINT HELD-OUT TARGET LDS; "
        f"folds={len(folds)} signs_from={sign_split_count}"
    )
    print("\nPER BIN (joint LDS; bin components summed before correlation)")
    print("BIN CHECKPOINTS SIGN  RAW MEAN RAW STD RAW+  FLIPPED FLIP+ MATCH")
    for row in bin_rows:
        print(
            f"{row['checkpoint_bin']:3d} "
            f"{row['checkpoint_start']:02d}-{row['checkpoint_end']:02d} "
            f"{row['flip_sign']:+4d} {row['raw_joint_mean_percent']:+9.3f}% "
            f"{row['raw_joint_std_percent']:7.3f}% {row['raw_positive_fraction']:5.2f} "
            f"{row['flipped_joint_mean_percent']:+8.3f}% "
            f"{row['flipped_positive_fraction']:5.2f} "
            f"{row['flip_matches_raw_lds_sign']:5d}"
        )
    print("\nPER CHECKPOINT")
    print("CKPT EPOCH BIN SIGN  RAW MEAN RAW STD RAW+  FLIPPED FLIP+ MATCH")
    for row in checkpoint_rows:
        print(
            f"{row['checkpoint']:4d} {row['epoch']:5d} {row['checkpoint_bin']:3d} "
            f"{row['flip_sign']:+4d} {row['raw_joint_mean_percent']:+9.3f}% "
            f"{row['raw_joint_std_percent']:7.3f}% {row['raw_positive_fraction']:5.2f} "
            f"{row['flipped_joint_mean_percent']:+8.3f}% "
            f"{row['flipped_positive_fraction']:5.2f} "
            f"{row['flip_matches_raw_lds_sign']:5d}"
        )
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
