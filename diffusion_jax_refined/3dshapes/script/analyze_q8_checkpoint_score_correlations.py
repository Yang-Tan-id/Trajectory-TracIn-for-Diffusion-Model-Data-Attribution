#!/usr/bin/env python3
"""Raw versus Q8-flipped checkpoint score-vector correlation matrices."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_signs(text: str) -> dict[int, int]:
    result = {}
    for item in text.split(","):
        coordinate, sign = item.split(":", 1)
        result[int(coordinate)] = 1 if sign.strip() == "+" else -1
    return result


def majority_signs(path: Path, query: int, variant: str, method: str):
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
        raise ValueError(f"missing selected signs for Q{query} {variant} {method}")
    for row in rows:
        for coordinate, sign in parse_signs(row["selected_signs"]).items():
            counts[coordinate].append(sign)
    return np.asarray(
        [1 if np.mean(counts[index]) >= 0 else -1 for index in sorted(counts)],
        dtype=np.int8,
    ), len(rows)


def rank_rows(values):
    order = np.argsort(values, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    row = np.arange(values.shape[0])[:, None]
    ranks[row, order] = np.arange(values.shape[1], dtype=np.float64)[None, :]
    return ranks


def corrcoef_rows(values):
    centered = values - np.mean(values, axis=1, keepdims=True)
    normalized = centered / np.maximum(
        np.linalg.norm(centered, axis=1, keepdims=True), 1e-12
    )
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def cosine_rows(values):
    normalized = values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True), 1e-12
    )
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def write_matrix(path, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["checkpoint"] + list(range(1, len(matrix) + 1)))
        for index, row in enumerate(matrix):
            writer.writerow([index + 1] + row.tolist())


def pair_summary(matrix, signs):
    result = defaultdict(list)
    for left in range(len(matrix)):
        for right in range(left + 1, len(matrix)):
            transition = ("+" if signs[left] > 0 else "-") + (
                "+" if signs[right] > 0 else "-"
            )
            category = "same_sign" if signs[left] == signs[right] else "different_sign"
            result[category].append(matrix[left, right])
            result[transition].append(matrix[left, right])
    rows = []
    for category, values in sorted(result.items()):
        array = np.asarray(values)
        rows.append(
            {
                "category": category,
                "pairs": len(array),
                "mean": float(np.mean(array)),
                "median": float(np.median(array)),
                "positive_fraction": float(np.mean(array > 0.0)),
                "negative_fraction": float(np.mean(array < 0.0)),
            }
        )
    return rows


def write_dicts(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-cache", type=Path, required=True)
    parser.add_argument("--per-split", type=Path, required=True)
    parser.add_argument("--query-id", type=int, default=8)
    parser.add_argument("--variant", default="query_train_l2")
    parser.add_argument("--method", default="five_bins")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.component_cache, allow_pickle=False) as payload:
        query_ids = np.asarray(payload["query_ids"], dtype=np.int32)
        checkpoints = np.asarray(payload["checkpoints"], dtype=np.int32)
        qslots = np.flatnonzero(query_ids == args.query_id)
        if len(qslots) != 1:
            raise ValueError(f"Q{args.query_id} absent from {query_ids.tolist()}")
        scores = np.asarray(payload[args.variant][int(qslots[0])], dtype=np.float64)

    signs, split_count = majority_signs(
        args.per_split, args.query_id, args.variant, args.method
    )
    if len(signs) != len(scores) or len(checkpoints) != len(scores):
        raise ValueError(
            f"shape mismatch: signs={len(signs)} scores={scores.shape} checkpoints={len(checkpoints)}"
        )
    flipped = signs[:, None] * scores

    matrices = {
        "raw_pearson": corrcoef_rows(scores),
        "flipped_pearson": corrcoef_rows(flipped),
        "raw_spearman": corrcoef_rows(rank_rows(scores)),
        "flipped_spearman": corrcoef_rows(rank_rows(flipped)),
        "raw_cosine": cosine_rows(scores),
        "flipped_cosine": cosine_rows(flipped),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, matrix in matrices.items():
        write_matrix(args.out_dir / f"{name}_matrix.csv", matrix)
        write_dicts(
            args.out_dir / f"{name}_pair_summary.csv", pair_summary(matrix, signs)
        )

    print(
        f"Q{args.query_id} {args.variant} CHECKPOINT SCORE CORRELATIONS; "
        f"signs from {split_count} splits"
    )
    print("SIGNS " + " ".join(f"{index + 1}:{'+' if sign > 0 else '-'}" for index, sign in enumerate(signs)))
    print("\nMETRIC            RAW ALL   FLIP ALL  RAW SAME RAW DIFF FLIP SAME FLIP DIFF")
    for metric in ("pearson", "spearman", "cosine"):
        raw = matrices[f"raw_{metric}"]
        flip = matrices[f"flipped_{metric}"]
        upper = np.triu_indices(len(raw), 1)
        same = signs[upper[0]] == signs[upper[1]]
        print(
            f"{metric:16s} {np.mean(raw[upper]):+8.4f} {np.mean(flip[upper]):+9.4f} "
            f"{np.mean(raw[upper][same]):+8.4f} {np.mean(raw[upper][~same]):+8.4f} "
            f"{np.mean(flip[upper][same]):+9.4f} {np.mean(flip[upper][~same]):+9.4f}"
        )

    means = np.mean(scores, axis=1)
    stds = np.std(scores, axis=1)
    positive = np.mean(scores > 0.0, axis=1)
    print("\nPER CHECKPOINT SCORE DISTRIBUTION")
    print("CKPT SIGN        MEAN         STD    POS%")
    distribution_rows = []
    for index in range(len(scores)):
        row = {
            "checkpoint": int(checkpoints[index]) + 1,
            "flip_sign": int(signs[index]),
            "mean": float(means[index]),
            "std": float(stds[index]),
            "positive_fraction": float(positive[index]),
        }
        distribution_rows.append(row)
        print(
            f"{row['checkpoint']:4d} {row['flip_sign']:+4d} "
            f"{row['mean']:+12.5e} {row['std']:11.5e} {row['positive_fraction']:7.3f}"
        )
    write_dicts(args.out_dir / "per_checkpoint_distribution.csv", distribution_rows)
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
