#!/usr/bin/env python3
"""Compare Endpoint-Pollute top-k sets between two score reductions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from dataset_config import _prompt_tag


VARIANTS = (
    ("raw", "score"),
    ("query_l2", "score_query_normalized"),
    ("train_l2", "score_train_l2_normalized"),
    ("query_train_l2", "score_query_train_l2_normalized"),
)


def parse_ints(text: str) -> list[int]:
    values = [int(token) for token in text.replace(",", " ").split()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("values must be nonempty and unique")
    return values


def score_root(
    result_root: Path,
    train_seed: int,
    record: dict[str, object],
    reduction: str,
) -> Path:
    seed = int(record.get("initial_seed", record.get("seed")))
    namespace = (
        "recreate_adamw_full_polluted_endpoint_delta_l2normalized_"
        f"{reduction}_aligned100x1_q0_99"
    )
    return (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
        / f"query_{_prompt_tag(str(record['prompt']))}"
        / f"initial_seed_{seed}"
        / f"traj_tracin_{namespace}"
    )


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(np.load(path / "scores.npy"), dtype=np.float64).reshape(-1)
    indices = np.asarray(
        np.load(path / "score_indices.npy"), dtype=np.int64
    ).reshape(-1)
    if scores.shape != indices.shape:
        raise ValueError(f"score/index mismatch under {path}")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"duplicate indices under {path}")
    return scores, indices


def descending_topk(scores: np.ndarray, indices: np.ndarray, topk: int) -> np.ndarray:
    ranking = np.where(np.isfinite(scores), scores, -np.inf)
    order = np.argsort(-ranking, kind="stable")[:topk]
    if len(order) != topk or np.any(~np.isfinite(ranking[order])):
        raise ValueError(f"fewer than {topk} finite scores")
    return indices[order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-file",
        type=Path,
        default=SHAPES_ROOT / "queries_in_distribution_plus_zero_seed_100_219.json",
    )
    parser.add_argument("--query-ids", type=parse_ints, default=list(range(100)))
    parser.add_argument("--topks", type=parse_ints, default=[400, 1000])
    parser.add_argument("--left", default="linear")
    parser.add_argument("--right", default="timestamp_sum_squared")
    parser.add_argument("--left-variant", choices=[value[0] for value in VARIANTS])
    parser.add_argument("--right-variant", choices=[value[0] for value in VARIANTS])
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records = json.loads(args.query_file.read_text())["queries"]
    result_root = SHAPES_ROOT / "result" / args.experiment
    rows: list[dict[str, object]] = []
    variant_directories = dict(VARIANTS)
    if (args.left_variant is None) != (args.right_variant is None):
        parser.error("--left-variant and --right-variant must be provided together")
    variant_pairs = (
        [(args.left_variant, args.right_variant)]
        if args.left_variant is not None
        else [(variant, variant) for variant, _ in VARIANTS]
    )

    for left_variant, right_variant in variant_pairs:
        left_directory = variant_directories[left_variant]
        right_directory = variant_directories[right_variant]
        pair_label = f"{left_variant}_vs_{right_variant}"
        for position, query_id in enumerate(args.query_ids, start=1):
            record = records[query_id]
            left_scores, left_indices = load(
                score_root(
                    result_root, args.train_seed, record, args.left
                ) / left_directory
            )
            right_scores, right_indices = load(
                score_root(
                    result_root, args.train_seed, record, args.right
                ) / right_directory
            )
            if not np.array_equal(left_indices, right_indices):
                raise ValueError(f"Q{query_id} {pair_label}: candidate index order differs")
            for topk in args.topks:
                left_top = descending_topk(left_scores, left_indices, topk)
                right_top = descending_topk(right_scores, right_indices, topk)
                overlap_count = int(np.intersect1d(left_top, right_top).size)
                rows.append(
                    {
                        "left_variant": left_variant,
                        "right_variant": right_variant,
                        "query": query_id,
                        "topk": topk,
                        "overlap_count": overlap_count,
                        "overlap_fraction": overlap_count / topk,
                    }
                )
            if position % 10 == 0 or position == len(args.query_ids):
                print(f"[load] {pair_label}: {position}/{len(args.query_ids)} queries")

    pair_tag = (
        f"_{args.left_variant}_vs_{args.right_variant}"
        if args.left_variant is not None
        else ""
    )
    output = args.output or (
        result_root / "analysis" /
        f"endpoint_pollute_{args.left}_vs_{args.right}{pair_tag}_topk_overlap.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("ENDPOINT-POLLUTE REDUCTION TOP-K OVERLAP")
    print(f"left={args.left}; right={args.right}")
    print("ranking: stored score descending on both sides")
    print("overlap = |left top-K ∩ right top-K| / K")
    print()
    print("VARIANT PAIR                    TOPK      MEAN    MEDIAN       MIN       MAX")
    print("-" * 78)
    for left_variant, right_variant in variant_pairs:
        pair_label = f"{left_variant} vs {right_variant}"
        for topk in args.topks:
            values = np.asarray(
                [
                    float(row["overlap_fraction"])
                    for row in rows
                    if row["left_variant"] == left_variant
                    and row["right_variant"] == right_variant
                    and row["topk"] == topk
                ]
            )
            print(
                f"{pair_label:<31s} {topk:4d} "
                f"{values.mean():9.2%} {np.median(values):9.2%} "
                f"{values.min():9.2%} {values.max():9.2%}"
            )
    print(f"\n[saved] {output}")


if __name__ == "__main__":
    main()
