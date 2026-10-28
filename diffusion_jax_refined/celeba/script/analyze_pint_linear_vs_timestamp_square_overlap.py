#!/usr/bin/env python3
"""Compare CelebA PINT linear and timestamp-sum-squared rankings."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np


CELEBA_ROOT = Path(__file__).resolve().parents[1]
if str(CELEBA_ROOT) not in sys.path:
    sys.path.insert(0, str(CELEBA_ROOT))

from dataset_config_attr40 import _prompt_path_tag


VARIANTS = (
    ("raw", "score"),
    ("query_l2", "score_query_normalized"),
    ("train_l2", "score_train_l2_normalized"),
    ("query_train_l2", "score_query_train_l2_normalized"),
)
TRANSFORMS = ("raw_gradient", "adamw_full")


def parse_ids(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def load_bank(path: Path) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(np.load(path / "scores.npy"), dtype=np.float64).reshape(-1)
    indices = np.asarray(
        np.load(path / "score_indices.npy"), dtype=np.int64
    ).reshape(-1)
    expected = np.arange(10_000, dtype=np.int64)
    if scores.shape != expected.shape or not np.array_equal(indices, expected):
        raise ValueError(f"bank is not the complete ordered full-10k universe: {path}")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"non-finite scores under {path}")
    return scores, indices


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
    if denominator == 0.0:
        raise ValueError("cannot correlate a constant score bank")
    return float(np.dot(left_rank, right_rank) / denominator)


def orient_linear_scores(scores: np.ndarray, transform: str) -> np.ndarray:
    """Map stored PINT linear scores to a common larger-is-better axis."""
    if transform == "raw_gradient":
        return -scores
    if transform == "adamw_full":
        return scores
    raise ValueError(f"unknown transform: {transform}")


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-file",
        type=Path,
        default=CELEBA_ROOT / "queries_attr40_val25_x4.json",
    )
    parser.add_argument("--query-ids", default=",".join(map(str, range(72))))
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.query_file.read_text())
    records = payload["queries"] if isinstance(payload, dict) else payload
    query_ids = parse_ids(args.query_ids)
    if not query_ids or len(query_ids) != len(set(query_ids)):
        parser.error("--query-ids must contain unique query IDs")
    if min(query_ids) < 0 or max(query_ids) >= len(records):
        parser.error(f"query IDs must be within [0, {len(records) - 1}]")
    if not 0 < args.topk <= 10_000:
        parser.error("--topk must be within [1, 10000]")

    result_root = CELEBA_ROOT / "result" / args.experiment
    output_dir = args.output_dir or (
        result_root
        / "analysis"
        / f"pint_linear_vs_timestamp_square_full10k_top{args.topk}_overlap"
    )

    rows: list[dict[str, object]] = []
    for query_id in query_ids:
        record = records[query_id]
        seed = int(record.get("initial_seed", record.get("seed")))
        query_root = (
            result_root
            / "attribution_score"
            / "prompted_solo"
            / f"train_seed_{args.train_seed}"
            / f"query_{_prompt_path_tag(str(record['prompt']))}"
            / f"initial_seed_{seed}"
        )
        for transform in TRANSFORMS:
            namespaces = {
                reduction: (
                    f"traj_tracin_celeba_endpoint_pollute_{transform}_"
                    f"train1x50_query1x50_{reduction}_"
                    "shape3d_semantics_full10000"
                )
                for reduction in ("linear", "timestamp_sum_squared")
            }
            for variant, directory in VARIANTS:
                linear_scores, linear_indices = load_bank(
                    query_root / namespaces["linear"] / directory
                )
                square_scores, square_indices = load_bank(
                    query_root / namespaces["timestamp_sum_squared"] / directory
                )
                if not np.array_equal(linear_indices, square_indices):
                    raise ValueError(
                        f"Q{query_id} {transform}/{variant}: universes differ"
                    )

                # The stored endpoint-Pollute linear score has a transform-specific
                # sign: raw-gradient ranks ascending, while AdamW ranks descending.
                linear_oriented = orient_linear_scores(linear_scores, transform)
                # Timestamp-sum-squared scores rank descending for both.
                square_oriented = square_scores
                linear_order = np.argsort(-linear_oriented, kind="stable")[: args.topk]
                square_order = np.argsort(-square_oriented, kind="stable")[: args.topk]
                linear_top = set(map(int, linear_indices[linear_order]))
                square_top = set(map(int, square_indices[square_order]))
                overlap = len(linear_top & square_top)
                rows.append(
                    {
                        "query_id": query_id,
                        "transform": transform,
                        "variant": variant,
                        "topk": args.topk,
                        "candidate_count": len(linear_indices),
                        "random_overlap_expectation": args.topk / len(linear_indices),
                        "overlap_count": overlap,
                        "overlap_fraction": overlap / args.topk,
                        "jaccard": overlap / (2 * args.topk - overlap),
                        "ranking_spearman": spearman(
                            linear_oriented, square_oriented
                        ),
                    }
                )
        print(f"[query] Q{query_id} compared 8/8 configurations", flush=True)

    summary_rows: list[dict[str, object]] = []
    for transform in TRANSFORMS:
        for variant, _ in VARIANTS:
            selected = [
                row
                for row in rows
                if row["transform"] == transform and row["variant"] == variant
            ]
            overlap = summarize(
                [float(row["overlap_fraction"]) for row in selected]
            )
            jaccard = summarize([float(row["jaccard"]) for row in selected])
            correlation = summarize(
                [float(row["ranking_spearman"]) for row in selected]
            )
            summary_rows.append(
                {
                    "transform": transform,
                    "variant": variant,
                    "queries": len(selected),
                    "topk": args.topk,
                    "candidate_count": 10_000,
                    "random_overlap_expectation": args.topk / 10_000,
                    **{f"overlap_{key}": value for key, value in overlap.items()},
                    **{f"jaccard_{key}": value for key, value in jaccard.items()},
                    **{f"spearman_{key}": value for key, value in correlation.items()},
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(output_dir / "per_query_overlap.csv", rows)
    atomic_csv(output_dir / "summary_overlap.csv", summary_rows)
    lines = [
        "CELEBA FULL-10K PINT — LINEAR VS TIMESTAMP-SUM-SQUARED",
        f"queries={len(query_ids)} topk={args.topk} candidates=10000",
        f"random independent top-k expectation={args.topk / 10_000:.2%}",
        (
            "linear ranks raw-gradient ascending and AdamW descending; "
            "timestamp-sum-squared ranks descending for both."
        ),
        "Spearman uses the common larger-is-better orientation.",
        "",
        (
            f"{'TRANSFORM':12s} {'VARIANT':16s} {'OVERLAP MEAN':>13s} "
            f"{'MEDIAN':>9s} {'MIN':>9s} {'MAX':>9s} "
            f"{'JACCARD':>10s} {'SPEARMAN':>10s}"
        ),
        "-" * 108,
    ]
    for row in summary_rows:
        lines.append(
            f"{str(row['transform']):12s} {str(row['variant']):16s} "
            f"{float(row['overlap_mean']):13.2%} "
            f"{float(row['overlap_median']):9.2%} "
            f"{float(row['overlap_min']):9.2%} "
            f"{float(row['overlap_max']):9.2%} "
            f"{float(row['jaccard_mean']):10.2%} "
            f"{float(row['spearman_mean']):+10.4f}"
        )
    report = "\n".join(lines) + "\n"
    report_path = output_dir / "overlap_report.txt"
    report_path.write_text(report)
    print()
    print(report, end="")
    print(f"[saved] {report_path}")
    print(f"[saved] {output_dir / 'summary_overlap.csv'}")
    print(f"[saved] {output_dir / 'per_query_overlap.csv'}")


if __name__ == "__main__":
    main()
