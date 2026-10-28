#!/usr/bin/env python3
"""Compare full-10k CelebA endpoint-Pollute and endpoint-TracIn rankings."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

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


def parse_values(text: str) -> list[str]:
    return [value.strip() for value in text.replace(" ", ",").split(",") if value.strip()]


def load_queries(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text())
    records = payload["queries"] if isinstance(payload, dict) else payload
    return {
        int(record.get("query_id", record.get("id", position))): record
        for position, record in enumerate(records)
    }


def query_seed(record: dict) -> int:
    return int(record.get("initial_seed", record.get("seed")))


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fields = list(rows[0]) if rows else ["query_id"]
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_bank(path: Path) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(np.load(path / "scores.npy"), dtype=np.float64).reshape(-1)
    indices = np.asarray(
        np.load(path / "score_indices.npy"), dtype=np.int64
    ).reshape(-1)
    expected = np.arange(10_000, dtype=np.int64)
    if scores.shape != expected.shape or not np.array_equal(indices, expected):
        raise ValueError(f"bank is not the complete ordered 10k universe: {path}")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"non-finite scores under {path}")
    return scores, indices


def rank_average(values: np.ndarray) -> np.ndarray:
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
    left_rank = rank_average(left)
    right_rank = rank_average(right)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(left_rank, right_rank) / denominator)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument(
        "--query-ids", default=",".join(str(value) for value in range(72))
    )
    parser.add_argument(
        "--transforms", default="raw_gradient,adamw_full"
    )
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    query_ids = [int(value) for value in parse_values(args.query_ids)]
    transforms = parse_values(args.transforms)
    unknown = sorted(set(transforms) - {"raw_gradient", "adamw_full"})
    if unknown:
        parser.error(f"unknown transforms: {unknown}")
    records = load_queries(args.query_file)
    result_root = CELEBA_ROOT / "result" / args.experiment
    output_dir = args.output_dir or (
        result_root / "analysis" / f"endpoint_full10k_cross_method_top{args.topk}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for query_id in query_ids:
        record = records[query_id]
        query_root = (
            result_root
            / "attribution_score"
            / "prompted_solo"
            / f"train_seed_{args.train_seed}"
            / f"query_{_prompt_path_tag(str(record['prompt']))}"
            / f"initial_seed_{query_seed(record)}"
        )
        for transform in transforms:
            pollute_namespace = (
                f"celeba_endpoint_pollute_{transform}_train1x50_query1x50_"
                "timestamp_sum_squared_shape3d_semantics_full10000"
            )
            tracin_namespace = (
                f"celeba_endpoint_tracin_{transform}_"
                "train_mean50_query_mean50_linear_shape3d_semantics_full10000"
            )
            for variant, directory in VARIANTS:
                pollute_scores, pollute_indices = load_bank(
                    query_root / f"traj_tracin_{pollute_namespace}" / directory
                )
                tracin_scores, tracin_indices = load_bank(
                    query_root / f"traj_tracin_{tracin_namespace}" / directory
                )
                if not np.array_equal(pollute_indices, tracin_indices):
                    raise ValueError(
                        f"Q{query_id} {transform}/{variant}: candidate universes differ"
                    )

                pollute_ranking = pollute_scores
                if transform == "raw_gradient":
                    # Raw-gradient linear TracIn ranks stored scores descending.
                    tracin_ranking = tracin_scores
                else:
                    # AdamW linear TracIn ranks stored scores ascending because
                    # the optimizer update contains the descent minus sign.
                    tracin_ranking = -tracin_scores
                pollute_order = np.argsort(-pollute_ranking, kind="stable")[: args.topk]
                tracin_order = np.argsort(-tracin_ranking, kind="stable")[: args.topk]
                pollute_top = set(map(int, pollute_indices[pollute_order]))
                tracin_top = set(map(int, tracin_indices[tracin_order]))
                overlap = len(pollute_top & tracin_top)
                rows.append(
                    {
                        "query_id": query_id,
                        "transform": transform,
                        "variant": variant,
                        "topk": args.topk,
                        "candidate_count": len(pollute_indices),
                        "random_overlap_expectation": args.topk / len(pollute_indices),
                        "overlap_count": overlap,
                        "overlap_fraction": overlap / args.topk,
                        "jaccard": overlap / (2 * args.topk - overlap),
                        "ranking_spearman": spearman(
                            pollute_ranking, tracin_ranking
                        ),
                    }
                )
        print(f"[query] Q{query_id} compared 8/8 configurations", flush=True)

    atomic_csv(output_dir / "per_query_cross_method_overlap.csv", rows)
    summary_rows: list[dict[str, object]] = []
    for transform in transforms:
        for variant, _ in VARIANTS:
            selected = [
                row
                for row in rows
                if row["transform"] == transform and row["variant"] == variant
            ]
            overlap_summary = summarize(
                [float(row["overlap_fraction"]) for row in selected]
            )
            spearman_summary = summarize(
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
                    **{
                        f"overlap_{key}": value
                        for key, value in overlap_summary.items()
                    },
                    **{
                        f"spearman_{key}": value
                        for key, value in spearman_summary.items()
                    },
                }
            )
    atomic_csv(output_dir / "summary_cross_method_overlap.csv", summary_rows)

    lines = [
        "CELEBA FULL-10K ENDPOINT-POLLUTE vs ENDPOINT-TRACIN",
        f"queries={len(query_ids)} topk={args.topk} candidates=10000",
        f"random independent top-k overlap expectation={args.topk / 10_000:.2%}",
        (
            "Pollute raw/AdamW rank descending; TracIn raw-gradient ranks "
            "descending and AdamW ranks ascending."
        ),
        "",
        (
            f"{'TRANSFORM':12s} {'VARIANT':16s} {'OVERLAP MEAN':>13s} "
            f"{'MEDIAN':>9s} {'MIN':>9s} {'MAX':>9s} "
            f"{'SPEARMAN MEAN':>14s} {'MEDIAN':>9s}"
        ),
        "-" * 105,
    ]
    for row in summary_rows:
        lines.append(
            f"{str(row['transform']):12s} {str(row['variant']):16s} "
            f"{float(row['overlap_mean']):13.2%} "
            f"{float(row['overlap_median']):9.2%} "
            f"{float(row['overlap_min']):9.2%} "
            f"{float(row['overlap_max']):9.2%} "
            f"{float(row['spearman_mean']):+14.4f} "
            f"{float(row['spearman_median']):+9.4f}"
        )
    report = "\n".join(lines) + "\n"
    report_path = output_dir / "cross_method_overlap_report.txt"
    report_path.write_text(report)
    print()
    print(report, end="")
    print(f"[saved] {output_dir / 'per_query_cross_method_overlap.csv'}")
    print(f"[saved] {output_dir / 'summary_cross_method_overlap.csv'}")
    print(f"[saved] {report_path}")


if __name__ == "__main__":
    main()
