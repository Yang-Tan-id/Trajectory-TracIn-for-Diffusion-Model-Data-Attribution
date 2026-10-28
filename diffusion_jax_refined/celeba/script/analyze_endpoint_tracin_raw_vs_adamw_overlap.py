#!/usr/bin/env python3
"""Compare CelebA PINT/TracIn raw-gradient and AdamW rankings."""

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


def parse_ids(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def load_queries(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return payload["queries"] if isinstance(payload, dict) else payload


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
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    left_ranks -= left_ranks.mean()
    right_ranks -= right_ranks.mean()
    denominator = float(np.linalg.norm(left_ranks) * np.linalg.norm(right_ranks))
    if denominator == 0.0:
        raise ValueError("cannot correlate a constant score bank")
    return float(np.dot(left_ranks, right_ranks) / denominator)


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
    parser.add_argument(
        "--method", choices=("tracin", "pint"), default="tracin"
    )
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    records = load_queries(args.query_file)
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
        / f"endpoint_{args.method}_raw_vs_adamw_full10k_top{args.topk}_overlap"
    )

    rows: list[dict[str, object]] = []
    for query_id in query_ids:
        record = records[query_id]
        prompt = str(record["prompt"])
        seed = int(record.get("initial_seed", record.get("seed")))
        query_root = (
            result_root
            / "attribution_score"
            / "prompted_solo"
            / f"train_seed_{args.train_seed}"
            / f"query_{_prompt_path_tag(prompt)}"
            / f"initial_seed_{seed}"
        )
        if args.method == "tracin":
            namespaces = {
                transform: (
                    f"traj_tracin_celeba_endpoint_tracin_{transform}_"
                    "train_mean50_query_mean50_linear_shape3d_semantics_full10000"
                )
                for transform in ("raw_gradient", "adamw_full")
            }
        else:
            namespaces = {
                transform: (
                    f"traj_tracin_celeba_endpoint_pollute_{transform}_"
                    "train1x50_query1x50_timestamp_sum_squared_"
                    "shape3d_semantics_full10000"
                )
                for transform in ("raw_gradient", "adamw_full")
            }
        for variant, directory in VARIANTS:
            raw_scores, raw_indices = load_bank(
                query_root / namespaces["raw_gradient"] / directory
            )
            adamw_scores, adamw_indices = load_bank(
                query_root / namespaces["adamw_full"] / directory
            )
            if not np.array_equal(raw_indices, adamw_indices):
                raise ValueError(f"Q{query_id} {variant}: candidate universes differ")

            raw_oriented = raw_scores
            if args.method == "tracin":
                # Linear AdamW TracIn includes the descent-direction minus
                # sign, so its stored scores rank ascending.
                adamw_oriented = -adamw_scores
            else:
                # PINT stores timestamp-sum-squared scores; both transforms
                # rank descending because squaring removes the update sign.
                adamw_oriented = adamw_scores
            raw_order = np.argsort(-raw_oriented, kind="stable")[: args.topk]
            adamw_order = np.argsort(-adamw_oriented, kind="stable")[: args.topk]
            raw_top = set(map(int, raw_indices[raw_order]))
            adamw_top = set(map(int, adamw_indices[adamw_order]))
            overlap = len(raw_top & adamw_top)
            rows.append(
                {
                    "query_id": query_id,
                    "variant": variant,
                    "topk": args.topk,
                    "candidate_count": len(raw_indices),
                    "random_overlap_expectation": args.topk / len(raw_indices),
                    "overlap_count": overlap,
                    "overlap_fraction": overlap / args.topk,
                    "jaccard": overlap / (2 * args.topk - overlap),
                    "ranking_spearman": spearman(raw_oriented, adamw_oriented),
                }
            )
        print(f"[query] Q{query_id} compared 4/4 variants", flush=True)

    summary_rows: list[dict[str, object]] = []
    for variant, _ in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        overlap = summarize([float(row["overlap_fraction"]) for row in selected])
        jaccard = summarize([float(row["jaccard"]) for row in selected])
        rank_correlation = summarize(
            [float(row["ranking_spearman"]) for row in selected]
        )
        summary_rows.append(
            {
                "variant": variant,
                "queries": len(selected),
                "topk": args.topk,
                "candidate_count": 10_000,
                "random_overlap_expectation": args.topk / 10_000,
                **{f"overlap_{key}": value for key, value in overlap.items()},
                **{f"jaccard_{key}": value for key, value in jaccard.items()},
                **{
                    f"spearman_{key}": value
                    for key, value in rank_correlation.items()
                },
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(output_dir / "per_query_overlap.csv", rows)
    atomic_csv(output_dir / "summary_overlap.csv", summary_rows)

    lines = [
        (
            "CELEBA FULL-10K "
            f"{'PINT' if args.method == 'pint' else 'ENDPOINT-TRACIN'} — "
            "RAW-GRADIENT VS ADAMW-FULL"
        ),
        f"queries={len(query_ids)} topk={args.topk} candidates=10000",
        f"random independent top-k expectation={args.topk / 10_000:.2%}",
        (
            "raw-gradient and AdamW timestamp-sum-squared scores both rank "
            "descending."
            if args.method == "pint"
            else "raw-gradient ranks stored linear score descending; "
            "AdamW ranks ascending."
        ),
        "Spearman uses the common larger-is-better orientation.",
        "",
        (
            f"{'VARIANT':16s} {'OVERLAP MEAN':>13s} {'MEDIAN':>9s} "
            f"{'MIN':>9s} {'MAX':>9s} {'JACCARD':>10s} "
            f"{'SPEARMAN':>10s}"
        ),
        "-" * 92,
    ]
    for row in summary_rows:
        lines.append(
            f"{str(row['variant']):16s} "
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
