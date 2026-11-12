#!/usr/bin/env python3
"""Measure same-query overlap and cross-query uniqueness for DAS and PINT."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_das_vs_pint_top1000_endpoint_perceptual import (
    CELEBA_ROOT,
    CONFIGURATIONS,
    das_topk,
    parse_ids,
)
from analyze_five_method_cross_query_top1000_overlap import (
    load_queries,
    query_score_root,
)
from analyze_pint_raw_vs_adamw_topk_endpoint_clip import (
    score_directory,
    select_topk,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-file",
        type=Path,
        default=CELEBA_ROOT / "queries_attr40_val25_x4.json",
    )
    parser.add_argument("--query-ids", type=parse_ids, default=list(range(60)))
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if not 0 < args.topk <= 10_000:
        parser.error("--topk must be within [1, 10000]")
    records = load_queries(args.query_file)
    missing = sorted(set(args.query_ids) - set(records))
    if missing:
        parser.error(f"query IDs missing from query file: {missing}")

    result_root = CELEBA_ROOT / "result" / args.experiment
    output_dir = args.output_dir or (
        result_root
        / "analysis"
        / f"das_vs_pint_top{args.topk}_overlap_uniqueness_q0_q59"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    selections: dict[tuple[int, str], set[int]] = {}
    for position, query_id in enumerate(args.query_ids, start=1):
        record = records[query_id]
        prompt = str(record["prompt"])
        seed = int(record.get("initial_seed", record.get("seed")))
        root = query_score_root(result_root, args.train_seed, record)
        selections[(query_id, CONFIGURATIONS[0])] = set(
            map(
                int,
                das_topk(
                    root
                    / "das_50x1_mc1_full10k_ema"
                    / "score"
                    / "lambda_200",
                    args.topk,
                ),
            )
        )
        for label, variant in zip(CONFIGURATIONS[1:], ("raw", "query_train_l2")):
            directory = score_directory(
                result_root,
                args.train_seed,
                prompt,
                seed,
                "pint",
                "adamw_full",
                "timestamp_sum_squared",
                variant,
            )
            selections[(query_id, label)] = set(
                map(
                    int,
                    select_topk(
                        directory,
                        "pint",
                        "adamw_full",
                        "timestamp_sum_squared",
                        args.topk,
                    ),
                )
            )
        print(f"[select] {position}/{len(args.query_ids)} Q{query_id}", flush=True)

    overlap_rows: list[dict[str, object]] = []
    for query_id in args.query_ids:
        for left, right in combinations(CONFIGURATIONS, 2):
            left_set = selections[(query_id, left)]
            right_set = selections[(query_id, right)]
            overlap = len(left_set & right_set)
            overlap_rows.append(
                {
                    "query_id": query_id,
                    "left": left,
                    "right": right,
                    "topk": args.topk,
                    "overlap_count": overlap,
                    "overlap_fraction": overlap / args.topk,
                    "jaccard": overlap / (2 * args.topk - overlap),
                    "left_only": args.topk - overlap,
                    "right_only": args.topk - overlap,
                }
            )

    uniqueness_rows: list[dict[str, object]] = []
    for label in CONFIGURATIONS:
        sets = [selections[(query_id, label)] for query_id in args.query_ids]
        union = set().union(*sets)
        frequencies = np.zeros(10_000, dtype=np.int64)
        for selected in sets:
            frequencies[np.fromiter(selected, dtype=np.int64)] += 1
        pairwise = [
            len(left & right) / args.topk for left, right in combinations(sets, 2)
        ]
        selected_frequency = frequencies[frequencies > 0]
        singleton_count = int(np.count_nonzero(frequencies == 1))
        uniqueness_rows.append(
            {
                "configuration": label,
                "queries": len(args.query_ids),
                "topk": args.topk,
                "union_count": len(union),
                "union_coverage": len(union) / 10_000,
                "singleton_count": singleton_count,
                "singleton_occurrence_fraction": singleton_count
                / (len(args.query_ids) * args.topk),
                "mean_query_frequency_among_selected": float(
                    selected_frequency.mean()
                ),
                "mean_pairwise_overlap_fraction": statistics.fmean(pairwise),
                "median_pairwise_overlap_fraction": statistics.median(pairwise),
                "std_pairwise_overlap_fraction": statistics.pstdev(pairwise),
            }
        )

    lines = [
        "CELEBA DAS VS PINT TOP-1000 OVERLAP AND CROSS-QUERY UNIQUENESS",
        "=" * 80,
        f"queries={len(args.query_ids)} range={min(args.query_ids)}..{max(args.query_ids)}",
        f"topk={args.topk}; candidate universe=10000",
        "Same-query overlap is intersection/Top-K; random expectation is 10%.",
        "Union coverage and singleton fraction are larger when selections are more query-specific.",
        "Pairwise overlap is smaller when selections are more query-specific.",
        "",
        "SAME-QUERY METHOD OVERLAP",
        f"{'PAIR':62s} {'MEAN':>9s} {'MEDIAN':>9s} {'STD':>9s}",
        "-" * 93,
    ]
    for left, right in combinations(CONFIGURATIONS, 2):
        values = [
            float(row["overlap_fraction"])
            for row in overlap_rows
            if row["left"] == left and row["right"] == right
        ]
        lines.append(
            f"{left + ' vs ' + right:62s} "
            f"{statistics.fmean(values):9.3%} "
            f"{statistics.median(values):9.3%} "
            f"{statistics.pstdev(values):9.3%}"
        )

    lines.extend(
        [
            "",
            "CROSS-QUERY UNIQUENESS",
            f"{'CONFIGURATION':32s} {'UNION':>7s} {'COVER':>9s} "
            f"{'SINGLE':>7s} {'PAIR-OV':>9s}",
            "-" * 72,
        ]
    )
    for row in uniqueness_rows:
        lines.append(
            f"{str(row['configuration']):32s} "
            f"{int(row['union_count']):7d} "
            f"{float(row['union_coverage']):9.3%} "
            f"{int(row['singleton_count']):7d} "
            f"{float(row['mean_pairwise_overlap_fraction']):9.3%}"
        )

    report = "\n".join(lines) + "\n"
    write_csv(output_dir / "per_query_method_overlap.csv", overlap_rows)
    write_csv(output_dir / "cross_query_uniqueness.csv", uniqueness_rows)
    (output_dir / "report.txt").write_text(report, encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "query_ids": args.query_ids,
                "topk": args.topk,
                "candidate_universe": 10_000,
                "configurations": list(CONFIGURATIONS),
                "random_overlap_expectation": args.topk / 10_000,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n" + report, end="")
    print(f"[saved] {output_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
