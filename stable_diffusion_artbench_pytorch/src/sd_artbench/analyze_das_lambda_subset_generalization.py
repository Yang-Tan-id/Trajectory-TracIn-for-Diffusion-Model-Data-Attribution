#!/usr/bin/env python3
"""Measure DAS lambda stability across fixed query subsets."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


METRICS = (
    ("CLIP cosine", "clip_cosine", True),
    ("DINO cosine", "dino_cosine", True),
    ("Pixel RMSE", "pixel_rmse", False),
    ("LPIPS", "lpips", False),
)
PATTERN = re.compile(r"^DAS lambda=(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-query-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-queries", type=int, default=60)
    parser.add_argument("--num-subsets", type=int, default=6)
    parser.add_argument("--partition-seed", type=int, default=2026)
    parser.add_argument("--fixed-lambda", type=float, default=1000.0)
    return parser.parse_args()


def load_data(path: Path) -> dict[tuple[int, float], dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    data: dict[tuple[int, float], dict[str, float]] = {}
    for row in rows:
        match = PATTERN.match(row["configuration"])
        if match is None:
            continue
        data[(int(row["query_index"]), float(match.group(1)))] = {
            metric: float(row[metric]) for _label, metric, _higher in METRICS
        }
    if not data:
        raise ValueError(f"no DAS lambda records in {path}")
    return data


def mean_by_lambda(
    data: dict[tuple[int, float], dict[str, float]],
    query_ids: list[int],
    lambdas: list[float],
    metric: str,
) -> dict[float, float]:
    return {
        damping: float(
            np.mean([data[(query, damping)][metric] for query in query_ids])
        )
        for damping in lambdas
    }


def ranking(
    values: dict[float, float], lambdas: list[float], higher_is_better: bool
) -> list[float]:
    return sorted(
        lambdas,
        key=lambda damping: values[damping],
        reverse=higher_is_better,
    )


def build_report(args: argparse.Namespace) -> str:
    if args.expected_queries < 2:
        raise ValueError("expected-queries must be at least two")
    if not 2 <= args.num_subsets <= args.expected_queries:
        raise ValueError("num-subsets must be between two and expected-queries")

    data = load_data(args.per_query_csv)
    queries = list(range(args.expected_queries))
    lambdas = sorted({damping for _query, damping in data})
    if args.fixed_lambda not in lambdas:
        raise ValueError(f"fixed lambda {args.fixed_lambda:g} is unavailable")
    missing = [
        (query, damping)
        for query in queries
        for damping in lambdas
        if (query, damping) not in data
    ]
    if missing:
        raise ValueError(f"missing {len(missing)} query/lambda records")

    generator = np.random.default_rng(args.partition_seed)
    permutation = generator.permutation(queries)
    folds = [
        sorted(map(int, fold))
        for fold in np.array_split(permutation, args.num_subsets)
    ]
    lines = [
        "ARTBENCH DAS LAMBDA GENERALIZATION — PERCEPTUAL TOP-1000",
        "=" * 112,
        (
            f"queries={args.expected_queries}; subsets={args.num_subsets}; "
            f"partition_seed={args.partition_seed}; fixed_lambda={args.fixed_lambda:g}"
        ),
        "No removal and no retraining. LOO selects lambda on all other subsets.",
        "",
    ]
    for index, fold in enumerate(folds, 1):
        lines.append(f"SUBSET {index}: " + ",".join(f"Q{query}" for query in fold))
    lines.append("")

    for label, metric, higher_is_better in METRICS:
        lines.extend(
            [
                label.upper(),
                "-" * 112,
                (
                    f"{'SUBSET':<9} {'SUBSET BEST':>13} {'FIXED RANK':>12} "
                    f"{'FIXED VALUE':>14} {'BEST VALUE':>14} {'GAP':>14} "
                    f"{'LOO SELECT':>12} {'LOO HELDOUT':>14}"
                ),
            ]
        )
        subset_bests: list[float] = []
        loo_choices: list[float] = []
        for fold_index, heldout in enumerate(folds, 1):
            training = sorted(set(queries) - set(heldout))
            heldout_values = mean_by_lambda(data, heldout, lambdas, metric)
            heldout_order = ranking(heldout_values, lambdas, higher_is_better)
            heldout_best = heldout_order[0]
            subset_bests.append(heldout_best)
            fixed_rank = heldout_order.index(args.fixed_lambda) + 1
            fixed_value = heldout_values[args.fixed_lambda]
            best_value = heldout_values[heldout_best]
            gap = (
                best_value - fixed_value
                if higher_is_better
                else fixed_value - best_value
            )
            training_values = mean_by_lambda(data, training, lambdas, metric)
            selected = ranking(training_values, lambdas, higher_is_better)[0]
            loo_choices.append(selected)
            lines.append(
                f"{fold_index:<9d}{heldout_best:>13g}{fixed_rank:>12d}"
                f"{fixed_value:>14.6f}{best_value:>14.6f}{gap:>14.6f}"
                f"{selected:>12g}{heldout_values[selected]:>14.6f}"
            )

        full_values = mean_by_lambda(data, queries, lambdas, metric)
        full_order = ranking(full_values, lambdas, higher_is_better)
        lines.extend(
            [
                "",
                f"Full-query best lambda: {full_order[0]:g}",
                (
                    f"Fixed lambda full-query rank: "
                    f"{full_order.index(args.fixed_lambda) + 1}/{len(lambdas)}"
                ),
                "Subset best lambdas: "
                + ", ".join(f"{value:g}" for value in subset_bests),
                "LOO-selected lambdas: "
                + ", ".join(f"{value:g}" for value in loo_choices),
                (
                    f"Fixed lambda subset-best count: "
                    f"{sum(value == args.fixed_lambda for value in subset_bests)}"
                    f"/{args.num_subsets}"
                ),
                (
                    f"Fixed lambda LOO-selection count: "
                    f"{sum(value == args.fixed_lambda for value in loo_choices)}"
                    f"/{args.num_subsets}"
                ),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
