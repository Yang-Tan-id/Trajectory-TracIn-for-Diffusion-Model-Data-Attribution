#!/usr/bin/env python3
"""Measure CelebA DAS lambda stability across fixed query subsets using LDS."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lds-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-queries", type=int, default=100)
    parser.add_argument("--num-subsets", type=int, default=10)
    parser.add_argument("--partition-seed", type=int, default=2026)
    parser.add_argument(
        "--fixed-lambda",
        type=float,
        default=None,
        help="Override target-specific full-query best lambda.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"query_id", "lambda", "target_function", "lds_percent"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"invalid DAS LDS CSV: {path}")
    return rows


def mean_by_lambda(
    data: dict[tuple[int, str, float], float],
    query_ids: list[int],
    target: str,
    lambdas: list[float],
) -> dict[float, float]:
    return {
        damping: float(
            np.mean([data[(query, target, damping)] for query in query_ids])
        )
        for damping in lambdas
    }


def build_report(args: argparse.Namespace) -> str:
    if args.expected_queries < 2:
        raise ValueError("expected-queries must be at least two")
    if not 2 <= args.num_subsets <= args.expected_queries:
        raise ValueError("num-subsets must be between two and expected-queries")
    rows = read_rows(args.lds_csv)
    data = {
        (int(row["query_id"]), row["target_function"], float(row["lambda"])): float(
            row["lds_percent"]
        )
        for row in rows
    }
    queries = list(range(args.expected_queries))
    targets = sorted({row["target_function"] for row in rows})
    lambdas = sorted({float(row["lambda"]) for row in rows})
    if not targets or not lambdas:
        raise ValueError("no targets or lambdas found")
    if args.fixed_lambda is not None and args.fixed_lambda not in lambdas:
        raise ValueError(f"fixed lambda {args.fixed_lambda:g} is unavailable")
    missing = [
        (query, target, damping)
        for query in queries
        for target in targets
        for damping in lambdas
        if (query, target, damping) not in data
    ]
    if missing:
        raise ValueError(f"missing {len(missing)} query/target/lambda records")

    generator = np.random.default_rng(args.partition_seed)
    folds = [
        sorted(map(int, fold))
        for fold in np.array_split(generator.permutation(queries), args.num_subsets)
    ]
    lines = [
        "CELEBA DAS LAMBDA GENERALIZATION — LDS",
        "=" * 112,
        (
            f"queries={args.expected_queries}; subsets={args.num_subsets}; "
            f"partition_seed={args.partition_seed}; LDS subset models=192"
        ),
        (
            "Fixed lambda is target-specific full-query best."
            if args.fixed_lambda is None
            else f"Fixed lambda override={args.fixed_lambda:g}."
        ),
        "LOO selects lambda using all queries outside the held-out query subset.",
        "",
    ]
    for index, fold in enumerate(folds, 1):
        lines.append(f"SUBSET {index}: " + ",".join(f"Q{query}" for query in fold))
    lines.append("")

    for target in targets:
        full_values = mean_by_lambda(data, queries, target, lambdas)
        full_order = sorted(lambdas, key=full_values.__getitem__, reverse=True)
        fixed_lambda = (
            full_order[0] if args.fixed_lambda is None else args.fixed_lambda
        )
        lines.extend(
            [
                target.upper(),
                "-" * 112,
                f"Full-query best lambda={full_order[0]:g}; fixed lambda={fixed_lambda:g}",
                (
                    f"{'SUBSET':<9} {'SUBSET BEST':>13} {'FIXED RANK':>12} "
                    f"{'FIXED LDS':>13} {'BEST LDS':>13} {'GAP (PP)':>12} "
                    f"{'LOO SELECT':>12} {'HELDOUT LDS':>13}"
                ),
            ]
        )
        subset_bests: list[float] = []
        loo_choices: list[float] = []
        for fold_index, heldout in enumerate(folds, 1):
            training = sorted(set(queries) - set(heldout))
            heldout_values = mean_by_lambda(data, heldout, target, lambdas)
            heldout_order = sorted(
                lambdas, key=heldout_values.__getitem__, reverse=True
            )
            subset_best = heldout_order[0]
            subset_bests.append(subset_best)
            training_values = mean_by_lambda(data, training, target, lambdas)
            loo_selected = max(lambdas, key=training_values.__getitem__)
            loo_choices.append(loo_selected)
            lines.append(
                f"{fold_index:<9d}{subset_best:>13g}"
                f"{heldout_order.index(fixed_lambda) + 1:>12d}"
                f"{heldout_values[fixed_lambda]:>13.4f}"
                f"{heldout_values[subset_best]:>13.4f}"
                f"{heldout_values[subset_best] - heldout_values[fixed_lambda]:>12.4f}"
                f"{loo_selected:>12g}{heldout_values[loo_selected]:>13.4f}"
            )
        lines.extend(
            [
                "",
                f"Fixed lambda full-query rank: {full_order.index(fixed_lambda) + 1}/{len(lambdas)}",
                "Subset best lambdas: "
                + ", ".join(f"{value:g}" for value in subset_bests),
                "LOO-selected lambdas: "
                + ", ".join(f"{value:g}" for value in loo_choices),
                (
                    f"Fixed lambda subset-best count: "
                    f"{sum(value == fixed_lambda for value in subset_bests)}"
                    f"/{args.num_subsets}"
                ),
                (
                    f"Fixed lambda LOO-selection count: "
                    f"{sum(value == fixed_lambda for value in loo_choices)}"
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
