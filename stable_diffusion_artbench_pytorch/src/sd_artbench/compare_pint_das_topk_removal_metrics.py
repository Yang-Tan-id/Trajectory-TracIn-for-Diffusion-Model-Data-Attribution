#!/usr/bin/env python3
"""Compare PINT and DAS removal metrics assembled from one or more shards."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel


METRICS = (
    ("Endpoint MSE", "endpoint_mse", "trajectory"),
    ("Endpoint RMSE", "endpoint_rmse", "trajectory"),
    ("Trajectory MSE", "trajectory_mse", "trajectory"),
    ("Trajectory RMSE", "trajectory_rmse", "trajectory"),
    ("Pixel-L2", "pixel_l2", "perceptual"),
    ("CLIP distance", "clip_distance", "perceptual"),
    ("LPIPS", "lpips", "perceptual"),
    ("DINO distance", "dino_distance", "perceptual"),
)


def parse_ids(text: str) -> list[int]:
    values: list[int] = []
    for token in text.replace(",", " ").split():
        if "-" in token[1:]:
            left, right = token.split("-", 1)
            values.extend(range(int(left), int(right) + 1))
        else:
            values.append(int(token))
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("query IDs must be nonempty and unique")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument(
        "--das-trajectory-csv", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--das-perceptual-csv", type=Path, action="append", required=True
    )
    parser.add_argument("--query-ids", type=parse_ids, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pint-method", default="endpoint_pollute_query_train_l2"
    )
    parser.add_argument("--das-method", default="das_lambda_1000")
    return parser.parse_args()


def read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select(
    paths: list[Path], method: str, query_ids: set[int]
) -> dict[int, dict[str, str]]:
    selected: dict[int, dict[str, str]] = {}
    for path in paths:
        for row in read(path):
            query = int(row["query_index"])
            if row["method"] == method and query in query_ids:
                selected[query] = row
    return selected


def build_report(args: argparse.Namespace) -> str:
    query_ids = set(args.query_ids)
    baseline_trajectory = args.baseline_dir / "endpoint_trajectory_metrics.csv"
    baseline_perceptual = args.baseline_dir / "perceptual_metrics.csv"
    pint_trajectory = select(
        [baseline_trajectory], args.pint_method, query_ids
    )
    pint_perceptual = select(
        [baseline_perceptual], args.pint_method, query_ids
    )
    das_trajectory = select(
        args.das_trajectory_csv, args.das_method, query_ids
    )
    das_perceptual = select(
        args.das_perceptual_csv, args.das_method, query_ids
    )
    groups = {
        "PINT trajectory": pint_trajectory,
        "PINT perceptual": pint_perceptual,
        "DAS trajectory": das_trajectory,
        "DAS perceptual": das_perceptual,
    }
    for label, rows in groups.items():
        missing = sorted(query_ids - set(rows))
        if missing:
            raise ValueError(f"{label} missing queries: {missing}")

    ordered = sorted(query_ids)
    lines = [
        f"ARTBENCH Q{ordered[0]}-Q{ordered[-1]} TOP-1000 REMOVAL — PINT VS DAS",
        "=" * 116,
        f"queries={len(ordered)}; DAS lambda=1000; removal=1000/5000",
        "Larger values indicate a larger counterfactual removal effect.",
        "Paired t-test uses DAS - PINT over the same queries.",
        "",
        (
            f"{'METRIC':<20} {'PINT MEAN':>12} {'DAS MEAN':>12} "
            f"{'DAS-PINT':>12} {'PINT WINS':>12} {'DAS WINS':>11} "
            f"{'t':>10} {'p':>12}"
        ),
        "-" * 116,
    ]
    sources = {
        "trajectory": (pint_trajectory, das_trajectory),
        "perceptual": (pint_perceptual, das_perceptual),
    }
    for label, field, family in METRICS:
        pint_rows, das_rows = sources[family]
        pint = np.asarray([float(pint_rows[q][field]) for q in ordered])
        das = np.asarray([float(das_rows[q][field]) for q in ordered])
        test = ttest_rel(das, pint)
        pint_wins = int(np.sum(pint > das))
        das_wins = int(np.sum(das > pint))
        lines.append(
            f"{label:<20}{pint.mean():>12.8f}{das.mean():>12.8f}"
            f"{np.mean(das - pint):>+12.8f}{pint_wins:>8}/{len(ordered):<3d}"
            f"{das_wins:>7}/{len(ordered):<3d}{test.statistic:>10.4f}"
            f"{test.pvalue:>12.5g}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
