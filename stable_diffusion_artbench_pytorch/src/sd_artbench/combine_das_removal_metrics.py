"""Combine DAS counterfactual trajectory and perceptual metrics into one report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


FIELDS = (
    "endpoint_mse",
    "endpoint_rmse",
    "trajectory_mse",
    "trajectory_rmse",
    "pixel_l2",
    "clip_distance",
    "lpips",
    "dino_distance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-queries", type=int, default=60)
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    trajectory = load_csv(args.input_dir / "endpoint_trajectory_metrics.csv")
    perceptual = load_csv(args.input_dir / "perceptual_metrics.csv")
    left = {(int(row["query_index"]), row["method"]): row for row in trajectory}
    right = {(int(row["query_index"]), row["method"]): row for row in perceptual}
    if set(left) != set(right):
        raise ValueError("trajectory and perceptual result keys differ")
    if len(left) != args.expected_queries:
        raise ValueError(f"expected {args.expected_queries} records, found {len(left)}")
    keys = sorted(left)
    lines = [
        "ARTBENCH DAS LAMBDA=1000 TOP-1000 REMOVAL — Q0-Q59",
        "=" * 132,
        f"queries={args.expected_queries}; removal=1000/5000; retraining=100 epochs",
        "All distances are measured against the corresponding full-data model output; larger means a larger removal effect.",
        "",
        (
            f"{'QUERY':<7} {'END-MSE':>11} {'END-RMSE':>11} {'TRAJ-MSE':>11} "
            f"{'TRAJ-RMSE':>11} {'PIXEL-L2':>11} {'CLIP-DIST':>11} "
            f"{'LPIPS':>11} {'DINO-DIST':>11}"
        ),
        "-" * 132,
    ]
    combined: list[dict[str, float]] = []
    for query, method in keys:
        row = {**left[(query, method)], **right[(query, method)]}
        values = {field: float(row[field]) for field in FIELDS}
        combined.append(values)
        lines.append(
            f"Q{query:<6d} " + " ".join(f"{values[field]:>11.7f}" for field in FIELDS)
        )
    lines.extend(["", "MEAN OVER 60 QUERIES", "-" * 132])
    lines.append(
        " " * 7
        + " ".join(
            f"{np.mean([row[field] for row in combined]):>11.7f}" for field in FIELDS
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {args.output}", flush=True)


if __name__ == "__main__":
    main()
