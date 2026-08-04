#!/usr/bin/env python3
"""
Simulate projection error for Traj-TracIn-style accumulated inner products.

The default mode uses the Johnson-Lindenstrauss error distribution directly.
That is the right first pass for the large setting:

  10000 train points x 50 checkpoints x 100 timestamps x D ~= 32.7M

Materializing raw gradients at that scale is not useful. Store projected
vectors or final score/error summaries instead.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def summarize(x: np.ndarray) -> dict[str, float]:
    abs_x = np.abs(x)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "mae": float(np.mean(abs_x)),
        "p50_abs": float(np.quantile(abs_x, 0.50)),
        "p90_abs": float(np.quantile(abs_x, 0.90)),
        "p95_abs": float(np.quantile(abs_x, 0.95)),
        "p99_abs": float(np.quantile(abs_x, 0.99)),
        "max_abs": float(np.max(abs_x)),
    }


def simulate_scores(
    n_points: int,
    gradient_dim: int,
    n_checkpoints: int,
    n_timestamps: int,
    projection_dims: list[int],
    seed: int,
    average_checkpoints: bool,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)

    # Existing code averages timestamps inside each checkpoint:
    #   score_ckpt = (1 / n_timestamps) * sum_t dot(q_t, g_t)
    # and then adds checkpoint scores. Optionally average checkpoints too.
    timestamp_weight = 1.0 / n_timestamps
    checkpoint_weight = 1.0 / n_checkpoints if average_checkpoints else 1.0
    sum_weight_sq = (
        n_checkpoints
        * n_timestamps
        * (timestamp_weight * checkpoint_weight) ** 2
    )

    # For independent random normalized gradients in R^D:
    # true dot has variance about 1/D.
    true_score_std = math.sqrt(sum_weight_sq / gradient_dim)
    true_scores = rng.normal(0.0, true_score_std, size=n_points)

    rows: list[dict[str, float]] = []
    for k in projection_dims:
        # JL dot estimator is unbiased, with variance about 1/k per unit-vector
        # dot term. Accumulated error variance is sum(weights^2) / k.
        error_std = math.sqrt(sum_weight_sq / k)
        errors = rng.normal(0.0, error_std, size=n_points)
        projected_scores = true_scores + errors

        corr = float(np.corrcoef(true_scores, projected_scores)[0, 1])
        sign_agreement = float(np.mean(np.signbit(true_scores) == np.signbit(projected_scores)))
        top_count = max(1, min(100, n_points))
        top_true = set(np.argpartition(true_scores, -top_count)[-top_count:].tolist())
        top_proj = set(np.argpartition(projected_scores, -top_count)[-top_count:].tolist())
        top_overlap = len(top_true & top_proj) / top_count

        row = {
            "projection_dim": float(k),
            "theory_error_std": error_std,
            "true_score_std": true_score_std,
            "pearson": corr,
            "sign_agreement": sign_agreement,
            "top100_overlap": float(top_overlap),
        }
        row.update({f"error_{key}": value for key, value in summarize(errors).items()})
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-points", type=int, default=10_000)
    parser.add_argument("--gradient-dim", type=int, default=32_661_123)
    parser.add_argument("--n-checkpoints", type=int, default=50)
    parser.add_argument("--n-timestamps", type=int, default=100)
    parser.add_argument(
        "--projection-dims",
        type=str,
        default="256,512,1024,2048,4096",
        help="Comma-separated projection dimensions.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--average-checkpoints",
        action="store_true",
        help="Also divide the final score by the number of checkpoints.",
    )
    parser.add_argument("--out", type=Path, default=Path("projection_error_summary.csv"))
    args = parser.parse_args()

    projection_dims = [int(x) for x in args.projection_dims.split(",") if x.strip()]
    rows = simulate_scores(
        n_points=args.n_points,
        gradient_dim=args.gradient_dim,
        n_checkpoints=args.n_checkpoints,
        n_timestamps=args.n_timestamps,
        projection_dims=projection_dims,
        seed=args.seed,
        average_checkpoints=args.average_checkpoints,
    )

    fieldnames = list(rows[0].keys())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"k={int(row['projection_dim']):4d} "
            f"std={row['error_std']:.6g} "
            f"p95_abs={row['error_p95_abs']:.6g} "
            f"p99_abs={row['error_p99_abs']:.6g} "
            f"pearson={row['pearson']:.4f} "
            f"top100={row['top100_overlap']:.3f}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
