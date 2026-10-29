#!/usr/bin/env python3
"""Relate ELBO-weighted denoiser deviation to removal output changes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr


def parse_ids(text: str) -> list[int]:
    values = [int(value) for value in text.replace(",", " ").split() if value]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def fixed_variance_elbo_weights(
    total_timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> np.ndarray:
    """Return beta_t / (2 alpha_t (1-alpha_bar_t)).

    This matches the model-dependent epsilon-MSE coefficient used by the
    repository's fixed-variance DDPM VLB target (sigma_t^2 = beta_t).
    """
    betas = np.linspace(beta_start, beta_end, total_timesteps, dtype=np.float64)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return betas / (2.0 * alphas * (1.0 - alpha_bars))


def correlation(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.size < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return {"pearson": float("nan"), "spearman": float("nan")}
    return {
        "pearson": float(pearsonr(left, right).statistic),
        "spearman": float(spearmanr(left, right).statistic),
    }


def load_rows(paths: Iterable[Path]) -> dict[tuple[int, int, int], dict[str, Any]]:
    rows: dict[tuple[int, int, int], dict[str, Any]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                key = (
                    int(raw["evaluation_query_id"]),
                    int(raw["random_model_id"]),
                    int(raw["step"]),
                )
                converted = {
                    "evaluation_query_id": key[0],
                    "random_model_id": key[1],
                    "step": key[2],
                    "timestep": int(raw["timestep"]),
                    "predicted_noise_mse": float(raw["predicted_noise_mse"]),
                    "endpoint_mse": float(raw["endpoint_mse"]),
                    "trajectory_mse": float(raw["trajectory_mse"]),
                }
                previous = rows.get(key)
                if previous is not None and previous != converted:
                    raise RuntimeError(f"conflicting duplicate row for {key}")
                rows[key] = converted
    return rows


def aggregate_pairs(
    rows: dict[tuple[int, int, int], dict[str, Any]],
    query_ids: Sequence[int],
    model_ids: Sequence[int],
    elbo_weights: np.ndarray,
    expected_steps: int = 50,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows: list[dict[str, Any]] = []
    schedule_rows: dict[int, dict[str, Any]] = {}
    quadrature_scale = len(elbo_weights) / float(expected_steps)

    for query_id in query_ids:
        for model_id in model_ids:
            values = []
            for step in range(expected_steps):
                key = (query_id, model_id, step)
                if key not in rows:
                    raise RuntimeError(f"missing row {key}")
                values.append(rows[key])
            timesteps = np.asarray([row["timestep"] for row in values], dtype=np.int32)
            if np.any(timesteps < 0) or np.any(timesteps >= len(elbo_weights)):
                raise RuntimeError(f"invalid timestep for Q{query_id} R{model_id}")
            deviations = np.asarray(
                [row["predicted_noise_mse"] for row in values], dtype=np.float64
            )
            selected_weights = elbo_weights[timesteps]
            first = values[0]
            pair_rows.append({
                "evaluation_query_id": query_id,
                "random_model_id": model_id,
                "models_per_query": len(model_ids),
                "steps": expected_steps,
                "endpoint_mse": first["endpoint_mse"],
                "trajectory_mse": first["trajectory_mse"],
                "noise_deviation_unweighted_mean": float(deviations.mean()),
                "noise_deviation_elbo_weighted_mean": float(
                    np.average(deviations, weights=selected_weights)
                ),
                "noise_deviation_elbo_quadrature": float(
                    quadrature_scale * np.sum(selected_weights * deviations)
                ),
            })
            for step, timestep, weight in zip(range(expected_steps), timesteps, selected_weights):
                record = {
                    "step": step,
                    "timestep": int(timestep),
                    "elbo_weight": float(weight),
                    "normalized_selected_weight": float(
                        weight / selected_weights.sum()
                    ),
                }
                old = schedule_rows.get(step)
                if old is not None and old != record:
                    raise RuntimeError("DDIM timestep schedule differs between pairs")
                schedule_rows[step] = record
    return pair_rows, [schedule_rows[step] for step in range(expected_steps)]


def summarize_queries(
    pair_rows: Sequence[dict[str, Any]], query_ids: Sequence[int]
) -> list[dict[str, Any]]:
    output = []
    for query_id in query_ids:
        selected = [row for row in pair_rows if row["evaluation_query_id"] == query_id]
        endpoint = np.asarray([row["endpoint_mse"] for row in selected])
        trajectory = np.asarray([row["trajectory_mse"] for row in selected])
        unweighted = np.asarray([
            row["noise_deviation_unweighted_mean"] for row in selected
        ])
        weighted = np.asarray([
            row["noise_deviation_elbo_quadrature"] for row in selected
        ])
        unweighted_endpoint = correlation(unweighted, endpoint)
        unweighted_trajectory = correlation(unweighted, trajectory)
        weighted_endpoint = correlation(weighted, endpoint)
        weighted_trajectory = correlation(weighted, trajectory)
        output.append({
            "evaluation_query_id": query_id,
            "models": len(selected),
            "unweighted_endpoint_pearson": unweighted_endpoint["pearson"],
            "unweighted_endpoint_spearman": unweighted_endpoint["spearman"],
            "unweighted_trajectory_pearson": unweighted_trajectory["pearson"],
            "unweighted_trajectory_spearman": unweighted_trajectory["spearman"],
            "elbo_endpoint_pearson": weighted_endpoint["pearson"],
            "elbo_endpoint_spearman": weighted_endpoint["spearman"],
            "elbo_trajectory_pearson": weighted_trajectory["pearson"],
            "elbo_trajectory_spearman": weighted_trajectory["spearman"],
        })
    return output


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_text(
    summaries: Sequence[dict[str, Any]],
    query_ids: Sequence[int],
    model_ids: Sequence[int],
    schedule_rows: Sequence[dict[str, Any]],
) -> str:
    columns = (
        "unweighted_endpoint_pearson", "unweighted_endpoint_spearman",
        "unweighted_trajectory_pearson", "unweighted_trajectory_spearman",
        "elbo_endpoint_pearson", "elbo_endpoint_spearman",
        "elbo_trajectory_pearson", "elbo_trajectory_spearman",
    )
    values = {name: np.asarray([row[name] for row in summaries]) for name in columns}
    lines = [
        "CELEBA RANDOM TOP-1000 — ELBO-WEIGHTED REFERENCE-TRAJECTORY DENOISER DEVIATION",
        "=" * 132,
        f"queries={len(query_ids)} ({min(query_ids)}..{max(query_ids)}) "
        f"models_per_query={len(model_ids)} ({min(model_ids)}..{max(model_ids)}) "
        f"pairs={len(query_ids) * len(model_ids)} steps={len(schedule_rows)}",
        "Weight: beta_t / [2 alpha_t (1-alpha_bar_t)], matching the repository's fixed-variance DDPM VLB target.",
        "Aggregate: (T/K) sum_k weight(t_k) * MSE[eps_subset(x_t^ref)-eps_full(x_t^ref)].",
        "This is an ELBO-weighted functional deviation on DDIM reference states, not a literal ELBO.",
        "Correlations are computed within each query across the selected random-removal models.",
        "",
        f"{'QUERY':<7}{'N':>5}  {'UNW END-P':>10}{'UNW END-S':>11}"
        f"{'UNW TRJ-P':>11}{'UNW TRJ-S':>11}  {'ELBO END-P':>11}"
        f"{'ELBO END-S':>11}{'ELBO TRJ-P':>11}{'ELBO TRJ-S':>11}",
        "-" * 132,
    ]
    for row in summaries:
        lines.append(
            f"Q{row['evaluation_query_id']:<6}{row['models']:>5}  "
            f"{row['unweighted_endpoint_pearson']:>+10.4f}"
            f"{row['unweighted_endpoint_spearman']:>+11.4f}"
            f"{row['unweighted_trajectory_pearson']:>+11.4f}"
            f"{row['unweighted_trajectory_spearman']:>+11.4f}  "
            f"{row['elbo_endpoint_pearson']:>+11.4f}"
            f"{row['elbo_endpoint_spearman']:>+11.4f}"
            f"{row['elbo_trajectory_pearson']:>+11.4f}"
            f"{row['elbo_trajectory_spearman']:>+11.4f}"
        )
    lines.extend(["-" * 132])
    for statistic, reducer in (("MEAN", np.nanmean), ("STD", np.nanstd)):
        lines.append(
            f"{statistic:<12}"
            + "".join(f"{reducer(values[name]):>+11.4f}" for name in columns)
        )
    endpoint_delta = values["elbo_endpoint_pearson"] - values["unweighted_endpoint_pearson"]
    trajectory_delta = values["elbo_trajectory_pearson"] - values["unweighted_trajectory_pearson"]
    lines.extend([
        "",
        "ELBO WEIGHTING CHANGE IN PEARSON CORRELATION",
        "-" * 70,
        f"Endpoint:   mean delta={np.nanmean(endpoint_delta):+.4f}; "
        f"improved queries={int(np.sum(endpoint_delta > 0))}/{len(endpoint_delta)}",
        f"Trajectory: mean delta={np.nanmean(trajectory_delta):+.4f}; "
        f"improved queries={int(np.sum(trajectory_delta > 0))}/{len(trajectory_delta)}",
        "",
        "SELECTED WEIGHT RANGE",
        "-" * 70,
        f"largest: step={max(schedule_rows, key=lambda row: row['elbo_weight'])['step']} "
        f"t={max(schedule_rows, key=lambda row: row['elbo_weight'])['timestep']} "
        f"weight={max(row['elbo_weight'] for row in schedule_rows):.8g}",
        f"smallest: step={min(schedule_rows, key=lambda row: row['elbo_weight'])['step']} "
        f"t={min(schedule_rows, key=lambda row: row['elbo_weight'])['timestep']} "
        f"weight={min(row['elbo_weight'] for row in schedule_rows):.8g}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--query-ids", type=parse_ids, default=list(range(10)))
    parser.add_argument("--model-ids", type=parse_ids, default=list(range(60)))
    parser.add_argument("--expected-steps", type=int, default=50)
    parser.add_argument("--total-timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    paths = sorted(args.root.glob("shard_*/cross_model_per_query_timestep.csv"))
    if not paths and (args.root / "cross_model_per_query_timestep.csv").is_file():
        paths = [args.root / "cross_model_per_query_timestep.csv"]
    if not paths:
        raise SystemExit(f"no cross-model CSV files found under {args.root}")
    output_dir = args.output_dir or (args.root / "elbo_weighted_q0_9_m0_59")
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = fixed_variance_elbo_weights(
        args.total_timesteps, args.beta_start, args.beta_end
    )
    rows = load_rows(paths)
    pairs, schedule = aggregate_pairs(
        rows, args.query_ids, args.model_ids, weights, args.expected_steps
    )
    summaries = summarize_queries(pairs, args.query_ids)
    write_csv(output_dir / "elbo_weighted_per_query_model.csv", pairs)
    write_csv(output_dir / "elbo_weight_schedule.csv", schedule)
    write_csv(output_dir / "elbo_weighted_correlations.csv", summaries)
    report = report_text(
        summaries, args.query_ids, args.model_ids, schedule
    )
    (output_dir / "elbo_weighted_correlation_report.txt").write_text(
        report, encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(json.dumps({
        "source_csvs": [str(path.resolve()) for path in paths],
        "query_ids": args.query_ids,
        "model_ids": args.model_ids,
        "expected_steps": args.expected_steps,
        "total_timesteps": args.total_timesteps,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "weight": "beta_t / (2 * alpha_t * (1 - alpha_bar_t))",
        "aggregation": "T/K * sum_k weight(t_k) * predicted_noise_mse(t_k)",
        "caveat": "reference DDIM states, so this is not a literal DDPM ELBO",
    }, indent=2) + "\n", encoding="utf-8")
    print(report)
    print(f"[saved] {output_dir}")


if __name__ == "__main__":
    main()
