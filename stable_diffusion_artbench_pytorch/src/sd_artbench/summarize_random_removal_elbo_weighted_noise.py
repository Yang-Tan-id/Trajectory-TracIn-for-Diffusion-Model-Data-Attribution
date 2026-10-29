"""Summarize ELBO-weighted ArtBench random-removal denoiser deviations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .config import ArtBenchConfig


def parse_csv_ints(text: str) -> list[int]:
    values = [int(value) for value in text.replace(",", " ").split() if value]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def fixed_variance_elbo_weights_from_betas(betas: np.ndarray) -> np.ndarray:
    """Return beta_t / (2 alpha_t (1-alpha_bar_t))."""
    betas = np.asarray(betas, dtype=np.float64)
    if betas.ndim != 1 or betas.size == 0:
        raise ValueError("betas must be a non-empty vector")
    if np.any(betas <= 0.0) or np.any(betas >= 1.0):
        raise ValueError("betas must lie strictly between zero and one")
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return betas / (2.0 * alphas * (1.0 - alpha_bars))


def scheduler_weights(model: str, local_files_only: bool) -> tuple[np.ndarray, dict[str, Any]]:
    # Lazy import lets the pure aggregation tests run without loading diffusers.
    from diffusers import DDIMScheduler

    scheduler = DDIMScheduler.from_pretrained(
        model, subfolder="scheduler", local_files_only=local_files_only
    )
    betas = scheduler.betas.detach().cpu().numpy().astype(np.float64)
    metadata = {
        "model": model,
        "num_train_timesteps": int(scheduler.config.num_train_timesteps),
        "beta_start": float(scheduler.config.beta_start),
        "beta_end": float(scheduler.config.beta_end),
        "beta_schedule": str(scheduler.config.beta_schedule),
        "prediction_type": str(scheduler.config.prediction_type),
    }
    return fixed_variance_elbo_weights_from_betas(betas), metadata


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
                    int(raw["evaluation_query_index"]),
                    int(raw["random_model_index"]),
                    int(raw["step"]),
                )
                converted = {
                    "evaluation_query_index": key[0],
                    "random_model_index": key[1],
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
    query_indices: Sequence[int],
    model_indices: Sequence[int],
    weights: np.ndarray,
    expected_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows: list[dict[str, Any]] = []
    schedule_by_step: dict[int, dict[str, Any]] = {}
    quadrature_scale = len(weights) / float(expected_steps)

    for query in query_indices:
        for model in model_indices:
            values = []
            for step in range(expected_steps):
                key = (query, model, step)
                if key not in rows:
                    raise RuntimeError(f"missing row {key}")
                values.append(rows[key])
            timesteps = np.asarray([row["timestep"] for row in values], dtype=np.int32)
            if np.any(timesteps < 0) or np.any(timesteps >= len(weights)):
                raise RuntimeError(f"invalid timestep for Q{query} R{model}")
            deviations = np.asarray(
                [row["predicted_noise_mse"] for row in values], dtype=np.float64
            )
            selected_weights = weights[timesteps]
            first = values[0]
            pair_rows.append({
                "evaluation_query_index": query,
                "random_model_index": model,
                "models_per_query": len(model_indices),
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
            selected_sum = float(selected_weights.sum())
            for step, timestep, weight in zip(range(expected_steps), timesteps, selected_weights):
                record = {
                    "step": step,
                    "timestep": int(timestep),
                    "elbo_weight": float(weight),
                    "normalized_selected_weight": float(weight / selected_sum),
                }
                previous = schedule_by_step.get(step)
                if previous is not None and previous != record:
                    raise RuntimeError("DDIM timestep schedule differs between pairs")
                schedule_by_step[step] = record
    return pair_rows, [schedule_by_step[step] for step in range(expected_steps)]


def summarize_queries(
    pair_rows: Sequence[dict[str, Any]], query_indices: Sequence[int]
) -> list[dict[str, Any]]:
    output = []
    for query in query_indices:
        selected = [row for row in pair_rows if row["evaluation_query_index"] == query]
        endpoint = np.asarray([row["endpoint_mse"] for row in selected])
        trajectory = np.asarray([row["trajectory_mse"] for row in selected])
        unweighted = np.asarray([
            row["noise_deviation_unweighted_mean"] for row in selected
        ])
        weighted = np.asarray([
            row["noise_deviation_elbo_quadrature"] for row in selected
        ])
        ue, ut = correlation(unweighted, endpoint), correlation(unweighted, trajectory)
        we, wt = correlation(weighted, endpoint), correlation(weighted, trajectory)
        output.append({
            "evaluation_query_index": query,
            "models": len(selected),
            "unweighted_endpoint_pearson": ue["pearson"],
            "unweighted_endpoint_spearman": ue["spearman"],
            "unweighted_trajectory_pearson": ut["pearson"],
            "unweighted_trajectory_spearman": ut["spearman"],
            "elbo_endpoint_pearson": we["pearson"],
            "elbo_endpoint_spearman": we["spearman"],
            "elbo_trajectory_pearson": wt["pearson"],
            "elbo_trajectory_spearman": wt["spearman"],
        })
    return output


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_report(
    summaries: Sequence[dict[str, Any]], query_indices: Sequence[int],
    model_indices: Sequence[int], schedule: Sequence[dict[str, Any]],
    scheduler_metadata: dict[str, Any],
) -> str:
    columns = (
        "unweighted_endpoint_pearson", "unweighted_endpoint_spearman",
        "unweighted_trajectory_pearson", "unweighted_trajectory_spearman",
        "elbo_endpoint_pearson", "elbo_endpoint_spearman",
        "elbo_trajectory_pearson", "elbo_trajectory_spearman",
    )
    arrays = {name: np.asarray([row[name] for row in summaries]) for name in columns}
    lines = [
        "ARTBENCH RANDOM TOP-1000 — ELBO-WEIGHTED REFERENCE-TRAJECTORY DENOISER DEVIATION",
        "=" * 132,
        f"queries={len(query_indices)} ({min(query_indices)}..{max(query_indices)}) "
        f"models_per_query={len(model_indices)} ({min(model_indices)}..{max(model_indices)}) "
        f"pairs={len(query_indices) * len(model_indices)} steps={len(schedule)}",
        f"scheduler={scheduler_metadata['model']} "
        f"beta_schedule={scheduler_metadata['beta_schedule']} "
        f"prediction_type={scheduler_metadata['prediction_type']}",
        "Weight: beta_t / [2 alpha_t (1-alpha_bar_t)], the same fixed-variance coefficient used in the CelebA ablation.",
        "Aggregate: (T/K) sum_k weight(t_k) * MSE[eps_removed(x_t^ref)-eps_full(x_t^ref)].",
        "This is an ELBO-weighted deviation on guided DDIM reference states, not a literal ELBO.",
        "",
        f"{'QUERY':<7}{'N':>5}  {'UNW END-P':>10}{'UNW END-S':>11}"
        f"{'UNW TRJ-P':>11}{'UNW TRJ-S':>11}  {'ELBO END-P':>11}"
        f"{'ELBO END-S':>11}{'ELBO TRJ-P':>11}{'ELBO TRJ-S':>11}",
        "-" * 132,
    ]
    for row in summaries:
        lines.append(
            f"Q{row['evaluation_query_index']:<6}{row['models']:>5}  "
            f"{row['unweighted_endpoint_pearson']:>+10.4f}"
            f"{row['unweighted_endpoint_spearman']:>+11.4f}"
            f"{row['unweighted_trajectory_pearson']:>+11.4f}"
            f"{row['unweighted_trajectory_spearman']:>+11.4f}  "
            f"{row['elbo_endpoint_pearson']:>+11.4f}"
            f"{row['elbo_endpoint_spearman']:>+11.4f}"
            f"{row['elbo_trajectory_pearson']:>+11.4f}"
            f"{row['elbo_trajectory_spearman']:>+11.4f}"
        )
    lines.append("-" * 132)
    for label, reducer in (("MEAN", np.nanmean), ("STD", np.nanstd)):
        lines.append(
            f"{label:<12}" + "".join(
                f"{reducer(arrays[name]):>+11.4f}" for name in columns
            )
        )
    endpoint_delta = arrays["elbo_endpoint_pearson"] - arrays["unweighted_endpoint_pearson"]
    trajectory_delta = arrays["elbo_trajectory_pearson"] - arrays["unweighted_trajectory_pearson"]
    largest = max(schedule, key=lambda row: row["elbo_weight"])
    smallest = min(schedule, key=lambda row: row["elbo_weight"])
    lines.extend([
        "", "ELBO WEIGHTING CHANGE IN PEARSON CORRELATION", "-" * 72,
        f"Endpoint:   mean delta={np.nanmean(endpoint_delta):+.4f}; "
        f"improved queries={int(np.sum(endpoint_delta > 0))}/{len(endpoint_delta)}",
        f"Trajectory: mean delta={np.nanmean(trajectory_delta):+.4f}; "
        f"improved queries={int(np.sum(trajectory_delta > 0))}/{len(trajectory_delta)}",
        "", "SELECTED WEIGHT RANGE", "-" * 72,
        f"largest: step={largest['step']} t={largest['timestep']} "
        f"weight={largest['elbo_weight']:.8g}",
        f"smallest: step={smallest['step']} t={smallest['timestep']} "
        f"weight={smallest['elbo_weight']:.8g}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    root = ArtBenchConfig().index_root.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path,
        default=root / "outputs/random-removal-noise-correlation-cross-q0-59",
    )
    parser.add_argument("--query-indices", type=parse_csv_ints, default=list(range(10)))
    parser.add_argument("--model-indices", type=parse_csv_ints, default=list(range(60)))
    parser.add_argument("--expected-steps", type=int, default=50)
    parser.add_argument("--model", default="lambdalabs/miniSD-diffusers")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    paths = sorted(args.input_root.glob("**/cross_model_per_query_timestep.csv"))
    if not paths:
        raise SystemExit(f"no cross-model CSV files under {args.input_root}")
    output_dir = args.output_dir or (args.input_root / "elbo_weighted_q0_9_m0_59")
    output_dir.mkdir(parents=True, exist_ok=True)

    weights, scheduler_metadata = scheduler_weights(args.model, args.local_files_only)
    rows = load_rows(paths)
    pairs, schedule = aggregate_pairs(
        rows, args.query_indices, args.model_indices, weights, args.expected_steps
    )
    summaries = summarize_queries(pairs, args.query_indices)
    write_csv(output_dir / "elbo_weighted_per_query_model.csv", pairs)
    write_csv(output_dir / "elbo_weight_schedule.csv", schedule)
    write_csv(output_dir / "elbo_weighted_correlations.csv", summaries)
    report = make_report(
        summaries, args.query_indices, args.model_indices, schedule,
        scheduler_metadata,
    )
    (output_dir / "elbo_weighted_correlation_report.txt").write_text(
        report, encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(json.dumps({
        "source_csvs": [str(path.resolve()) for path in paths],
        "query_indices": args.query_indices,
        "model_indices": args.model_indices,
        "expected_steps": args.expected_steps,
        "scheduler": scheduler_metadata,
        "weight": "beta_t / (2 * alpha_t * (1 - alpha_bar_t))",
        "aggregation": "T/K * sum_k weight(t_k) * predicted_noise_mse(t_k)",
        "caveat": "guided DDIM reference states, so this is not a literal DDPM ELBO",
    }, indent=2) + "\n", encoding="utf-8")
    print(report)
    print(f"[saved] {output_dir}")


if __name__ == "__main__":
    main()
