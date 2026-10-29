"""Compare true 10% removal effects for raw-gradient and Full-AdamW PINT."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

from exp_config import ROOT


OUT_ROOT = ROOT / "topk_removal_pint_endpoint_retrac_random_60q_10pct"
QUERY_IDS = tuple(range(60))
FRACTION_TAG = "remove_10pct"
RAW_TAG = "pint_raw_gradient_raw"
ADAMW_TAG = "pint_raw"
METRICS = (
    "noise_trajectory_mse",
    "trajectory_mse",
    "endpoint_mse",
    "simple_loss_abs_change",
)


def load_evaluation(tag: str, query_id: int) -> dict:
    path = OUT_ROOT / tag / FRACTION_TAG / f"q{query_id:02d}" / "evaluation.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path) as handle:
        return json.load(handle)


def main() -> None:
    raw = {q: load_evaluation(RAW_TAG, q) for q in QUERY_IDS}
    adamw = {q: load_evaluation(ADAMW_TAG, q) for q in QUERY_IDS}
    rows = []
    summary = {
        "difference_definition": "Full-AdamW PINT minus raw-gradient PINT",
        "query_ids": list(QUERY_IDS),
        "metrics": {},
    }
    for query_id in QUERY_IDS:
        row = {"query_id": query_id, "family": raw[query_id]["family"]}
        for metric in METRICS:
            raw_value = float(raw[query_id][metric])
            adamw_value = float(adamw[query_id][metric])
            row[f"raw_gradient_{metric}"] = raw_value
            row[f"adamw_{metric}"] = adamw_value
            row[f"adamw_minus_raw_{metric}"] = adamw_value - raw_value
        rows.append(row)

    for metric in METRICS:
        raw_values = np.array([raw[q][metric] for q in QUERY_IDS], dtype=np.float64)
        adamw_values = np.array([adamw[q][metric] for q in QUERY_IDS], dtype=np.float64)
        difference = adamw_values - raw_values
        wins = int(np.sum(difference > 0))
        losses = int(np.sum(difference < 0))
        summary["metrics"][metric] = {
            "raw_gradient_mean": float(raw_values.mean()),
            "raw_gradient_std": float(raw_values.std()),
            "adamw_mean": float(adamw_values.mean()),
            "adamw_std": float(adamw_values.std()),
            "adamw_minus_raw_mean": float(difference.mean()),
            "adamw_minus_raw_std": float(difference.std()),
            "adamw_over_raw_mean_ratio": (
                float(adamw_values.mean() / raw_values.mean())
                if raw_values.mean() != 0
                else None
            ),
            "adamw_wins": wins,
            "ties": int(np.sum(difference == 0)),
            "raw_gradient_wins": losses,
            "p_wilcoxon_two_sided": float(
                wilcoxon(difference, alternative="two-sided").pvalue
            ),
            "p_sign_two_sided": (
                float(binomtest(wins, wins + losses, 0.5).pvalue)
                if wins + losses
                else 1.0
            ),
        }

    overlap_path = OUT_ROOT / "pint_raw_gradient_vs_adamw_topk_overlap.json"
    with open(overlap_path) as handle:
        overlap = json.load(handle)
    overlap_values = np.array(
        [row["overlap_fraction"] for row in overlap], dtype=np.float64
    )
    summary["topk_overlap_fraction"] = {
        "mean": float(overlap_values.mean()),
        "std": float(overlap_values.std()),
        "min": float(overlap_values.min()),
        "max": float(overlap_values.max()),
    }

    json_path = OUT_ROOT / "pint_raw_gradient_vs_adamw_topk_comparison.json"
    csv_path = OUT_ROOT / "pint_raw_gradient_vs_adamw_topk_per_query.csv"
    text_path = OUT_ROOT / "pint_raw_gradient_vs_adamw_topk_comparison.txt"
    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(text_path, "w") as handle:
        handle.write("PINT TRUE 10% TOP-K REMOVAL: RAW-GRADIENT VS FULL-ADAMW\n")
        handle.write("difference = Full-AdamW minus raw-gradient; q00-q59\n\n")
        handle.write(
            f"Top-K overlap: {overlap_values.mean():.4f} +/- "
            f"{overlap_values.std():.4f}\n\n"
        )
        handle.write(
            f"{'metric':28s} {'raw-gradient':24s} {'Full-AdamW':24s} "
            f"{'delta':>12s} {'ratio':>9s} {'wins':>8s} {'p-W':>12s}\n"
        )
        for metric in METRICS:
            item = summary["metrics"][metric]
            handle.write(
                f"{metric:28s} "
                f"{item['raw_gradient_mean']:.6e}+/-{item['raw_gradient_std']:.3e} "
                f"{item['adamw_mean']:.6e}+/-{item['adamw_std']:.3e} "
                f"{item['adamw_minus_raw_mean']:+.3e} "
                f"{item['adamw_over_raw_mean_ratio']:9.4f} "
                f"{item['adamw_wins']:02d}/60 "
                f"{item['p_wilcoxon_two_sided']:.6g}\n"
            )
    print(text_path.read_text(), flush=True)
    print(f"[saved] {json_path}", flush=True)
    print(f"[saved] {csv_path}", flush=True)
    print(f"[saved] {text_path}", flush=True)


if __name__ == "__main__":
    main()
