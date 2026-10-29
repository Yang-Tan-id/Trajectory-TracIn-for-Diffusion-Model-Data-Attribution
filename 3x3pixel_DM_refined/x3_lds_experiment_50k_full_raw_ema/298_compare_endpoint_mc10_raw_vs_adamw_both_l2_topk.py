"""Compare raw-gradient and Full-AdamW Endpoint MC10 Both-L2 removals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from exp_config import ROOT


QUERY_IDS = tuple(range(60))
FRACTION = "remove_10pct"
RAW_ROOT = ROOT / "topk_removal_endpoint_raw_gradient_mc10_both_l2_60q_10pct"
ADAMW_ROOT = ROOT / (
    "topk_removal_pint_endpoint_retrac_random_60q_10pct_"
    "final_endpoint_linear_adamw_sign"
)
RAW_TAG = "endpoint_mc10_raw_gradient_both_l2"
ADAMW_TAG = "endpoint_tracin_both_l2"
METRICS = (
    "noise_trajectory_mse",
    "trajectory_mse",
    "endpoint_mse",
    "simple_loss_abs_change",
)


def directory(root: Path, tag: str, query_id: int) -> Path:
    return root / tag / FRACTION / f"q{query_id:02d}"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path) as handle:
        return json.load(handle)


def main():
    raw = {
        q: load_json(directory(RAW_ROOT, RAW_TAG, q) / "evaluation.json")
        for q in QUERY_IDS
    }
    adamw = {
        q: load_json(directory(ADAMW_ROOT, ADAMW_TAG, q) / "evaluation.json")
        for q in QUERY_IDS
    }
    overlaps = []
    for q in QUERY_IDS:
        raw_indices = np.load(
            directory(RAW_ROOT, RAW_TAG, q) / "removed_indices.npy"
        )
        adamw_indices = np.load(
            directory(ADAMW_ROOT, ADAMW_TAG, q) / "removed_indices.npy"
        )
        overlaps.append(
            np.intersect1d(raw_indices, adamw_indices).size / raw_indices.size
        )

    output = {
        "query_ids": list(QUERY_IDS),
        "difference": "Full-AdamW minus raw-gradient",
        "raw_ranking": "descending saved score (LDS sign=-1)",
        "adamw_ranking": "ascending saved score (LDS sign=+1)",
        "topk_overlap": {
            "mean": float(np.mean(overlaps)),
            "std": float(np.std(overlaps)),
            "min": float(np.min(overlaps)),
            "max": float(np.max(overlaps)),
        },
        "metrics": {},
    }
    lines = [
        "ENDPOINT MC10 BOTH-L2 TRUE 10% REMOVAL: RAW-GRADIENT VS FULL-ADAMW",
        "q00-q59; delta = Full-AdamW minus raw-gradient",
        "raw ranks descending; AdamW ranks ascending (their respective LDS signs)",
        "",
        (
            f"Top-K overlap: {np.mean(overlaps):.4f} +/- {np.std(overlaps):.4f} "
            f"[min={np.min(overlaps):.4f}, max={np.max(overlaps):.4f}]"
        ),
        "",
        (
            f"{'metric':28s} {'raw-gradient':24s} {'Full-AdamW':24s} "
            f"{'delta':>12s} {'ratio':>9s} {'wins':>8s} {'p-W':>12s}"
        ),
    ]
    for metric in METRICS:
        raw_values = np.asarray([raw[q][metric] for q in QUERY_IDS], dtype=np.float64)
        adamw_values = np.asarray(
            [adamw[q][metric] for q in QUERY_IDS], dtype=np.float64
        )
        delta = adamw_values - raw_values
        item = {
            "raw_mean": float(raw_values.mean()),
            "raw_std": float(raw_values.std()),
            "adamw_mean": float(adamw_values.mean()),
            "adamw_std": float(adamw_values.std()),
            "adamw_minus_raw_mean": float(delta.mean()),
            "adamw_over_raw_ratio": float(adamw_values.mean() / raw_values.mean()),
            "adamw_wins": int(np.sum(delta > 0)),
            "raw_wins": int(np.sum(delta < 0)),
            "p_wilcoxon_two_sided": float(wilcoxon(delta).pvalue),
        }
        output["metrics"][metric] = item
        lines.append(
            f"{metric:28s} "
            f"{item['raw_mean']:.6e}+/-{item['raw_std']:.3e} "
            f"{item['adamw_mean']:.6e}+/-{item['adamw_std']:.3e} "
            f"{item['adamw_minus_raw_mean']:+.3e} "
            f"{item['adamw_over_raw_ratio']:9.4f} "
            f"{item['adamw_wins']:02d}/60 "
            f"{item['p_wilcoxon_two_sided']:.6g}"
        )

    json_path = RAW_ROOT / "endpoint_mc10_raw_vs_adamw_both_l2.json"
    text_path = RAW_ROOT / "endpoint_mc10_raw_vs_adamw_both_l2.txt"
    with open(json_path, "w") as handle:
        json.dump(output, handle, indent=2)
    text_path.write_text("\n".join(lines) + "\n")
    print(text_path.read_text(), flush=True)
    print(f"[saved] {json_path}", flush=True)
    print(f"[saved] {text_path}", flush=True)


if __name__ == "__main__":
    main()
