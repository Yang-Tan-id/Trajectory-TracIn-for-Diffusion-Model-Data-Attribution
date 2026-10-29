"""Compare raw-gradient PINT and Endpoint MC10 Both-L2 true removals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from exp_config import ROOT


QUERY_IDS = tuple(range(60))
FRACTION = "remove_10pct"
PINT_ROOT = ROOT / "topk_removal_raw_gradient_pint_both_endpoint_mc1_60q_10pct"
ENDPOINT_ROOT = ROOT / "topk_removal_endpoint_raw_gradient_mc10_both_l2_60q_10pct"
PINT_TAG = "pint_both_l2_raw_gradient"
ENDPOINT_TAG = "endpoint_mc10_raw_gradient_both_l2"
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
    pint = {
        q: load_json(directory(PINT_ROOT, PINT_TAG, q) / "evaluation.json")
        for q in QUERY_IDS
    }
    endpoint = {
        q: load_json(directory(ENDPOINT_ROOT, ENDPOINT_TAG, q) / "evaluation.json")
        for q in QUERY_IDS
    }
    overlaps = []
    for q in QUERY_IDS:
        left = np.load(directory(PINT_ROOT, PINT_TAG, q) / "removed_indices.npy")
        right = np.load(
            directory(ENDPOINT_ROOT, ENDPOINT_TAG, q) / "removed_indices.npy"
        )
        overlaps.append(np.intersect1d(left, right).size / left.size)

    output = {
        "query_ids": list(QUERY_IDS),
        "difference": "Endpoint MC10 Both-L2 minus PINT Both-L2",
        "topk_overlap": {
            "mean": float(np.mean(overlaps)),
            "std": float(np.std(overlaps)),
            "min": float(np.min(overlaps)),
            "max": float(np.max(overlaps)),
        },
        "metrics": {},
    }
    lines = [
        "RAW-GRADIENT BOTH-L2 TRUE 10% REMOVAL: PINT VS ENDPOINT MC10",
        "q00-q59; delta = Endpoint MC10 minus PINT",
        "",
        (
            f"Top-K overlap: {np.mean(overlaps):.4f} +/- {np.std(overlaps):.4f} "
            f"[min={np.min(overlaps):.4f}, max={np.max(overlaps):.4f}]"
        ),
        "",
        (
            f"{'metric':28s} {'PINT Both-L2':24s} {'Endpoint MC10 Both-L2':24s} "
            f"{'delta':>12s} {'ratio':>9s} {'wins':>8s} {'p-W':>12s}"
        ),
    ]
    for metric in METRICS:
        pvalues = np.asarray([pint[q][metric] for q in QUERY_IDS], dtype=np.float64)
        evalues = np.asarray(
            [endpoint[q][metric] for q in QUERY_IDS], dtype=np.float64
        )
        delta = evalues - pvalues
        item = {
            "pint_mean": float(pvalues.mean()),
            "pint_std": float(pvalues.std()),
            "endpoint_mean": float(evalues.mean()),
            "endpoint_std": float(evalues.std()),
            "endpoint_minus_pint_mean": float(delta.mean()),
            "endpoint_over_pint_ratio": float(evalues.mean() / pvalues.mean()),
            "endpoint_wins": int(np.sum(delta > 0)),
            "pint_wins": int(np.sum(delta < 0)),
            "p_wilcoxon_two_sided": float(wilcoxon(delta).pvalue),
        }
        output["metrics"][metric] = item
        lines.append(
            f"{metric:28s} "
            f"{item['pint_mean']:.6e}+/-{item['pint_std']:.3e} "
            f"{item['endpoint_mean']:.6e}+/-{item['endpoint_std']:.3e} "
            f"{item['endpoint_minus_pint_mean']:+.3e} "
            f"{item['endpoint_over_pint_ratio']:9.4f} "
            f"{item['endpoint_wins']:02d}/60 "
            f"{item['p_wilcoxon_two_sided']:.6g}"
        )

    json_path = ENDPOINT_ROOT / "endpoint_mc10_vs_pint_both_l2.json"
    text_path = ENDPOINT_ROOT / "endpoint_mc10_vs_pint_both_l2.txt"
    with open(json_path, "w") as handle:
        json.dump(output, handle, indent=2)
    text_path.write_text("\n".join(lines) + "\n")
    print(text_path.read_text(), flush=True)
    print(f"[saved] {json_path}", flush=True)
    print(f"[saved] {text_path}", flush=True)


if __name__ == "__main__":
    main()
