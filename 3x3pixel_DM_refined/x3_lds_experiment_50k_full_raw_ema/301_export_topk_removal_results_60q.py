#!/usr/bin/env python3
"""Export true x3 top-10% removal effects for completed Q00-Q59 arms."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from exp_config import ROOT


QUERY_IDS = tuple(range(60))
FRACTION = "remove_10pct"
METRICS = (
    "noise_trajectory_mse",
    "trajectory_mse",
    "endpoint_mse",
    "simple_loss_abs_change",
)

ADAMW_ROOT = ROOT / (
    "topk_removal_pint_endpoint_retrac_random_60q_10pct_"
    "final_endpoint_linear_adamw_sign"
)
OLD_RAW_ROOT = ROOT / "topk_removal_retrac_endtracin_tracindas_100q"

# Raw-gradient PINT/Both-L2 is intentionally omitted: that removal arm was not run.
ARMS = (
    ("PINT / RAW", "RAW-GRADIENT", OLD_RAW_ROOT, "tracin_das_aligned_timestamp_square"),
    ("PINT / RAW", "FULL-ADAMW", ADAMW_ROOT, "pint_raw"),
    ("PINT / BOTH-L2", "FULL-ADAMW", ADAMW_ROOT, "pint_both_l2"),
    (
        "ENDPOINT TRACIN / RAW",
        "RAW-GRADIENT",
        ROOT / "topk_removal_endpoint_raw_gradient_mc10_raw_60q_10pct",
        "endpoint_mc10_raw_gradient_raw",
    ),
    ("ENDPOINT TRACIN / RAW", "FULL-ADAMW", ADAMW_ROOT, "endpoint_tracin_raw"),
    (
        "ENDPOINT TRACIN / BOTH-L2",
        "RAW-GRADIENT",
        ROOT / "topk_removal_endpoint_raw_gradient_mc10_both_l2_60q_10pct",
        "endpoint_mc10_raw_gradient_both_l2",
    ),
    (
        "ENDPOINT TRACIN / BOTH-L2",
        "FULL-ADAMW",
        ADAMW_ROOT,
        "endpoint_tracin_both_l2",
    ),
    ("RETRAC / BOTH-L2", "RAW-GRADIENT", OLD_RAW_ROOT, "diffusion_retrac_normalized"),
    ("RETRAC / BOTH-L2", "FULL-ADAMW", ADAMW_ROOT, "retrac_both_l2"),
    ("RANDOM REMOVAL", "CONTROL", ADAMW_ROOT, "random"),
)


def evaluation_path(root: Path, tag: str, query_id: int) -> Path:
    return root / tag / FRACTION / f"q{query_id:02d}" / "evaluation.json"


def load_arm(root: Path, tag: str) -> dict[int, dict]:
    rows = {}
    for query_id in QUERY_IDS:
        path = evaluation_path(root, tag, query_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        rows[query_id] = json.loads(path.read_text())
    return rows


def safe_wilcoxon(difference: np.ndarray) -> float:
    if not np.any(difference != 0):
        return 1.0
    return float(wilcoxon(difference, alternative="two-sided").pvalue)


def main() -> None:
    loaded = {}
    for method, transform, root, tag in ARMS:
        loaded[(method, transform)] = load_arm(root, tag)
        print(f"[loaded] {transform:12s} {method}", flush=True)

    random_rows = loaded[("RANDOM REMOVAL", "CONTROL")]
    summary = []
    per_query = []
    for method, transform, _, _ in ARMS:
        rows = loaded[(method, transform)]
        for query_id in QUERY_IDS:
            record = {"query_id": query_id, "method": method, "transform": transform}
            record.update({metric: float(rows[query_id][metric]) for metric in METRICS})
            per_query.append(record)
        for metric in METRICS:
            values = np.asarray([rows[q][metric] for q in QUERY_IDS], dtype=np.float64)
            random_values = np.asarray(
                [random_rows[q][metric] for q in QUERY_IDS], dtype=np.float64
            )
            difference = values - random_values
            random_mean = float(random_values.mean())
            summary.append(
                {
                    "method": method,
                    "transform": transform,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "ratio_to_random": (
                        float(values.mean() / random_mean) if random_mean else None
                    ),
                    "wins_vs_random": int(np.sum(difference > 0)),
                    "p_wilcoxon_vs_random": safe_wilcoxon(difference),
                }
            )

    paired = []
    methods = tuple(dict.fromkeys(method for method, *_ in ARMS))
    for method in methods:
        raw_key = (method, "RAW-GRADIENT")
        adamw_key = (method, "FULL-ADAMW")
        if raw_key not in loaded or adamw_key not in loaded:
            continue
        for metric in METRICS:
            raw = np.asarray(
                [loaded[raw_key][q][metric] for q in QUERY_IDS], dtype=np.float64
            )
            adamw = np.asarray(
                [loaded[adamw_key][q][metric] for q in QUERY_IDS], dtype=np.float64
            )
            difference = adamw - raw
            paired.append(
                {
                    "method": method,
                    "metric": metric,
                    "raw_mean": float(raw.mean()),
                    "adamw_mean": float(adamw.mean()),
                    "adamw_minus_raw": float(difference.mean()),
                    "adamw_over_raw": (
                        float(adamw.mean() / raw.mean()) if raw.mean() else None
                    ),
                    "adamw_wins": int(np.sum(difference > 0)),
                    "p_wilcoxon": safe_wilcoxon(difference),
                }
            )

    lines = [
        "X3 TRUE TOP-10% REMOVAL RESULTS — Q00-Q59",
        "Larger removal effect means the selected Top-K set changes the model more.",
        "Raw-gradient PINT/Both-L2 is omitted because that removal arm was not run.",
        "",
    ]
    for metric in METRICS:
        lines.extend(
            [
                f"METRIC: {metric}",
                (
                    f"{'METHOD':30s} {'TRANSFORM':12s} {'MEAN +/- STD':27s} "
                    f"{'RATIO/RAND':>11s} {'WINS':>8s} {'P-WILCOXON':>12s}"
                ),
                "-" * 110,
            ]
        )
        for row in summary:
            if row["metric"] != metric:
                continue
            ratio = row["ratio_to_random"]
            lines.append(
                f"{row['method']:30s} {row['transform']:12s} "
                f"{row['mean']:.6e} +/- {row['std']:.3e} "
                f"{ratio:11.4f} {row['wins_vs_random']:2d}/60 "
                f"{row['p_wilcoxon_vs_random']:12.6g}"
            )
        lines.append("")

    lines.extend(
        [
            "FULL-ADAMW MINUS RAW-GRADIENT",
            "Positive delta means Full-AdamW Top-K removal has the larger true effect.",
            (
                f"{'METHOD':30s} {'METRIC':25s} {'RAW':>12s} {'ADAMW':>12s} "
                f"{'DELTA':>12s} {'RATIO':>9s} {'WINS':>8s} {'P-W':>10s}"
            ),
            "-" * 125,
        ]
    )
    for row in paired:
        lines.append(
            f"{row['method']:30s} {row['metric']:25s} "
            f"{row['raw_mean']:12.6e} {row['adamw_mean']:12.6e} "
            f"{row['adamw_minus_raw']:+12.3e} {row['adamw_over_raw']:9.4f} "
            f"{row['adamw_wins']:2d}/60 {row['p_wilcoxon']:10.6g}"
        )

    lines.extend(["", "PER-QUERY RESULTS"])
    for metric in METRICS:
        lines.extend(["", f"METRIC: {metric}"])
        header = "QUERY" + "".join(
            f"  {method} [{transform}]" for method, transform, _, _ in ARMS
        )
        lines.append(header)
        for query_id in QUERY_IDS:
            values = []
            for method, transform, _, _ in ARMS:
                values.append(loaded[(method, transform)][query_id][metric])
            lines.append(
                f"q{query_id:02d}" + "".join(f"  {float(value):.8e}" for value in values)
            )

    output = ROOT / "topk_removal_results_raw_vs_adamw_60q.txt"
    output.write_text("\n".join(lines) + "\n")
    output.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "adamw_minus_raw": paired}, indent=2) + "\n"
    )
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_query[0].keys())
        writer.writeheader()
        writer.writerows(per_query)
    print("\n".join(lines[:100]))
    print(f"[saved] {output}")
    print(f"[saved] {output.with_suffix('.json')}")
    print(f"[saved] {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
