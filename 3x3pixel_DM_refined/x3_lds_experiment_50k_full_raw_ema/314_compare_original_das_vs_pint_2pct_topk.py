#!/usr/bin/env python3
"""Compare existing original-DAS and raw-gradient PINT true 2% removals."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from exp_config import ROOT


QUERY_IDS = tuple(range(100))
METRICS = ("trajectory_mse", "endpoint_mse")
DAS_ROOT = ROOT / "topk_removal_retrain" / "das_ema_lambda_10"
PINT_ROOT = (
    ROOT
    / "topk_removal_retrac_endtracin_tracindas_100q"
    / "tracin_das_aligned_timestamp_square"
    / "remove_02pct"
)
OUT_PREFIX = ROOT / "original_das_ema_lambda10_vs_raw_pint_2pct_q00_q99"


def load_arm(root: Path, query_id: int) -> tuple[dict, np.ndarray]:
    query_dir = root / f"q{query_id:02d}"
    evaluation_path = query_dir / "evaluation.json"
    removed_path = query_dir / "removed_indices.npy"
    if not evaluation_path.is_file():
        raise FileNotFoundError(evaluation_path)
    if not removed_path.is_file():
        raise FileNotFoundError(removed_path)
    return json.loads(evaluation_path.read_text()), np.load(removed_path).astype(np.int64)


def scope_ids(scope: str) -> tuple[int, ...]:
    if scope == "all":
        return QUERY_IDS
    if scope == "prompted":
        return tuple(range(75))
    if scope == "unprompted":
        return tuple(range(75, 100))
    raise ValueError(scope)


def main() -> None:
    rows = []
    for query_id in QUERY_IDS:
        das, das_removed = load_arm(DAS_ROOT, query_id)
        pint, pint_removed = load_arm(PINT_ROOT, query_id)
        if len(das_removed) != 1000 or len(pint_removed) != 1000:
            raise ValueError(
                f"q{query_id:02d}: expected 1000 removals, "
                f"found DAS={len(das_removed)}, PINT={len(pint_removed)}"
            )
        overlap = len(set(das_removed.tolist()) & set(pint_removed.tolist()))
        row = {
            "query_id": query_id,
            "family": "prompted" if query_id < 75 else "unprompted",
            "overlap_count": overlap,
            "overlap_fraction": overlap / 1000.0,
        }
        for metric in METRICS:
            das_value = float(das[metric])
            pint_value = float(pint[metric])
            row[f"das_{metric}"] = das_value
            row[f"pint_{metric}"] = pint_value
            row[f"delta_pint_minus_das_{metric}"] = pint_value - das_value
        rows.append(row)

    payload = {
        "comparison": "raw-gradient PINT minus original DAS-EMA lambda=10",
        "removal_fraction": 0.02,
        "topk": 1000,
        "query_ids": list(QUERY_IDS),
        "scopes": {},
        "per_query": rows,
    }
    lines = [
        "X3 TRUE 2% REMOVAL — ORIGINAL DAS-EMA LAMBDA=10 VS RAW-GRADIENT PINT",
        "delta = PINT - DAS; positive means PINT removal has the larger true effect",
        "queries=q00-q99; topk=1000/50000; evaluation parameters=EMA",
        "",
    ]
    by_query = {int(row["query_id"]): row for row in rows}
    for scope in ("all", "prompted", "unprompted"):
        ids = scope_ids(scope)
        selected = [by_query[query_id] for query_id in ids]
        scope_result = {
            "n": len(selected),
            "overlap": {
                "mean": float(np.mean([row["overlap_fraction"] for row in selected])),
                "std": float(np.std([row["overlap_fraction"] for row in selected])),
            },
            "metrics": {},
        }
        lines.extend(
            [
                f"SCOPE: {scope} (n={len(selected)})",
                f"Top-K overlap: {100 * scope_result['overlap']['mean']:.2f}% "
                f"+/- {100 * scope_result['overlap']['std']:.2f}%",
                (
                    f"{'metric':22s} {'DAS mean+/-std':25s} {'PINT mean+/-std':25s} "
                    f"{'delta':>12s} {'ratio':>9s} {'wins':>8s} {'p-W':>12s}"
                ),
            ]
        )
        for metric in METRICS:
            das = np.asarray([row[f"das_{metric}"] for row in selected])
            pint = np.asarray([row[f"pint_{metric}"] for row in selected])
            delta = pint - das
            pvalue = float(wilcoxon(delta, alternative="two-sided").pvalue)
            result = {
                "das_mean": float(das.mean()),
                "das_std": float(das.std()),
                "pint_mean": float(pint.mean()),
                "pint_std": float(pint.std()),
                "mean_delta": float(delta.mean()),
                "pint_over_das_ratio": float(pint.mean() / das.mean()),
                "pint_wins": int(np.sum(delta > 0)),
                "wilcoxon_two_sided_p": pvalue,
            }
            scope_result["metrics"][metric] = result
            lines.append(
                f"{metric:22s} "
                f"{das.mean():.6e}+/-{das.std():.3e} "
                f"{pint.mean():.6e}+/-{pint.std():.3e} "
                f"{delta.mean():+12.3e} {pint.mean()/das.mean():9.4f} "
                f"{int(np.sum(delta > 0)):3d}/{len(delta):<3d} {pvalue:12.6g}"
            )
        payload["scopes"][scope] = scope_result
        lines.append("")

    lines.extend(
        [
            "PER QUERY",
            (
                f"{'query':5s} {'family':10s} {'overlap':>8s} "
                f"{'DAS traj':>12s} {'PINT traj':>12s} {'delta':>12s} "
                f"{'DAS endpoint':>14s} {'PINT endpoint':>14s} {'delta':>12s}"
            ),
        ]
    )
    for row in rows:
        lines.append(
            f"q{row['query_id']:02d}   {row['family']:10s} "
            f"{100*row['overlap_fraction']:7.2f}% "
            f"{row['das_trajectory_mse']:12.6e} "
            f"{row['pint_trajectory_mse']:12.6e} "
            f"{row['delta_pint_minus_das_trajectory_mse']:+12.3e} "
            f"{row['das_endpoint_mse']:14.6e} "
            f"{row['pint_endpoint_mse']:14.6e} "
            f"{row['delta_pint_minus_das_endpoint_mse']:+12.3e}"
        )

    json_path = OUT_PREFIX.with_suffix(".json")
    text_path = OUT_PREFIX.with_suffix(".txt")
    csv_path = OUT_PREFIX.with_suffix(".csv")
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    text_path.write_text("\n".join(lines) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(lines), flush=True)
    print(f"[saved] {json_path}")
    print(f"[saved] {text_path}")
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
