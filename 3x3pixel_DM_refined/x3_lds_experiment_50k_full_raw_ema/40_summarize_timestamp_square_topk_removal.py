"""Summarize timestamp-square removal and optionally pair with existing DAS."""

import csv
import json
from pathlib import Path

import numpy as np

from exp_config import ROOT


OUT_ROOT = ROOT / "topk_removal_traj_first_raw_timestamp_square_q00_q09"
DAS_ROOT = ROOT / "topk_removal_joint_mucs_vs_original_das_q00_q09"
DAS_TAG = "das_ema_lambda_10"
METRICS = (
    "trajectory_mse",
    "trajectory_rmse",
    "trajectory_max_abs",
    "endpoint_mse",
    "endpoint_rmse",
    "endpoint_l2",
    "endpoint_max_abs",
)


def load_rows(root, jobs):
    rows = []
    missing = []
    for job in jobs:
        path = Path(job["job_dir"]) / "evaluation.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        with open(path) as handle:
            rows.append(json.load(handle))
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} evaluations; first={missing[0]}")
    return rows


def main():
    with open(OUT_ROOT / "jobs.json") as handle:
        jobs = json.load(handle)
    rows = load_rows(OUT_ROOT, jobs)
    if len(rows) != 10:
        raise ValueError(f"expected 10 timestamp-square evaluations, found {len(rows)}")

    summary = {
        "method": rows[0]["method"],
        "count": len(rows),
        "ranking": "1,000 largest saved scores; no LDS sign",
        "score_param_source": rows[0]["score_param_source"],
        "eval_param_source": rows[0]["eval_param_source"],
        "metrics": {},
    }
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary["metrics"][metric] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    per_query_path = OUT_ROOT / "per_query_results.csv"
    fields = [
        "job_id",
        "query_id",
        "family",
        "method_tag",
        "method",
        "score_param_source",
        "eval_param_source",
        "topk",
        "initial_seed",
        *METRICS,
    ]
    with open(per_query_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["query_id"]))

    das_rows = {}
    for query_id in range(10):
        path = DAS_ROOT / DAS_TAG / f"q{query_id:02d}" / "evaluation.json"
        if path.is_file():
            with open(path) as handle:
                das_rows[query_id] = json.load(handle)
    paired_rows = []
    if len(das_rows) == 10:
        method_rows = {int(row["query_id"]): row for row in rows}
        for query_id in range(10):
            pair = {"query_id": query_id, "family": method_rows[query_id]["family"]}
            for metric in METRICS:
                method_value = float(method_rows[query_id][metric])
                das_value = float(das_rows[query_id][metric])
                pair[f"timestamp_square_{metric}"] = method_value
                pair[f"das_{metric}"] = das_value
                pair[f"delta_timestamp_square_minus_das_{metric}"] = (
                    method_value - das_value
                )
            paired_rows.append(pair)

        paired_fields = ["query_id", "family"]
        for metric in METRICS:
            paired_fields.extend(
                [
                    f"timestamp_square_{metric}",
                    f"das_{metric}",
                    f"delta_timestamp_square_minus_das_{metric}",
                ]
            )
        paired_path = OUT_ROOT / "paired_with_existing_das.csv"
        with open(paired_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=paired_fields)
            writer.writeheader()
            writer.writerows(paired_rows)
        summary["das_comparison"] = {
            "available": True,
            "das_root": str(DAS_ROOT),
            "delta_definition": "timestamp_square_minus_das",
            "metrics": {},
        }
        for metric in METRICS:
            deltas = np.asarray(
                [
                    row[f"delta_timestamp_square_minus_das_{metric}"]
                    for row in paired_rows
                ],
                dtype=np.float64,
            )
            summary["das_comparison"]["metrics"][metric] = {
                "mean_delta": float(deltas.mean()),
                "median_delta": float(np.median(deltas)),
                "timestamp_square_larger_count": int(np.sum(deltas > 0)),
                "das_larger_count": int(np.sum(deltas < 0)),
            }
    else:
        summary["das_comparison"] = {
            "available": False,
            "found_evaluations": len(das_rows),
            "expected_evaluations": 10,
            "das_root": str(DAS_ROOT),
        }

    summary_path = OUT_ROOT / "summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[saved] {per_query_path}", flush=True)
    if paired_rows:
        print(f"[saved] {OUT_ROOT / 'paired_with_existing_das.csv'}", flush=True)
    print(f"[saved] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
