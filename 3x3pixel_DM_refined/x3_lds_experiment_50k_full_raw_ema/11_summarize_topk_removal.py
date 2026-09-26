"""Summarize endpoint and trajectory changes for all removal models."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from exp_config import ROOT


OUT_ROOT = ROOT / "topk_removal_retrain"
METRICS = (
    "trajectory_mse", "trajectory_rmse", "trajectory_max_abs",
    "endpoint_mse", "endpoint_rmse", "endpoint_l2", "endpoint_max_abs",
)


def main():
    with open(OUT_ROOT / "jobs.json") as handle:
        jobs = json.load(handle)
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
        raise FileNotFoundError(f"missing {len(missing)} evaluations; first: {missing[0]}")

    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method_tag"]].append(row)
    summary = {"num_models": len(rows), "methods": {}}
    for method, method_rows in sorted(by_method.items()):
        stats = {
            "count": len(method_rows),
            "score_param_source": method_rows[0]["score_param_source"],
            "eval_param_source": method_rows[0]["eval_param_source"],
        }
        for metric in METRICS:
            values = np.asarray([row[metric] for row in method_rows], dtype=np.float64)
            stats[metric] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        summary["methods"][method] = stats

    method_names = sorted(by_method)
    if len(method_names) != 2:
        raise ValueError(f"expected two methods, found {method_names}")
    left_name, right_name = method_names
    left = {int(row["query_id"]): row for row in by_method[left_name]}
    right = {int(row["query_id"]): row for row in by_method[right_name]}
    if set(left) != set(right):
        raise ValueError("the two methods do not contain the same query ids")

    paired_rows = []
    paired_summary = {
        "left_method": left_name,
        "right_method": right_name,
        "delta_definition": "right_minus_left; larger metric means larger removal effect",
        "metrics": {},
    }
    for query_id in sorted(left):
        pair = {
            "query_id": query_id,
            "family": left[query_id]["family"],
        }
        for metric in METRICS:
            left_value = float(left[query_id][metric])
            right_value = float(right[query_id][metric])
            pair[f"{left_name}_{metric}"] = left_value
            pair[f"{right_name}_{metric}"] = right_value
            pair[f"delta_{metric}"] = right_value - left_value
        paired_rows.append(pair)

    for metric in METRICS:
        deltas = np.asarray(
            [row[f"delta_{metric}"] for row in paired_rows], dtype=np.float64
        )
        paired_summary["metrics"][metric] = {
            "mean_delta": float(deltas.mean()),
            "median_delta": float(np.median(deltas)),
            "right_larger_count": int(np.sum(deltas > 0)),
            "left_larger_count": int(np.sum(deltas < 0)),
            "tie_count": int(np.sum(deltas == 0)),
            "right_larger_fraction": float(np.mean(deltas > 0)),
        }
    summary["paired_comparison"] = paired_summary

    overlap_path = OUT_ROOT / "method_overlap.json"
    if overlap_path.is_file():
        with open(overlap_path) as handle:
            overlap_rows = json.load(handle)
        overlap_counts = np.asarray(
            [row["overlap"] for row in overlap_rows], dtype=np.float64
        )
        summary["topk_overlap"] = {
            "mean_count": float(overlap_counts.mean()),
            "median_count": float(np.median(overlap_counts)),
            "mean_fraction": float(overlap_counts.mean() / 1000.0),
            "min_count": int(overlap_counts.min()),
            "max_count": int(overlap_counts.max()),
        }

    with open(OUT_ROOT / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    fieldnames = [
        "job_id", "query_id", "family", "method_tag", "method", "lambda",
        "score_param_source", "eval_param_source", "topk", "initial_seed", *METRICS,
    ]
    with open(OUT_ROOT / "per_query_results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["method_tag"], row["query_id"])))

    paired_fields = ["query_id", "family"]
    for metric in METRICS:
        paired_fields.extend(
            [
                f"{left_name}_{metric}",
                f"{right_name}_{metric}",
                f"delta_{metric}",
            ]
        )
    with open(OUT_ROOT / "paired_comparison.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=paired_fields)
        writer.writeheader()
        writer.writerows(paired_rows)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"[saved] {OUT_ROOT / 'summary.json'}", flush=True)
    print(f"[saved] {OUT_ROOT / 'per_query_results.csv'}", flush=True)
    print(f"[saved] {OUT_ROOT / 'paired_comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
