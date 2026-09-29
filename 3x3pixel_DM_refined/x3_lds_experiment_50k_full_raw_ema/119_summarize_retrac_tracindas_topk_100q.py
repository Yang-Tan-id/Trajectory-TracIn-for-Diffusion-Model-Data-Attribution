"""Write per-query removal effects and pairwise method differences."""

import csv
import json
from collections import defaultdict

import numpy as np

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


METRICS = (
    "trajectory_mse",
    "trajectory_rmse",
    "trajectory_max_abs",
    "endpoint_mse",
    "endpoint_rmse",
    "endpoint_l2",
    "endpoint_max_abs",
)


def load_prepare_module():
    path = Path(__file__).with_name("117_prepare_retrac_tracindas_topk_100q.py")
    spec = spec_from_file_location("retrac_topk_prepare", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main():
    prepare = load_prepare_module()
    with open(prepare.OUT_ROOT / "jobs.json") as handle:
        jobs = json.load(handle)
    rows = []
    missing = []
    for job in jobs:
        path = Path(job["job_dir"]) / "evaluation.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        with open(path) as handle:
            row = json.load(handle)
        row["removal_fraction"] = float(job["removal_fraction"])
        row["removal_fraction_tag"] = job["removal_fraction_tag"]
        rows.append(row)
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} evaluations; first missing: {missing[0]}"
        )
    if len(rows) != 900:
        raise ValueError(f"expected 900 evaluations, found {len(rows)}")

    method_tags = [item["tag"] for item in prepare.METHODS]
    by_key = {
        (row["method_tag"], row["removal_fraction_tag"], int(row["query_id"])): row
        for row in rows
    }
    wide_rows = []
    pairwise_rows = []
    for fraction in prepare.REMOVAL_FRACTIONS:
        fraction_name = prepare.fraction_tag(fraction)
        for query_id in range(100):
            base_rows = [by_key[(method, fraction_name, query_id)] for method in method_tags]
            wide = {
                "query_id": query_id,
                "family": base_rows[0]["family"],
                "removal_fraction": float(fraction),
                "topk": int(base_rows[0]["topk"]),
            }
            for method, row in zip(method_tags, base_rows):
                for metric in METRICS:
                    wide[f"{method}__{metric}"] = float(row[metric])
            wide_rows.append(wide)
            for left_index in range(len(method_tags)):
                for right_index in range(left_index + 1, len(method_tags)):
                    left = method_tags[left_index]
                    right = method_tags[right_index]
                    pair = {
                        "query_id": query_id,
                        "family": base_rows[0]["family"],
                        "removal_fraction": float(fraction),
                        "topk": int(base_rows[0]["topk"]),
                        "left_method": left,
                        "right_method": right,
                    }
                    for metric in METRICS:
                        left_value = float(by_key[(left, fraction_name, query_id)][metric])
                        right_value = float(by_key[(right, fraction_name, query_id)][metric])
                        pair[f"left_{metric}"] = left_value
                        pair[f"right_{metric}"] = right_value
                        pair[f"difference_{metric}"] = right_value - left_value
                    pairwise_rows.append(pair)

    summary = {
        "job_count": len(rows),
        "difference_definition": "right_method minus left_method",
        "methods": {},
        "pairwise": {},
    }
    for method in method_tags:
        summary["methods"][method] = {}
        for fraction in prepare.REMOVAL_FRACTIONS:
            fraction_name = prepare.fraction_tag(fraction)
            selected = [
                by_key[(method, fraction_name, query_id)] for query_id in range(100)
            ]
            summary["methods"][method][fraction_name] = {
                metric: stats([row[metric] for row in selected]) for metric in METRICS
            }
    grouped_pairs = defaultdict(list)
    for row in pairwise_rows:
        key = (
            row["left_method"],
            row["right_method"],
            f"remove_{int(round(100 * row['removal_fraction'])):02d}pct",
        )
        grouped_pairs[key].append(row)
    for (left, right, fraction_name), selected in grouped_pairs.items():
        key = f"{right}_minus_{left}"
        summary["pairwise"].setdefault(key, {})[fraction_name] = {
            metric: stats([row[f"difference_{metric}"] for row in selected])
            for metric in METRICS
        }

    overlap_path = prepare.OUT_ROOT / "method_overlap.json"
    with open(overlap_path) as handle:
        overlap = json.load(handle)
    summary["topk_overlap"] = {}
    grouped_overlap = defaultdict(list)
    for row in overlap:
        key = (
            row["left_method"],
            row["right_method"],
            f"remove_{int(round(100 * row['removal_fraction'])):02d}pct",
        )
        grouped_overlap[key].append(row["overlap_fraction"])
    for (left, right, fraction_name), values in grouped_overlap.items():
        key = f"{left}__vs__{right}"
        summary["topk_overlap"].setdefault(key, {})[fraction_name] = stats(values)

    with open(prepare.OUT_ROOT / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(prepare.OUT_ROOT / "per_query_results.csv", "w", newline="") as handle:
        fields = [
            "query_id", "family", "method_tag", "method", "removal_fraction",
            "removal_fraction_tag", "topk", *METRICS,
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            sorted(rows, key=lambda row: (row["query_id"], row["topk"], row["method_tag"]))
        )
    with open(prepare.OUT_ROOT / "per_query_method_differences.csv", "w", newline="") as handle:
        fields = [
            "query_id", "family", "removal_fraction", "topk",
            "left_method", "right_method",
        ]
        for metric in METRICS:
            fields.extend([f"left_{metric}", f"right_{metric}", f"difference_{metric}"])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairwise_rows)

    text_path = prepare.OUT_ROOT / "per_query_endpoint_trajectory_differences.txt"
    with open(text_path, "w") as handle:
        handle.write(
            "difference = right method - left method; positive means right removal changed the generated result more\n\n"
        )
        for row in pairwise_rows:
            handle.write(
                f"q{row['query_id']:02d} {row['family']:<10s} "
                f"remove={100*row['removal_fraction']:>4.0f}% ({row['topk']:>4d}) "
                f"{row['right_method']} - {row['left_method']} | "
                f"trajectory_mse={row['difference_trajectory_mse']:+.9e} "
                f"endpoint_mse={row['difference_endpoint_mse']:+.9e} "
                f"endpoint_l2={row['difference_endpoint_l2']:+.9e}\n"
            )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[saved] {prepare.OUT_ROOT / 'summary.json'}", flush=True)
    print(f"[saved] {prepare.OUT_ROOT / 'per_query_results.csv'}", flush=True)
    print(
        f"[saved] {prepare.OUT_ROOT / 'per_query_method_differences.csv'}",
        flush=True,
    )
    print(f"[saved] {text_path}", flush=True)


if __name__ == "__main__":
    main()
