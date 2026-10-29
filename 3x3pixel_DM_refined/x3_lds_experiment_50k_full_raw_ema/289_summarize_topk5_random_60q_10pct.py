"""Summarize five attribution-guided 10% removals against random removal."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon


METRICS = (
    "noise_trajectory_mse",
    "trajectory_mse",
    "trajectory_rmse",
    "trajectory_max_abs",
    "endpoint_mse",
    "endpoint_rmse",
    "endpoint_l2",
    "endpoint_max_abs",
    "simple_loss_base",
    "simple_loss_removal",
    "simple_loss_increase",
    "simple_loss_abs_change",
)
KEY_METRICS = (
    "noise_trajectory_mse",
    "trajectory_mse",
    "endpoint_mse",
    "simple_loss_abs_change",
)


def load_prepare_module():
    path = Path(__file__).with_name("287_prepare_topk5_random_60q_10pct.py")
    spec = spec_from_file_location("topk5_random_prepare", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stats(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def paired_stats(left, right) -> dict:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    difference = left - right
    nonzero = difference[difference != 0]
    p_wilcoxon = (
        float(wilcoxon(difference, alternative="two-sided").pvalue)
        if nonzero.size
        else 1.0
    )
    wins = int(np.sum(difference > 0))
    losses = int(np.sum(difference < 0))
    p_sign = (
        float(binomtest(wins, wins + losses, 0.5).pvalue)
        if wins + losses
        else 1.0
    )
    denominator = float(np.mean(right))
    return {
        "difference": stats(difference),
        "mean_ratio": (
            float(np.mean(left) / denominator) if denominator != 0 else None
        ),
        "wins": wins,
        "ties": int(np.sum(difference == 0)),
        "losses": losses,
        "p_wilcoxon_two_sided": p_wilcoxon,
        "p_sign_two_sided": p_sign,
    }


def main() -> None:
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
        row["method_label"] = job["method_label"]
        row["prediction_sign"] = job["prediction_sign"]
        row["removal_fraction"] = float(job["removal_fraction"])
        row["removal_fraction_tag"] = job["removal_fraction_tag"]
        rows.append(row)
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} evaluations; first missing: {missing[0]}"
        )
    if len(rows) != prepare.EXPECTED_JOB_COUNT:
        raise ValueError(
            f"expected {prepare.EXPECTED_JOB_COUNT} evaluations, found {len(rows)}"
        )

    method_tags = [item["tag"] for item in prepare.METHODS]
    method_labels = {item["tag"]: item["label"] for item in prepare.METHODS}
    by_key = {(row["method_tag"], int(row["query_id"])): row for row in rows}
    if len(by_key) != prepare.EXPECTED_JOB_COUNT:
        raise ValueError("duplicate method/query evaluation keys")

    summary = {
        "job_count": len(rows),
        "query_ids": list(prepare.QUERY_IDS),
        "removal_fraction": prepare.REMOVAL_FRACTION,
        "topk": prepare.TOPK,
        "methods": {},
        "pairwise": {},
        "vs_random": {},
    }
    for method in method_tags:
        selected = [by_key[(method, query_id)] for query_id in prepare.QUERY_IDS]
        summary["methods"][method] = {
            "label": method_labels[method],
            "metrics": {
                metric: stats([row[metric] for row in selected]) for metric in METRICS
            },
        }

    pairwise_rows = []
    for left, right in combinations(method_tags, 2):
        pair_key = f"{left}_minus_{right}"
        summary["pairwise"][pair_key] = {}
        for metric in METRICS:
            left_values = [by_key[(left, q)][metric] for q in prepare.QUERY_IDS]
            right_values = [by_key[(right, q)][metric] for q in prepare.QUERY_IDS]
            summary["pairwise"][pair_key][metric] = paired_stats(
                left_values, right_values
            )
        for query_id in prepare.QUERY_IDS:
            item = {
                "query_id": query_id,
                "family": by_key[(left, query_id)]["family"],
                "left_method": left,
                "right_method": right,
            }
            for metric in METRICS:
                left_value = float(by_key[(left, query_id)][metric])
                right_value = float(by_key[(right, query_id)][metric])
                item[f"left_{metric}"] = left_value
                item[f"right_{metric}"] = right_value
                item[f"difference_{metric}"] = left_value - right_value
            pairwise_rows.append(item)

    for method in method_tags:
        if method == "random":
            continue
        summary["vs_random"][method] = {}
        for metric in METRICS:
            method_values = [by_key[(method, q)][metric] for q in prepare.QUERY_IDS]
            random_values = [by_key[("random", q)][metric] for q in prepare.QUERY_IDS]
            summary["vs_random"][method][metric] = paired_stats(
                method_values, random_values
            )

    overlap_path = prepare.OUT_ROOT / "method_overlap.json"
    with open(overlap_path) as handle:
        overlap_rows = json.load(handle)
    overlap_grouped = defaultdict(list)
    for row in overlap_rows:
        overlap_grouped[(row["left_method"], row["right_method"])].append(
            row["overlap_fraction"]
        )
    summary["topk_overlap"] = {
        f"{left}__vs__{right}": stats(values)
        for (left, right), values in overlap_grouped.items()
    }

    with open(prepare.OUT_ROOT / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(prepare.OUT_ROOT / "per_query_results.csv", "w", newline="") as handle:
        fields = [
            "query_id",
            "family",
            "method_tag",
            "method_label",
            "method",
            "prediction_sign",
            "removal_fraction",
            "removal_fraction_tag",
            "topk",
            *METRICS,
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["query_id"], row["method_tag"])))
    with open(
        prepare.OUT_ROOT / "per_query_method_differences.csv", "w", newline=""
    ) as handle:
        fields = ["query_id", "family", "left_method", "right_method"]
        for metric in METRICS:
            fields.extend(
                [f"left_{metric}", f"right_{metric}", f"difference_{metric}"]
            )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairwise_rows)

    report_path = prepare.OUT_ROOT / "summary.txt"
    with open(report_path, "w") as handle:
        handle.write(
            "X3 Q00-Q59 TRUE 10% REMOVAL: FIVE ATTRIBUTION METHODS + RANDOM\n"
        )
        handle.write(
            "All values are true EMA-model removal effects. p-values are paired, "
            "two-sided Wilcoxon tests against the query-matched random control.\n\n"
        )
        for metric in KEY_METRICS:
            handle.write(f"METRIC: {metric}\n")
            handle.write(
                f"{'method':32s} {'mean +/- std':27s} {'ratio/random':>13s} "
                f"{'wins':>8s} {'p-wilcoxon':>13s}\n"
            )
            for method in method_tags:
                metric_stats = summary["methods"][method]["metrics"][metric]
                if method == "random":
                    ratio = 1.0
                    wins = "--"
                    p_value = "--"
                else:
                    comparison = summary["vs_random"][method][metric]
                    ratio = comparison["mean_ratio"]
                    wins = f"{comparison['wins']}/60"
                    p_value = f"{comparison['p_wilcoxon_two_sided']:.6g}"
                handle.write(
                    f"{method_labels[method]:32s} "
                    f"{metric_stats['mean']:.6e} +/- {metric_stats['std']:.3e} "
                    f"{ratio:13.4f} {wins:>8s} {p_value:>13s}\n"
                )
            handle.write("\n")

    print(report_path.read_text(), flush=True)
    print(f"[saved] {prepare.OUT_ROOT / 'summary.json'}", flush=True)
    print(f"[saved] {prepare.OUT_ROOT / 'per_query_results.csv'}", flush=True)
    print(
        f"[saved] {prepare.OUT_ROOT / 'per_query_method_differences.csv'}",
        flush=True,
    )
    print(f"[saved] {report_path}", flush=True)


if __name__ == "__main__":
    main()
