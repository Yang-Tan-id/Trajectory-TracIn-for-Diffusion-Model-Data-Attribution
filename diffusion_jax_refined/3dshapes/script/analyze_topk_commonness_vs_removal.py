#!/usr/bin/env python3
"""Relate cross-query removal-set commonness to counterfactual damage.

The analysis deliberately reads ``removal_manifest.json`` rather than
reconstructing rankings from score files.  Consequently every set analyzed
here is exactly the set that was excluded during its corresponding retrain.
No model training or evaluation is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "retrac_adamw_both_l2_neg",
    "endpoint_pollute_adamw_timestamp_train_l2",
    "endpoint_tracin_adamw_full_100x1_raw_neg",
)
METHOD_LABELS = {
    "retrac_adamw_both_l2_neg": "ReTrac",
    "endpoint_pollute_adamw_timestamp_train_l2": "Endpoint-Pollute",
    "endpoint_tracin_adamw_full_100x1_raw_neg": "Endpoint-TracIn",
}


def parse_ids(text: str) -> list[int]:
    values = [int(token) for token in text.replace(",", " ").split()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("query ids must be nonempty and unique")
    return values


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 2:
        return float("nan")
    left = left - left.mean()
    right = right - right.mean()
    denom = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denom) if denom else float("nan")


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return pearson(average_ranks(left), average_ranks(right))


def one_match(root: Path, query_id: int, filename: str) -> Path | None:
    matches = sorted(root.glob(f"query_{query_id:03d}_seed_*/{filename}"))
    if len(matches) > 1:
        raise RuntimeError(
            f"ambiguous Q{query_id} {filename} under {root}: {len(matches)} matches"
        )
    return matches[0] if matches else None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def correlations(
    rows: list[dict[str, object]], left: str, right: str
) -> tuple[float, float]:
    x = np.asarray([float(row[left]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row[right]) for row in rows], dtype=np.float64)
    return pearson(x, y), spearman(x, y)


def fmt(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:+.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--query-ids", type=parse_ids, default=list(range(60)))
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--candidate-size", type=int, default=5000)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    unknown = [value for value in methods if value not in METHOD_LABELS]
    if unknown:
        parser.error(f"unknown methods: {unknown}")
    if not (0 < args.topk <= args.candidate_size):
        parser.error("topk must be in [1, candidate-size]")

    root = (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "retrac_endpoint_topk_removal_60q"
    )
    output_dir = args.output_dir or (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "analysis"
        / f"top{args.topk}_commonness_vs_removal"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    per_method: dict[str, dict[int, dict[str, object]]] = {}
    rows: list[dict[str, object]] = []
    report: list[str] = []

    for method in methods:
        method_root = root / method / f"topk_{args.topk:04d}"
        records: dict[int, dict[str, object]] = {}
        for query_id in args.query_ids:
            manifest_path = one_match(method_root, query_id, "removal_manifest.json")
            if manifest_path is None:
                continue
            manifest = json.loads(manifest_path.read_text())
            removed = np.asarray(
                manifest["removed_dataset_indices"], dtype=np.int64
            ).reshape(-1)
            if len(removed) != args.topk or len(np.unique(removed)) != args.topk:
                raise ValueError(
                    f"{manifest_path}: expected {args.topk} unique removed indices"
                )
            if np.any((removed < 0) | (removed >= args.candidate_size)):
                raise ValueError(f"{manifest_path}: removed index outside candidate universe")
            metrics_path = one_match(method_root, query_id, "counterfactual_metrics.json")
            metrics = json.loads(metrics_path.read_text()) if metrics_path else None
            records[query_id] = {
                "removed": removed,
                "manifest": manifest,
                "metrics": metrics,
            }

        if len(records) < 2:
            report.append(f"{METHOD_LABELS[method]}: only {len(records)} manifests; skipped")
            continue

        query_ids = sorted(records)
        frequency = np.zeros(args.candidate_size, dtype=np.int64)
        for query_id in query_ids:
            frequency[np.asarray(records[query_id]["removed"])] += 1

        # This is a query-independent consensus ranking.  It is diagnostic
        # only: no counterfactual model has been trained for this set.
        global_order = np.lexsort((np.arange(args.candidate_size), -frequency))
        global_topk = global_order[: args.topk]
        np.save(output_dir / f"{method}_global_frequency_top{args.topk}.npy", global_topk)

        method_rows: list[dict[str, object]] = []
        for query_id in query_ids:
            removed = np.asarray(records[query_id]["removed"], dtype=np.int64)
            other_frequency = frequency.copy()
            other_frequency[removed] -= 1
            other_count = len(query_ids) - 1
            commonness = float(other_frequency[removed].mean() / other_count)

            # Leave this query out when constructing its global-frequency set,
            # avoiding the mechanical self-overlap bias.
            loo_order = np.lexsort(
                (np.arange(args.candidate_size), -other_frequency)
            )
            loo_global = loo_order[: args.topk]
            loo_overlap = float(np.intersect1d(removed, loo_global).size / args.topk)

            metrics = records[query_id]["metrics"]
            row: dict[str, object] = {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "query": query_id,
                "manifest_queries": len(query_ids),
                "mean_other_query_selection_rate": commonness,
                "loo_global_topk_overlap": loo_overlap,
                "has_metrics": int(metrics is not None),
                "endpoint_difference": (
                    float(metrics["endpoint_difference"]) if metrics else float("nan")
                ),
                "trajectory_difference": (
                    float(metrics["trajectory_difference"]) if metrics else float("nan")
                ),
            }
            method_rows.append(row)
            rows.append(row)

        per_method[method] = records
        metric_rows = [row for row in method_rows if int(row["has_metrics"])]
        common_values = np.asarray(
            [float(row["mean_other_query_selection_rate"]) for row in method_rows]
        )
        overlap_values = np.asarray(
            [float(row["loo_global_topk_overlap"]) for row in method_rows]
        )
        report.extend(
            [
                METHOD_LABELS[method].upper(),
                f"  manifests={len(method_rows)} metrics={len(metric_rows)}",
                (
                    "  commonness mean/median="
                    f"{common_values.mean():.2%}/{np.median(common_values):.2%}"
                ),
                (
                    "  leave-one-out global-topK overlap mean/median="
                    f"{overlap_values.mean():.2%}/{np.median(overlap_values):.2%}"
                ),
            ]
        )
        for target in ("endpoint_difference", "trajectory_difference"):
            for feature in (
                "mean_other_query_selection_rate",
                "loo_global_topk_overlap",
            ):
                p, s = correlations(metric_rows, feature, target)
                report.append(
                    f"  {feature} vs {target}: Pearson={fmt(p)} Spearman={fmt(s)}"
                )
        report.append("")

    paired_rows: list[dict[str, object]] = []
    retrac_name = METHODS[0]
    pollute_name = METHODS[1]
    if retrac_name in per_method and pollute_name in per_method:
        row_lookup = {(str(row["method"]), int(row["query"])): row for row in rows}
        common_queries = sorted(
            set(per_method[retrac_name]) & set(per_method[pollute_name])
        )
        for query_id in common_queries:
            retrac = row_lookup[(retrac_name, query_id)]
            pollute = row_lookup[(pollute_name, query_id)]
            if not int(retrac["has_metrics"]) or not int(pollute["has_metrics"]):
                continue
            paired_rows.append(
                {
                    "query": query_id,
                    "retrac_commonness": retrac["mean_other_query_selection_rate"],
                    "pollute_commonness": pollute["mean_other_query_selection_rate"],
                    "commonness_delta_retrac_minus_pollute": (
                        float(retrac["mean_other_query_selection_rate"])
                        - float(pollute["mean_other_query_selection_rate"])
                    ),
                    "retrac_global_overlap": retrac["loo_global_topk_overlap"],
                    "pollute_global_overlap": pollute["loo_global_topk_overlap"],
                    "endpoint_advantage_retrac_minus_pollute": (
                        float(retrac["endpoint_difference"])
                        - float(pollute["endpoint_difference"])
                    ),
                    "trajectory_advantage_retrac_minus_pollute": (
                        float(retrac["trajectory_difference"])
                        - float(pollute["trajectory_difference"])
                    ),
                }
            )

        report.extend(["PAIRED RETRAC ADVANTAGE", f"  paired metrics={len(paired_rows)}"])
        for target in (
            "endpoint_advantage_retrac_minus_pollute",
            "trajectory_advantage_retrac_minus_pollute",
        ):
            for feature in (
                "retrac_commonness",
                "retrac_global_overlap",
                "commonness_delta_retrac_minus_pollute",
            ):
                p, s = correlations(paired_rows, feature, target)
                report.append(
                    f"  {feature} vs {target}: Pearson={fmt(p)} Spearman={fmt(s)}"
                )

    write_csv(output_dir / "per_query_commonness_and_removal.csv", rows)
    write_csv(output_dir / "paired_retrac_advantage.csv", paired_rows)
    report.extend(
        [
            "",
            "NOTE",
            "  global-frequency top-K is diagnostic only; it has no removal metric",
            "  until a model is actually retrained after excluding that fixed set.",
            f"[saved] {output_dir / 'per_query_commonness_and_removal.csv'}",
            f"[saved] {output_dir / 'paired_retrac_advantage.csv'}",
        ]
    )
    report_text = "\n".join(report) + "\n"
    (output_dir / "report.txt").write_text(report_text)
    print(report_text, end="")


if __name__ == "__main__":
    main()
