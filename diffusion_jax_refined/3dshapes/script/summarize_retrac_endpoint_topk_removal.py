#!/usr/bin/env python3
"""Summarize the paired 60-query top-k removal experiment."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METHODS = (
    "retrac_adamw_both_l2_neg",
    "endpoint_pollute_adamw_timestamp_train_l2",
)
TOPKS = (400, 1000)


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--query-ids", default=",".join(map(str, range(60))))
    args = parser.parse_args()
    query_ids = parse_ints(args.query_ids)
    root = (
        Path(__file__).resolve().parents[1]
        / "result"
        / args.experiment
        / "retrac_endpoint_topk_removal_60q"
    )
    rows = []
    missing = []
    for method in METHODS:
        for topk in TOPKS:
            for query_id in query_ids:
                matches = sorted(
                    (root / method / f"topk_{topk:04d}").glob(
                        f"query_{query_id:03d}_seed_*/counterfactual_metrics.json"
                    )
                )
                if len(matches) != 1:
                    missing.append((method, topk, query_id, len(matches)))
                    continue
                payload = json.loads(matches[0].read_text())
                rows.append(
                    {
                        "method": method,
                        "topk": topk,
                        "fraction_of_full_training_set": topk / 20_000,
                        "query": query_id,
                        "initial_seed": int(payload["initial_seed"]),
                        "endpoint_difference": float(payload["endpoint_difference"]),
                        "trajectory_difference": float(payload["trajectory_difference"]),
                        "checkpoint": payload["removal_checkpoint"],
                    }
                )
    if missing:
        preview = "\n".join(
            f"{method} topk={topk} Q{query_id}: found {count}"
            for method, topk, query_id, count in missing[:20]
        )
        raise RuntimeError(f"Missing {len(missing)} results; first entries:\n{preview}")

    output = root / "counterfactual_results.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("METHOD                                      TOPK    ENDPOINT mean±std       TRAJECTORY mean±std")
    print("-" * 104)
    for method in METHODS:
        for topk in TOPKS:
            selected = [
                row for row in rows if row["method"] == method and row["topk"] == topk
            ]
            endpoint = [row["endpoint_difference"] for row in selected]
            trajectory = [row["trajectory_difference"] for row in selected]
            print(
                f"{method:<44s} {topk:4d} "
                f"{statistics.fmean(endpoint):12.6g}±{statistics.stdev(endpoint):10.6g} "
                f"{statistics.fmean(trajectory):12.6g}±{statistics.stdev(trajectory):10.6g}"
            )

    paired_rows = []
    print("\nPAIRED DIFFERENCE: ReTrac(-score) minus endpoint-pollute(raw score)")
    print("TOPK    ENDPOINT mean±std       TRAJECTORY mean±std")
    print("-" * 62)
    for topk in TOPKS:
        lookup = {
            (row["method"], row["query"]): row
            for row in rows
            if row["topk"] == topk
        }
        endpoint_delta = []
        trajectory_delta = []
        for query_id in query_ids:
            retrac = lookup[(METHODS[0], query_id)]
            endpoint = lookup[(METHODS[1], query_id)]
            ed = retrac["endpoint_difference"] - endpoint["endpoint_difference"]
            td = retrac["trajectory_difference"] - endpoint["trajectory_difference"]
            endpoint_delta.append(ed)
            trajectory_delta.append(td)
            paired_rows.append(
                {
                    "topk": topk,
                    "query": query_id,
                    "retrac_minus_endpoint_endpoint_difference": ed,
                    "retrac_minus_endpoint_trajectory_difference": td,
                }
            )
        print(
            f"{topk:4d} "
            f"{statistics.fmean(endpoint_delta):12.6g}±{statistics.stdev(endpoint_delta):10.6g} "
            f"{statistics.fmean(trajectory_delta):12.6g}±{statistics.stdev(trajectory_delta):10.6g}"
        )

    paired_output = root / "paired_method_differences.csv"
    with paired_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    print(f"\n[saved] {output}")
    print(f"[saved] {paired_output}")


if __name__ == "__main__":
    main()
