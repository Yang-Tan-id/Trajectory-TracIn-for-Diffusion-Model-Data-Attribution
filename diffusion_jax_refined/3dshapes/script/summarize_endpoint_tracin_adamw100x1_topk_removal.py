#!/usr/bin/env python3
"""Summarize Endpoint-TracIn AdamW-full 100x1 top-k removal results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METHOD = "endpoint_tracin_adamw_full_100x1_raw_neg"
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
    for topk in TOPKS:
        for query_id in query_ids:
            matches = sorted(
                (root / METHOD / f"topk_{topk:04d}").glob(
                    f"query_{query_id:03d}_seed_*/counterfactual_metrics.json"
                )
            )
            if len(matches) != 1:
                missing.append((topk, query_id, len(matches)))
                continue
            payload = json.loads(matches[0].read_text())
            rows.append(
                {
                    "method": METHOD,
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
            f"topk={topk} Q{query_id}: found {count}"
            for topk, query_id, count in missing[:20]
        )
        raise RuntimeError(f"Missing {len(missing)} results; first entries:\n{preview}")

    output = root / f"{METHOD}_counterfactual_results.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("METHOD                                      TOPK    ENDPOINT mean±std       TRAJECTORY mean±std")
    print("-" * 104)
    for topk in TOPKS:
        selected = [row for row in rows if row["topk"] == topk]
        endpoint = [row["endpoint_difference"] for row in selected]
        trajectory = [row["trajectory_difference"] for row in selected]
        print(
            f"{METHOD:<44s} {topk:4d} "
            f"{statistics.fmean(endpoint):12.6g}±{statistics.stdev(endpoint):10.6g} "
            f"{statistics.fmean(trajectory):12.6g}±{statistics.stdev(trajectory):10.6g}"
        )
    print(f"\n[saved] {output}")


if __name__ == "__main__":
    main()
