#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = ("traj_next", "traj_previous", "das_mc4_lambda1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "result" / args.experiment / "top1000_removal"
    query_ids = [int(x) for x in args.query_ids.replace(",", " ").split()]
    rows = []
    for method in METHODS:
        for query_id in query_ids:
            matches = sorted((root / method).glob(f"query_{query_id:03d}_seed_*/counterfactual_metrics.json"))
            if len(matches) != 1:
                raise RuntimeError(f"Expected one result for {method} Q{query_id}; found {len(matches)}")
            payload = json.loads(matches[0].read_text())
            rows.append(
                {
                    "method": method,
                    "query": query_id,
                    "initial_seed": int(payload["initial_seed"]),
                    "endpoint_counterfactual": float(payload["endpoint_counterfactual"]),
                    "trajectory_counterfactual": float(payload["trajectory_counterfactual"]),
                }
            )

    output = root / "counterfactual_summary.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("METHOD              QUERY      ENDPOINT        TRAJECTORY")
    print("-" * 62)
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        for row in selected:
            print(
                f"{method:<20s} Q{row['query']:<3d} "
                f"{row['endpoint_counterfactual']:14.7g} {row['trajectory_counterfactual']:14.7g}"
            )
        endpoint = sum(row["endpoint_counterfactual"] for row in selected) / len(selected)
        trajectory = sum(row["trajectory_counterfactual"] for row in selected) / len(selected)
        print(f"{method:<20s} MEAN {endpoint:14.7g} {trajectory:14.7g}\n")
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
