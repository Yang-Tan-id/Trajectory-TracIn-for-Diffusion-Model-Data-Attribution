#!/usr/bin/env python3
"""Summarize CelebA Q0-Q9 top-1000 query-L2 removal for two endpoint methods."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METHODS = (
    "endpoint_pollute_adamw_timestamp_query_l2_pos",
    "endpoint_tracin_adamw_linear_query_l2_neg",
)
SHORT = {
    METHODS[0]: "POLLUTE-QUERY-L2",
    METHODS[1]: "TRACIN-QUERY-L2",
}
EXPECTED_SIGN = {METHODS[0]: 1, METHODS[1]: -1}
TOPK = 1000


def parse_ids(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def find_metric(root: Path, method: str, query_id: int) -> Path:
    matches = sorted(
        (root / method).glob(
            f"query_{query_id:03d}_seed_*/counterfactual_metrics.json"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one metric for {method} Q{query_id}, found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()

    query_ids = parse_ids(args.query_ids)
    rows: list[dict[str, object]] = []
    for query_id in query_ids:
        for method in METHODS:
            path = find_metric(args.root, method, query_id)
            payload = json.loads(path.read_text())
            removed = [int(value) for value in payload.get("removed_dataset_indices", [])]
            if int(payload.get("topk", -1)) != TOPK:
                raise ValueError(f"{path}: expected topk={TOPK}")
            if len(removed) != TOPK or len(set(removed)) != TOPK:
                raise ValueError(f"{path}: removal set is not {TOPK} unique indices")
            if int(payload["ranking_sign"]) != EXPECTED_SIGN[method]:
                raise ValueError(f"{path}: unexpected ranking sign")
            rows.append(
                {
                    "query_id": query_id,
                    "method": method,
                    "endpoint_difference": float(payload["endpoint_difference"]),
                    "trajectory_difference": float(payload["trajectory_difference"]),
                    "metrics_path": str(path.resolve()),
                }
            )

    by_key = {(int(row["query_id"]), str(row["method"])): row for row in rows}
    lines = [
        "CELEBA TOP-1000 QUERY-L2 REMOVAL — Q0-Q9",
        "LARGER DIFFERENCE IS BETTER",
        "",
        "QUERY  POLLUTE-END  TRACIN-END  DELTA      POLLUTE-TRAJ TRACIN-TRAJ DELTA",
        "-" * 86,
    ]
    for query_id in query_ids:
        left = by_key[(query_id, METHODS[0])]
        right = by_key[(query_id, METHODS[1])]
        lines.append(
            f"Q{query_id:<3d} "
            f"{left['endpoint_difference']:+12.8g} {right['endpoint_difference']:+11.8g} "
            f"{left['endpoint_difference'] - right['endpoint_difference']:+10.8g}  "
            f"{left['trajectory_difference']:+12.8g} {right['trajectory_difference']:+11.8g} "
            f"{left['trajectory_difference'] - right['trajectory_difference']:+10.8g}"
        )

    lines.extend(["", "SUMMARY", "-" * 70])
    for target in ("endpoint_difference", "trajectory_difference"):
        left_values = [float(by_key[(qid, METHODS[0])][target]) for qid in query_ids]
        right_values = [float(by_key[(qid, METHODS[1])][target]) for qid in query_ids]
        deltas = [left - right for left, right in zip(left_values, right_values)]
        lines.extend(
            [
                target.upper(),
                f"  POLLUTE mean={statistics.fmean(left_values):+.9g} median={statistics.median(left_values):+.9g}",
                f"  TRACIN  mean={statistics.fmean(right_values):+.9g} median={statistics.median(right_values):+.9g}",
                f"  paired delta mean={statistics.fmean(deltas):+.9g}",
                f"  wins: POLLUTE={sum(value > 0 for value in deltas)}, TRACIN={sum(value < 0 for value in deltas)}, ties={sum(value == 0 for value in deltas)}",
                "",
            ]
        )

    report = "\n".join(lines).rstrip() + "\n"
    print(report, end="")
    prefix = args.output_prefix or args.root / "top1000_q0_q9_query_l2_summary"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".txt").write_text(report)
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {prefix.with_suffix('.txt')}")
    print(f"[saved] {prefix.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
