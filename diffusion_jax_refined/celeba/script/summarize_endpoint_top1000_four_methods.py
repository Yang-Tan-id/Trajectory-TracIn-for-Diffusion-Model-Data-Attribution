#!/usr/bin/env python3
"""Summarize CelebA Q0-Q9 top-1000 removal for four endpoint methods."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METHODS = (
    "endpoint_pollute_adamw_timestamp_raw_pos",
    "endpoint_pollute_adamw_timestamp_both_l2_pos",
    "endpoint_tracin_adamw_linear_raw_neg",
    "endpoint_tracin_adamw_linear_both_l2_neg",
)
SHORT = {
    METHODS[0]: "POLLUTE-RAW",
    METHODS[1]: "POLLUTE-BOTH-L2",
    METHODS[2]: "TRACIN-RAW",
    METHODS[3]: "TRACIN-BOTH-L2",
}
EXPECTED_SIGN = {
    METHODS[0]: 1,
    METHODS[1]: 1,
    METHODS[2]: -1,
    METHODS[3]: -1,
}
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
            ranking_sign = int(payload["ranking_sign"])
            if int(payload.get("topk", -1)) != TOPK:
                raise ValueError(f"{path}: topk={payload.get('topk')}, expected {TOPK}")
            if len(removed) != TOPK or len(set(removed)) != TOPK:
                raise ValueError(f"{path}: removal set is not {TOPK} unique indices")
            if ranking_sign != EXPECTED_SIGN[method]:
                raise ValueError(
                    f"{path}: ranking_sign={ranking_sign}, expected {EXPECTED_SIGN[method]}"
                )
            rows.append(
                {
                    "query_id": query_id,
                    "method": method,
                    "ranking_sign": ranking_sign,
                    "endpoint_difference": float(payload["endpoint_difference"]),
                    "trajectory_difference": float(payload["trajectory_difference"]),
                    "metrics_path": str(path.resolve()),
                }
            )

    by_key = {(int(row["query_id"]), str(row["method"])): row for row in rows}
    lines = [
        "CELEBA TOP-1000 REMOVAL (10% OF FULL TRAINING SET) — Q0-Q9",
        "LARGER DIFFERENCE IS BETTER",
        "",
        (
            "QUERY  POLLUTE-RAW-END  POLLUTE-L2-END  TRACIN-RAW-END  "
            "TRACIN-L2-END  POLLUTE-RAW-TRAJ POLLUTE-L2-TRAJ "
            "TRACIN-RAW-TRAJ TRACIN-L2-TRAJ"
        ),
        "-" * 146,
    ]
    for query_id in query_ids:
        values = [by_key[(query_id, method)] for method in METHODS]
        lines.append(
            f"Q{query_id:<3d} "
            f"{values[0]['endpoint_difference']:+16.8g} "
            f"{values[1]['endpoint_difference']:+15.8g} "
            f"{values[2]['endpoint_difference']:+15.8g} "
            f"{values[3]['endpoint_difference']:+14.8g}  "
            f"{values[0]['trajectory_difference']:+16.8g} "
            f"{values[1]['trajectory_difference']:+15.8g} "
            f"{values[2]['trajectory_difference']:+15.8g} "
            f"{values[3]['trajectory_difference']:+14.8g}"
        )

    lines.extend(["", "SUMMARY", "-" * 88])
    for target in ("endpoint_difference", "trajectory_difference"):
        lines.append(target.upper())
        means: dict[str, float] = {}
        for method in METHODS:
            values = [float(by_key[(query_id, method)][target]) for query_id in query_ids]
            means[method] = statistics.fmean(values)
            lines.append(
                f"  {SHORT[method]:18s} mean={means[method]:+.9g} "
                f"median={statistics.median(values):+.9g} "
                f"std={statistics.pstdev(values):.9g}"
            )
        wins = {method: 0 for method in METHODS}
        for query_id in query_ids:
            winner = max(METHODS, key=lambda method: by_key[(query_id, method)][target])
            wins[winner] += 1
        lines.append(
            "  wins: "
            + ", ".join(f"{SHORT[method]}={wins[method]}" for method in METHODS)
        )
        for left, right in (
            (METHODS[0], METHODS[2]),
            (METHODS[1], METHODS[3]),
            (METHODS[1], METHODS[0]),
            (METHODS[3], METHODS[2]),
        ):
            lines.append(
                f"  {SHORT[left]} minus {SHORT[right]} mean="
                f"{means[left] - means[right]:+.9g}"
            )
        lines.append("")

    report = "\n".join(lines).rstrip() + "\n"
    print(report, end="")

    prefix = args.output_prefix or args.root / "top1000_q0_q9_four_method_summary"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    txt_path = prefix.with_suffix(".txt")
    csv_path = prefix.with_suffix(".csv")
    txt_path.write_text(report)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {txt_path}")
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
