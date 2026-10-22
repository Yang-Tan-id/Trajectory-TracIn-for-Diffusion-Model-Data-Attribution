#!/usr/bin/env python3
"""Summarize CLIP comparisons for four full-10k top-1000 removal models."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METHODS = (
    "endpoint_pollute_adamw_timestamp_query_l2_full10k_pos",
    "endpoint_pollute_adamw_timestamp_both_l2_full10k_pos",
    "endpoint_tracin_adamw_linear_query_l2_full10k_neg",
    "endpoint_tracin_adamw_linear_both_l2_full10k_neg",
)
SHORT = {
    METHODS[0]: "P-Q-L2",
    METHODS[1]: "P-BOTH-L2",
    METHODS[2]: "T-Q-L2",
    METHODS[3]: "T-BOTH-L2",
}


def parse_ids(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    args = parser.parse_args()

    rows: list[dict] = []
    for query_id in parse_ids(args.query_ids):
        matches = sorted(args.root.glob(f"query_{query_id:03d}_seed_*/clip_metrics.json"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Q{query_id}: expected one clip_metrics.json, found {len(matches)}")
        payload = json.loads(matches[0].read_text())
        records = {record["method"]: record for record in payload["records"]}
        for method in METHODS:
            if method not in records:
                raise KeyError(f"{matches[0]}: missing {method}")
            row = dict(records[method])
            row["clip_model"] = payload["clip_model"]
            row["metrics_path"] = str(matches[0].resolve())
            rows.append(row)

    by_key = {(int(row["query_id"]), row["method"]): row for row in rows}
    query_ids = parse_ids(args.query_ids)
    lines = [
        "CELEBA FULL-10K TOP-1000 REMOVAL — CLIP IMAGE COMPARISON",
        "CLIP(original) higher means more semantically similar to the original sample.",
        "CLIP distance = 1 - CLIP(original); larger means a larger semantic change.",
        "",
        "QUERY       P-Q-L2   P-BOTH-L2      T-Q-L2   T-BOTH-L2",
        "-" * 67,
    ]
    for query_id in query_ids:
        values = [by_key[(query_id, method)]["clip_original_cosine"] for method in METHODS]
        lines.append(f"Q{query_id:<3d} " + " ".join(f"{value:+11.6f}" for value in values))
    lines.append("-" * 67)
    for label, key in (
        ("MEAN COS", "clip_original_cosine"),
        ("MEAN DIST", "clip_original_distance"),
        ("MEAN TXT", "clip_prompt_cosine"),
    ):
        values = [
            statistics.fmean(float(by_key[(query_id, method)][key]) for query_id in query_ids)
            for method in METHODS
        ]
        lines.append(f"{label:<9s}" + " ".join(f"{value:+11.6f}" for value in values))
    report = "\n".join(lines) + "\n"
    print(report, end="")

    txt_path = args.root / "clip_summary_q0_q9.txt"
    csv_path = args.root / "clip_per_query_q0_q9.csv"
    txt_path.write_text(report)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {txt_path}")
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
