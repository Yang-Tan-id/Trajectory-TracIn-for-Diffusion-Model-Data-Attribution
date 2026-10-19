#!/usr/bin/env python3
"""Summarize CelebA Q0-Q9 top-1000 removal across six endpoint variants."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from itertools import product
from pathlib import Path


METHODS = (
    "endpoint_pollute_adamw_timestamp_raw_pos",
    "endpoint_pollute_adamw_timestamp_query_l2_pos",
    "endpoint_pollute_adamw_timestamp_both_l2_pos",
    "endpoint_tracin_adamw_linear_raw_neg",
    "endpoint_tracin_adamw_linear_query_l2_neg",
    "endpoint_tracin_adamw_linear_both_l2_neg",
)
SHORT = {
    METHODS[0]: "P-RAW",
    METHODS[1]: "P-Q-L2",
    METHODS[2]: "P-BOTH-L2",
    METHODS[3]: "T-RAW",
    METHODS[4]: "T-Q-L2",
    METHODS[5]: "T-BOTH-L2",
}
EXPECTED_SIGN = {
    METHODS[0]: 1,
    METHODS[1]: 1,
    METHODS[2]: 1,
    METHODS[3]: -1,
    METHODS[4]: -1,
    METHODS[5]: -1,
}
COMPARISONS = (
    (METHODS[0], METHODS[3]),
    (METHODS[1], METHODS[4]),
    (METHODS[2], METHODS[5]),
    (METHODS[1], METHODS[0]),
    (METHODS[2], METHODS[0]),
    (METHODS[1], METHODS[2]),
    (METHODS[4], METHODS[3]),
    (METHODS[5], METHODS[3]),
    (METHODS[4], METHODS[5]),
)
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


def exact_sign_flip_p(differences: list[float], alternative: str) -> float:
    observed = statistics.fmean(differences)
    null = [
        statistics.fmean(sign * value for sign, value in zip(signs, differences))
        for signs in product((-1.0, 1.0), repeat=len(differences))
    ]
    tolerance = 1e-15
    if alternative == "greater":
        return sum(value >= observed - tolerance for value in null) / len(null)
    return sum(abs(value) >= abs(observed) - tolerance for value in null) / len(null)


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
        "CELEBA TOP-1000 REMOVAL — RAW vs QUERY-L2 vs BOTH-L2 — Q0-Q9",
        "P=ENDPOINT-POLLUTE; T=ENDPOINT-TRACIN; LARGER DIFFERENCE IS BETTER",
        "",
    ]
    for target in ("endpoint_difference", "trajectory_difference"):
        lines.extend(
            [
                target.upper(),
                "QUERY       P-RAW      P-Q-L2   P-BOTH-L2       T-RAW      T-Q-L2   T-BOTH-L2",
                "-" * 91,
            ]
        )
        for query_id in query_ids:
            values = [float(by_key[(query_id, method)][target]) for method in METHODS]
            lines.append(
                f"Q{query_id:<3d} " + " ".join(f"{value:+11.8g}" for value in values)
            )
        lines.append("-" * 91)
        for statistic_name, statistic in (
            ("MEAN", statistics.fmean),
            ("MEDIAN", statistics.median),
            ("STD", statistics.pstdev),
        ):
            values = [
                statistic(float(by_key[(query_id, method)][target]) for query_id in query_ids)
                for method in METHODS
            ]
            prefix = " " if statistic_name == "STD" else "+"
            lines.append(
                f"{statistic_name:<5s} "
                + " ".join(
                    f"{value:{prefix}11.8g}" if statistic_name != "STD" else f"{value:11.8g}"
                    for value in values
                )
            )
        wins = {method: 0 for method in METHODS}
        for query_id in query_ids:
            winner = max(METHODS, key=lambda method: by_key[(query_id, method)][target])
            wins[winner] += 1
        lines.append("WINS  " + " ".join(f"{wins[method]:>11d}" for method in METHODS))
        lines.extend(["", "PAIRED EXACT SIGN-FLIP COMPARISONS (LEFT - RIGHT)"])
        for left, right in COMPARISONS:
            differences = [
                float(by_key[(query_id, left)][target])
                - float(by_key[(query_id, right)][target])
                for query_id in query_ids
            ]
            lines.append(
                f"  {SHORT[left]:10s} - {SHORT[right]:10s}: "
                f"mean={statistics.fmean(differences):+.9g} "
                f"wins={sum(value > 0 for value in differences)}/{len(differences)} "
                f"p1={exact_sign_flip_p(differences, 'greater'):.6f} "
                f"p2={exact_sign_flip_p(differences, 'two-sided'):.6f}"
            )
        lines.append("")

    report = "\n".join(lines).rstrip() + "\n"
    print(report, end="")
    prefix = args.output_prefix or args.root / "top1000_q0_q9_six_method_summary"
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
