#!/usr/bin/env python3
"""Compare five CelebA Q0-Q9 full-10k top-1000 removal methods."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from itertools import product
from pathlib import Path


METHODS = (
    "endpoint_pollute_adamw_timestamp_query_l2_full10k_pos",
    "endpoint_pollute_adamw_timestamp_both_l2_full10k_pos",
    "endpoint_tracin_adamw_linear_query_l2_full10k_neg",
    "endpoint_tracin_adamw_linear_both_l2_full10k_neg",
    "retrac_adamw_query_train_l2_full10k_neg",
)
SHORT = {
    METHODS[0]: "P-Q-L2",
    METHODS[1]: "P-BOTH-L2",
    METHODS[2]: "T-Q-L2",
    METHODS[3]: "T-BOTH-L2",
    METHODS[4]: "RETRAC-L2",
}
EXPECTED_SIGN = {
    METHODS[0]: 1,
    METHODS[1]: 1,
    METHODS[2]: -1,
    METHODS[3]: -1,
    METHODS[4]: -1,
}
COMPARISONS = (
    (METHODS[0], METHODS[2]),
    (METHODS[1], METHODS[3]),
    (METHODS[1], METHODS[4]),
    (METHODS[0], METHODS[4]),
    (METHODS[3], METHODS[4]),
    (METHODS[2], METHODS[4]),
)
TOPK = 1000


def parse_ids(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def find_metric(root: Path, method: str, query_id: int) -> Path:
    matches = sorted(
        (root / method).glob(f"query_{query_id:03d}_seed_*/counterfactual_metrics.json")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one metric for {method} Q{query_id}, found {len(matches)}"
        )
    return matches[0]


def exact_label_swap_p(differences: list[float], *, two_sided: bool) -> float:
    observed = statistics.fmean(differences)
    null = [
        statistics.fmean(sign * value for sign, value in zip(signs, differences))
        for signs in product((-1.0, 1.0), repeat=len(differences))
    ]
    tolerance = 1e-15
    if two_sided:
        return sum(abs(value) >= abs(observed) - tolerance for value in null) / len(null)
    return sum(value >= observed - tolerance for value in null) / len(null)


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
            if any(index < 0 or index >= 10_000 for index in removed):
                raise ValueError(f"{path}: removal index outside [0, 9999]")
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
        "CELEBA FULL-10K TOP-1000 REMOVAL — FIVE METHODS — Q0-Q9",
        "P=ENDPOINT-POLLUTE; T=ENDPOINT-TRACIN; LARGER DIFFERENCE IS BETTER",
        "Pollute stored score descending; TracIn/ReTrac stored score ascending.",
        "",
    ]
    for target in ("endpoint_difference", "trajectory_difference"):
        lines.extend(
            [
                target.upper(),
                "QUERY       P-Q-L2   P-BOTH-L2      T-Q-L2   T-BOTH-L2   RETRAC-L2",
                "-" * 81,
            ]
        )
        for query_id in query_ids:
            values = [float(by_key[(query_id, method)][target]) for method in METHODS]
            lines.append(f"Q{query_id:<3d} " + " ".join(f"{value:+11.8g}" for value in values))
        lines.append("-" * 81)
        for name, function in (
            ("MEAN", statistics.fmean),
            ("MEDIAN", statistics.median),
            ("STD", statistics.pstdev),
        ):
            values = [
                function(float(by_key[(query_id, method)][target]) for query_id in query_ids)
                for method in METHODS
            ]
            lines.append(f"{name:<6s}" + " ".join(f"{value:+11.8g}" for value in values))
        wins = {method: 0 for method in METHODS}
        for query_id in query_ids:
            winner = max(METHODS, key=lambda method: by_key[(query_id, method)][target])
            wins[winner] += 1
        lines.append("WINS  " + " ".join(f"{wins[method]:>11d}" for method in METHODS))
        lines.extend(["", "PAIRED EXACT LABEL-SWAP (LEFT - RIGHT)"])
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
                f"p1={exact_label_swap_p(differences, two_sided=False):.6f} "
                f"p2={exact_label_swap_p(differences, two_sided=True):.6f}"
            )
        lines.append("")

    report = "\n".join(lines).rstrip() + "\n"
    print(report, end="")
    prefix = args.output_prefix or args.root / "full10k_top1000_q0_q9_five_method_summary"
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
