#!/usr/bin/env python3
"""Export overlap/uniqueness for ten completed x3 true-removal arms."""

from __future__ import annotations

import argparse
import json
import statistics
from itertools import combinations
from pathlib import Path

import numpy as np

from exp_config import ROOT


QUERY_IDS = tuple(range(60))
TOPK = 5_000
DATASET_SIZE = 50_000
FRACTION = "remove_10pct"

ADAMW_ROOT = ROOT / (
    "topk_removal_pint_endpoint_retrac_random_60q_10pct_"
    "final_endpoint_linear_adamw_sign"
)
OLD_RAW_ROOT = ROOT / "topk_removal_retrac_endtracin_tracindas_100q"

METHODS = (
    (
        "PINT / RAW",
        "RAW-GRADIENT",
        OLD_RAW_ROOT,
        "tracin_das_aligned_timestamp_square",
    ),
    ("PINT / RAW", "FULL-ADAMW", ADAMW_ROOT, "pint_raw"),
    (
        "PINT / BOTH-L2",
        "RAW-GRADIENT",
        ROOT / "topk_removal_raw_gradient_pint_both_endpoint_mc1_60q_10pct",
        "pint_both_l2_raw_gradient",
    ),
    ("PINT / BOTH-L2", "FULL-ADAMW", ADAMW_ROOT, "pint_both_l2"),
    (
        "ENDPOINT TRACIN / RAW",
        "RAW-GRADIENT",
        ROOT / "topk_removal_endpoint_raw_gradient_mc10_raw_60q_10pct",
        "endpoint_mc10_raw_gradient_raw",
    ),
    (
        "ENDPOINT TRACIN / RAW",
        "FULL-ADAMW",
        ADAMW_ROOT,
        "endpoint_tracin_raw",
    ),
    (
        "ENDPOINT TRACIN / BOTH-L2",
        "RAW-GRADIENT",
        ROOT / "topk_removal_endpoint_raw_gradient_mc10_both_l2_60q_10pct",
        "endpoint_mc10_raw_gradient_both_l2",
    ),
    (
        "ENDPOINT TRACIN / BOTH-L2",
        "FULL-ADAMW",
        ADAMW_ROOT,
        "endpoint_tracin_both_l2",
    ),
    (
        "RETRAC / BOTH-L2",
        "RAW-GRADIENT",
        OLD_RAW_ROOT,
        "diffusion_retrac_normalized",
    ),
    ("RETRAC / BOTH-L2", "FULL-ADAMW", ADAMW_ROOT, "retrac_both_l2"),
)


def job_dir(root: Path, tag: str, query_id: int) -> Path:
    return root / tag / FRACTION / f"q{query_id:02d}"


def load_removed(root: Path, tag: str, query_id: int) -> set[int]:
    path = job_dir(root, tag, query_id) / "removed_indices.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.asarray(np.load(path), dtype=np.int64).reshape(-1)
    if len(values) != TOPK or len(np.unique(values)) != TOPK:
        raise ValueError(f"expected {TOPK} unique indices: {path}")
    if np.any(values < 0) or np.any(values >= DATASET_SIZE):
        raise ValueError(f"index outside [0, {DATASET_SIZE}): {path}")
    return set(map(int, values))


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "topk_removed_set_overlap_raw_vs_adamw_60q.txt",
    )
    args = parser.parse_args()

    selections: dict[tuple[str, str], dict[int, set[int]]] = {}
    for method, transform, root, tag in METHODS:
        key = (method, transform)
        selections[key] = {
            query_id: load_removed(root, tag, query_id)
            for query_id in QUERY_IDS
        }
        print(f"[loaded] {transform:12s} {method}", flush=True)

    summaries: list[dict[str, object]] = []
    for method, transform, _, _ in METHODS:
        key = (method, transform)
        sets = selections[key]
        overlaps: list[float] = []
        jaccards: list[float] = []
        for left_query, right_query in combinations(QUERY_IDS, 2):
            count = len(sets[left_query] & sets[right_query])
            overlaps.append(count / TOPK)
            jaccards.append(count / (2 * TOPK - count))
        union = set().union(*(sets[query_id] for query_id in QUERY_IDS))
        intersection = set.intersection(
            *(sets[query_id] for query_id in QUERY_IDS)
        )
        summaries.append(
            {
                "method": method,
                "transform": transform,
                "overlap": describe(overlaps),
                "unique_mean": 1.0 - statistics.fmean(overlaps),
                "jaccard_mean": statistics.fmean(jaccards),
                "union_count": len(union),
                "all_query_intersection_count": len(intersection),
            }
        )

    paired: list[dict[str, object]] = []
    conceptual_methods = tuple(dict.fromkeys(method for method, *_ in METHODS))
    for method in conceptual_methods:
        raw = selections[(method, "RAW-GRADIENT")]
        adamw = selections[(method, "FULL-ADAMW")]
        same_query_overlap: list[float] = []
        same_query_jaccard: list[float] = []
        for query_id in QUERY_IDS:
            count = len(raw[query_id] & adamw[query_id])
            same_query_overlap.append(count / TOPK)
            same_query_jaccard.append(count / (2 * TOPK - count))
        raw_summary = next(
            row
            for row in summaries
            if row["method"] == method and row["transform"] == "RAW-GRADIENT"
        )
        adamw_summary = next(
            row
            for row in summaries
            if row["method"] == method and row["transform"] == "FULL-ADAMW"
        )
        raw_overlap = float(raw_summary["overlap"]["mean"])
        adamw_overlap = float(adamw_summary["overlap"]["mean"])
        paired.append(
            {
                "method": method,
                "raw_cross_query_overlap": raw_overlap,
                "adamw_cross_query_overlap": adamw_overlap,
                "adamw_minus_raw_overlap": adamw_overlap - raw_overlap,
                "raw_unique": 1.0 - raw_overlap,
                "adamw_unique": 1.0 - adamw_overlap,
                "adamw_minus_raw_unique": raw_overlap - adamw_overlap,
                "same_query_raw_adamw_overlap": describe(same_query_overlap),
                "same_query_raw_adamw_jaccard_mean": statistics.fmean(
                    same_query_jaccard
                ),
            }
        )

    lines = [
        "X3 TRUE TOP-10% REMOVAL SET UNIQUENESS — Q00-Q59",
        f"dataset={DATASET_SIZE} removed={TOPK} queries={len(QUERY_IDS)}",
        f"cross-query pairs per arm={len(QUERY_IDS) * (len(QUERY_IDS) - 1) // 2}",
        f"independent-random overlap expectation={TOPK / DATASET_SIZE:.2%}",
        "All values are computed from the saved removed_indices.npy files.",
        "",
        "CROSS-QUERY UNIQUENESS WITHIN EACH ARM",
        (
            f"{'METHOD':30s} {'TRANSFORM':12s} {'OVERLAP':>9s} "
            f"{'MEDIAN':>9s} {'STD':>8s} {'UNIQUE':>9s} {'JACCARD':>9s} "
            f"{'UNION':>11s} {'ALL-INT':>8s}"
        ),
        "-" * 125,
    ]
    for row in summaries:
        overlap = row["overlap"]
        lines.append(
            f"{str(row['method']):30s} {str(row['transform']):12s} "
            f"{float(overlap['mean']):9.2%} {float(overlap['median']):9.2%} "
            f"{float(overlap['std']):8.2%} {float(row['unique_mean']):9.2%} "
            f"{float(row['jaccard_mean']):9.2%} "
            f"{int(row['union_count']):5d}/{DATASET_SIZE:<5d} "
            f"{int(row['all_query_intersection_count']):8d}"
        )

    lines.extend(
        [
            "",
            "RAW-GRADIENT VS FULL-ADAMW",
            "DELTA-OV < 0 means AdamW is more query-specific.",
            (
                f"{'METHOD':30s} {'RAW-OV':>9s} {'ADAMW-OV':>9s} "
                f"{'DELTA-OV':>9s} {'RAW-UNIQ':>10s} {'ADAMW-UNIQ':>11s} "
                f"{'SAME-Q OV':>10s} {'SAME-Q JAC':>11s}"
            ),
            "-" * 125,
        ]
    )
    for row in paired:
        lines.append(
            f"{str(row['method']):30s} "
            f"{float(row['raw_cross_query_overlap']):9.2%} "
            f"{float(row['adamw_cross_query_overlap']):9.2%} "
            f"{float(row['adamw_minus_raw_overlap']):+9.2%} "
            f"{float(row['raw_unique']):10.2%} "
            f"{float(row['adamw_unique']):11.2%} "
            f"{float(row['same_query_raw_adamw_overlap']['mean']):10.2%} "
            f"{float(row['same_query_raw_adamw_jaccard_mean']):11.2%}"
        )

    report = "\n".join(lines) + "\n"
    print("\n" + report, end="")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    json_path = args.output.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {"summary": summaries, "raw_gradient_vs_adamw": paired},
            indent=2,
        )
        + "\n"
    )
    print(f"[saved] {args.output}")
    print(f"[saved] {json_path}")


if __name__ == "__main__":
    main()
