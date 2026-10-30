#!/usr/bin/env python3
"""Measure cross-query Top-K uniqueness for CelebA full-10k methods."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


CELEBA_ROOT = Path(__file__).resolve().parents[1]
if str(CELEBA_ROOT) not in sys.path:
    sys.path.insert(0, str(CELEBA_ROOT))

from dataset_config_attr40 import _prompt_path_tag


POLLUTE_NAMESPACE = (
    "celeba_endpoint_pollute_adamw_full_train1x50_query1x50_"
    "timestamp_sum_squared_shape3d_semantics_full10000"
)
POLLUTE_RAW_GRADIENT_NAMESPACE = (
    "celeba_endpoint_pollute_raw_gradient_train1x50_query1x50_"
    "timestamp_sum_squared_shape3d_semantics_full10000"
)
TRACIN_NAMESPACE = (
    "celeba_endpoint_tracin_adamw_full_train_mean50_query_mean50_"
    "linear_shape3d_semantics_full10000"
)
TRACIN_RAW_GRADIENT_NAMESPACE = (
    "celeba_endpoint_tracin_raw_gradient_train_mean50_query_mean50_"
    "linear_shape3d_semantics_full10000"
)
RETRAC_NAMESPACE = (
    "celeba_paper_retrac_adamw_full_train_replay4_batch1_"
    "query_mean50_of_full_l2_linear_shape3d_semantics_full10000"
)
RETRAC_RAW_GRADIENT_NAMESPACE = (
    "celeba_paper_retrac_raw_gradient_train_replay4_batch1_"
    "query_mean50_of_full_l2_linear_shape3d_semantics_full10000"
)

# ranking_sign converts every stored score into a common larger-is-better
# orientation before selecting Top-K.
METHODS = (
    ("PINT RAW-GRADIENT / RAW", POLLUTE_RAW_GRADIENT_NAMESPACE, "score", +1),
    (
        "PINT RAW-GRADIENT / BOTH-L2",
        POLLUTE_RAW_GRADIENT_NAMESPACE,
        "score_query_train_l2_normalized",
        +1,
    ),
    ("PINT ADAMW / RAW", POLLUTE_NAMESPACE, "score", +1),
    (
        "PINT ADAMW / BOTH-L2",
        POLLUTE_NAMESPACE,
        "score_query_train_l2_normalized",
        +1,
    ),
    (
        "ENDPOINT TRACIN RAW-GRADIENT / RAW",
        TRACIN_RAW_GRADIENT_NAMESPACE,
        "score",
        +1,
    ),
    (
        "ENDPOINT TRACIN RAW-GRADIENT / BOTH-L2",
        TRACIN_RAW_GRADIENT_NAMESPACE,
        "score_query_train_l2_normalized",
        +1,
    ),
    ("ENDPOINT TRACIN ADAMW / RAW", TRACIN_NAMESPACE, "score", -1),
    (
        "ENDPOINT TRACIN ADAMW / BOTH-L2",
        TRACIN_NAMESPACE,
        "score_query_train_l2_normalized",
        -1,
    ),
    (
        "RETRAC RAW-GRADIENT / BOTH-L2",
        RETRAC_RAW_GRADIENT_NAMESPACE,
        "score_query_train_l2_normalized",
        +1,
    ),
    (
        "RETRAC ADAMW / BOTH-L2",
        RETRAC_NAMESPACE,
        "score_query_train_l2_normalized",
        -1,
    ),
)

METHOD_PAIRS = (
    ("PINT / RAW", "PINT RAW-GRADIENT / RAW", "PINT ADAMW / RAW"),
    (
        "PINT / BOTH-L2",
        "PINT RAW-GRADIENT / BOTH-L2",
        "PINT ADAMW / BOTH-L2",
    ),
    (
        "ENDPOINT TRACIN / RAW",
        "ENDPOINT TRACIN RAW-GRADIENT / RAW",
        "ENDPOINT TRACIN ADAMW / RAW",
    ),
    (
        "ENDPOINT TRACIN / BOTH-L2",
        "ENDPOINT TRACIN RAW-GRADIENT / BOTH-L2",
        "ENDPOINT TRACIN ADAMW / BOTH-L2",
    ),
    (
        "RETRAC / BOTH-L2",
        "RETRAC RAW-GRADIENT / BOTH-L2",
        "RETRAC ADAMW / BOTH-L2",
    ),
)


def parse_ids(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def load_queries(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text())
    records = payload["queries"] if isinstance(payload, dict) else payload
    return {
        int(record.get("query_id", record.get("id", position))): record
        for position, record in enumerate(records)
    }


def query_seed(record: dict) -> int:
    return int(record.get("initial_seed", record.get("seed")))


def query_score_root(
    result_root: Path, train_seed: int, record: dict
) -> Path:
    return (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
        / f"query_{_prompt_path_tag(str(record['prompt']))}"
        / f"initial_seed_{query_seed(record)}"
    )


def select_topk(
    directory: Path, *, topk: int, ranking_sign: int
) -> tuple[set[int], set[int]]:
    scores = np.asarray(np.load(directory / "scores.npy"), dtype=np.float64).reshape(-1)
    indices = np.asarray(
        np.load(directory / "score_indices.npy"), dtype=np.int64
    ).reshape(-1)
    expected = np.arange(10_000, dtype=np.int64)
    if scores.shape != expected.shape or not np.array_equal(indices, expected):
        raise ValueError(f"not an ordered full-10k score bank: {directory}")
    oriented = ranking_sign * scores
    finite = np.isfinite(oriented)
    if int(finite.sum()) < topk:
        raise ValueError(f"fewer than {topk} finite scores: {directory}")
    ranking = np.where(finite, oriented, -np.inf)
    order = np.argsort(-ranking, kind="stable")[:topk]
    return set(map(int, indices)), set(map(int, indices[order]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe(values: list[float]) -> tuple[float, float, float, float, float]:
    return (
        statistics.fmean(values),
        statistics.median(values),
        statistics.pstdev(values),
        min(values),
        max(values),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-file",
        type=Path,
        default=CELEBA_ROOT / "queries_attr40_val25_x4.json",
    )
    parser.add_argument("--query-ids", default=",".join(map(str, range(60))))
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    query_ids = parse_ids(args.query_ids)
    if len(query_ids) < 2 or len(query_ids) != len(set(query_ids)):
        parser.error("--query-ids must contain at least two unique query IDs")
    if not 0 < args.topk <= 10_000:
        parser.error("--topk must be in [1, 10000]")

    records = load_queries(args.query_file)
    missing = sorted(set(query_ids) - set(records))
    if missing:
        parser.error(f"query IDs missing from query file: {missing}")

    result_root = CELEBA_ROOT / "result" / args.experiment
    output_dir = args.output_dir or (
        result_root
        / "analysis"
        / f"raw_gradient_adamw_cross_query_top{args.topk}_overlap_q{min(query_ids)}_q{max(query_ids)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    selections: dict[str, dict[int, set[int]]] = {
        label: {} for label, *_ in METHODS
    }
    common_universe: set[int] | None = None
    for query_id in query_ids:
        root = query_score_root(result_root, args.train_seed, records[query_id])
        for label, namespace, score_directory, ranking_sign in METHODS:
            universe, selected = select_topk(
                root / f"traj_tracin_{namespace}" / score_directory,
                topk=args.topk,
                ranking_sign=ranking_sign,
            )
            if common_universe is None:
                common_universe = universe
            elif universe != common_universe:
                raise ValueError(f"Q{query_id} {label}: candidate universe differs")
            selections[label][query_id] = selected
        print(f"[loaded] Q{query_id} {len(METHODS)}/{len(METHODS)} methods", flush=True)

    universe_size = len(common_universe or ())
    random_overlap = args.topk / universe_size
    pair_rows: list[dict[str, object]] = []
    per_query_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for label, *_ in METHODS:
        method_sets = selections[label]
        for left_query, right_query in combinations(query_ids, 2):
            overlap_count = len(method_sets[left_query] & method_sets[right_query])
            overlap_fraction = overlap_count / args.topk
            pair_rows.append(
                {
                    "method": label,
                    "left_query": left_query,
                    "right_query": right_query,
                    "overlap_count": overlap_count,
                    "overlap_fraction": overlap_fraction,
                    "unique_fraction": 1.0 - overlap_fraction,
                    "jaccard": overlap_count / (2 * args.topk - overlap_count),
                }
            )

        method_pair_rows = [row for row in pair_rows if row["method"] == label]
        overlap_values = [float(row["overlap_fraction"]) for row in method_pair_rows]
        jaccard_values = [float(row["jaccard"]) for row in method_pair_rows]
        overlap_stats = describe(overlap_values)
        jaccard_stats = describe(jaccard_values)
        union = set().union(*(method_sets[query_id] for query_id in query_ids))
        all_intersection = set.intersection(
            *(method_sets[query_id] for query_id in query_ids)
        )

        exclusive_counts: list[int] = []
        for query_id in query_ids:
            others = set().union(
                *(method_sets[other] for other in query_ids if other != query_id)
            )
            exclusive = len(method_sets[query_id] - others)
            exclusive_counts.append(exclusive)
            query_overlaps = [
                float(row["overlap_fraction"])
                for row in method_pair_rows
                if query_id in (row["left_query"], row["right_query"])
            ]
            per_query_rows.append(
                {
                    "method": label,
                    "query_id": query_id,
                    "mean_overlap_with_other_queries": statistics.fmean(query_overlaps),
                    "mean_unique_fraction": 1.0 - statistics.fmean(query_overlaps),
                    "exclusive_to_this_query_count": exclusive,
                }
            )

        summary_rows.append(
            {
                "method": label,
                "queries": len(query_ids),
                "query_pairs": len(method_pair_rows),
                "topk": args.topk,
                "candidate_count": universe_size,
                "random_overlap_expectation": random_overlap,
                "overlap_mean": overlap_stats[0],
                "overlap_median": overlap_stats[1],
                "overlap_std": overlap_stats[2],
                "overlap_min": overlap_stats[3],
                "overlap_max": overlap_stats[4],
                "unique_fraction_mean": 1.0 - overlap_stats[0],
                "jaccard_mean": jaccard_stats[0],
                "union_count": len(union),
                "union_fraction_of_dataset": len(union) / universe_size,
                "intersection_all_queries_count": len(all_intersection),
                "mean_exclusive_count_per_query": statistics.fmean(exclusive_counts),
            }
        )

    pair_rows_by_method = {
        label: [row for row in pair_rows if row["method"] == label]
        for label, *_ in METHODS
    }
    comparison_rows: list[dict[str, object]] = []
    for method, raw_label, adamw_label in METHOD_PAIRS:
        raw_rows = pair_rows_by_method[raw_label]
        adamw_rows = pair_rows_by_method[adamw_label]
        raw_pairs = [
            (int(row["left_query"]), int(row["right_query"]))
            for row in raw_rows
        ]
        adamw_pairs = [
            (int(row["left_query"]), int(row["right_query"]))
            for row in adamw_rows
        ]
        if raw_pairs != adamw_pairs:
            raise RuntimeError(f"query-pair ordering differs for {method}")
        raw_values = [float(row["overlap_fraction"]) for row in raw_rows]
        adamw_values = [float(row["overlap_fraction"]) for row in adamw_rows]
        deltas = [adamw - raw for raw, adamw in zip(raw_values, adamw_values)]
        comparison_rows.append(
            {
                "method": method,
                "query_pairs": len(deltas),
                "raw_gradient_overlap_mean": statistics.fmean(raw_values),
                "adamw_overlap_mean": statistics.fmean(adamw_values),
                "adamw_minus_raw_overlap": statistics.fmean(deltas),
                "raw_gradient_unique_mean": 1.0 - statistics.fmean(raw_values),
                "adamw_unique_mean": 1.0 - statistics.fmean(adamw_values),
                "adamw_minus_raw_unique": -statistics.fmean(deltas),
                "adamw_lower_overlap_pairs": sum(delta < 0.0 for delta in deltas),
                "adamw_higher_overlap_pairs": sum(delta > 0.0 for delta in deltas),
                "tied_pairs": sum(delta == 0.0 for delta in deltas),
            }
        )

    lines = [
        "CELEBA FULL-10K RAW-GRADIENT + ADAMW CROSS-QUERY TOP-K UNIQUENESS",
        f"queries={len(query_ids)} ({min(query_ids)}..{max(query_ids)}) topk={args.topk} candidates={universe_size}",
        f"query pairs per method={len(query_ids) * (len(query_ids) - 1) // 2}",
        f"independent-random overlap expectation={random_overlap:.2%}",
        "lower overlap means more query-specific; unique=1-overlap",
        "",
        (
            f"{'METHOD':42s} {'OVERLAP':>9s} {'MEDIAN':>9s} {'STD':>8s} "
            f"{'UNIQUE':>9s} {'JACCARD':>9s} {'UNION':>9s} {'ALL-INT':>8s}"
        ),
        "-" * 117,
    ]
    for row in summary_rows:
        lines.append(
            f"{str(row['method']):42s} "
            f"{float(row['overlap_mean']):9.2%} "
            f"{float(row['overlap_median']):9.2%} "
            f"{float(row['overlap_std']):8.2%} "
            f"{float(row['unique_fraction_mean']):9.2%} "
            f"{float(row['jaccard_mean']):9.2%} "
            f"{int(row['union_count']):5d}/{universe_size:<3d} "
            f"{int(row['intersection_all_queries_count']):8d}"
        )
    lines.extend(
        [
            "",
            "UNION is the number of distinct training examples selected by at least one query.",
            "ALL-INT is the number selected by every query.",
            "",
            "RAW-GRADIENT VS ADAMW",
            "Negative DELTA-OV means AdamW is more query-specific; positive DELTA-UNIQ means the same.",
            "",
            (
                f"{'METHOD':30s} {'RAW-OV':>9s} {'ADAMW-OV':>9s} "
                f"{'DELTA-OV':>9s} {'RAW-UNIQ':>10s} {'ADAMW-UNIQ':>11s} "
                f"{'DELTA-UNIQ':>11s} {'ADAMW LOWER':>13s}"
            ),
            "-" * 112,
        ]
    )
    for row in comparison_rows:
        lines.append(
            f"{str(row['method']):30s} "
            f"{float(row['raw_gradient_overlap_mean']):9.2%} "
            f"{float(row['adamw_overlap_mean']):9.2%} "
            f"{float(row['adamw_minus_raw_overlap']):+9.2%} "
            f"{float(row['raw_gradient_unique_mean']):10.2%} "
            f"{float(row['adamw_unique_mean']):11.2%} "
            f"{float(row['adamw_minus_raw_unique']):+11.2%} "
            f"{int(row['adamw_lower_overlap_pairs']):5d}/{int(row['query_pairs']):<7d}"
        )
    report = "\n".join(lines) + "\n"
    print("\n" + report, end="")

    write_csv(output_dir / "pairwise_cross_query_overlap.csv", pair_rows)
    write_csv(output_dir / "per_query_uniqueness.csv", per_query_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "raw_gradient_vs_adamw.csv", comparison_rows)
    report_path = output_dir / "report.txt"
    report_path.write_text(report)
    print(f"[saved] {report_path}")
    print(f"[saved] {output_dir / 'summary.csv'}")
    print(f"[saved] {output_dir / 'raw_gradient_vs_adamw.csv'}")
    print(f"[saved] {output_dir / 'pairwise_cross_query_overlap.csv'}")
    print(f"[saved] {output_dir / 'per_query_uniqueness.csv'}")


if __name__ == "__main__":
    main()
