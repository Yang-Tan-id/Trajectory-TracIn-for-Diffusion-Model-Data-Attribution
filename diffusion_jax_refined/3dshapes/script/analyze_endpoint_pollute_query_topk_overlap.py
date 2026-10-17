#!/usr/bin/env python3
"""Measure top-k datapoint overlap across Shape3D endpoint-pollute queries.

The primary analysis compares every pair of queries separately for the four
normalization variants.  Ranking follows the endpoint-pollute P1 convention:
larger stored scores rank first.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from dataset_config import _prompt_tag


VARIANTS = (
    ("raw", "score"),
    ("query_l2", "score_query_normalized"),
    ("train_l2", "score_train_l2_normalized"),
    ("query_train_l2", "score_query_train_l2_normalized"),
)
DEFAULT_NAMESPACES = (
    "recreate_adamw_full_polluted_endpoint_delta_l2normalized_"
    "timestamp_sum_squared_aligned100x1_q0_99",
    "recreate_adamw_full_polluted_endpoint_delta_l2normalized_"
    "timestamp_sum_squared_q0_99",
)


def parse_ids(text: str) -> list[int]:
    values = [int(token) for token in text.replace(",", " ").split()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("query ids must be nonempty and unique")
    return values


def score_root(
    result_root: Path,
    train_seed: int,
    record: dict[str, object],
    namespace: str,
) -> Path:
    seed = int(record.get("initial_seed", record.get("seed")))
    return (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
        / f"query_{_prompt_tag(str(record['prompt']))}"
        / f"initial_seed_{seed}"
        / f"traj_tracin_{namespace}"
    )


def namespace_is_complete(
    result_root: Path,
    train_seed: int,
    records: list[dict[str, object]],
    query_ids: list[int],
    namespace: str,
) -> bool:
    for query_id in query_ids:
        root = score_root(result_root, train_seed, records[query_id], namespace)
        for _, directory in VARIANTS:
            if not (root / directory / "scores.npy").is_file():
                return False
            if not (root / directory / "score_indices.npy").is_file():
                return False
    return True


def choose_namespace(
    requested: str,
    result_root: Path,
    train_seed: int,
    records: list[dict[str, object]],
    query_ids: list[int],
) -> str:
    candidates = DEFAULT_NAMESPACES if requested == "auto" else (requested,)
    for namespace in candidates:
        if namespace_is_complete(
            result_root, train_seed, records, query_ids, namespace
        ):
            return namespace
    preview = "\n".join(f"  - {value}" for value in candidates)
    raise FileNotFoundError(
        "no requested namespace is complete for all queries and variants:\n"
        f"{preview}"
    )


def load_topk(path: Path, topk: int) -> tuple[frozenset[int], np.ndarray]:
    scores = np.asarray(np.load(path / "scores.npy"), dtype=np.float64).reshape(-1)
    indices = np.asarray(
        np.load(path / "score_indices.npy"), dtype=np.int64
    ).reshape(-1)
    if scores.shape != indices.shape:
        raise ValueError(f"score/index shape mismatch under {path}")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"duplicate score indices under {path}")
    finite = np.isfinite(scores)
    if int(finite.sum()) < topk:
        raise ValueError(
            f"only {int(finite.sum())} finite scores under {path}; topk={topk}"
        )
    ranking = np.where(finite, scores, -np.inf)
    order = np.argsort(-ranking, kind="stable")[:topk]
    return frozenset(int(value) for value in indices[order]), indices


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-file",
        type=Path,
        default=SHAPES_ROOT / "queries_in_distribution_plus_zero_seed_100_219.json",
    )
    parser.add_argument(
        "--query-ids",
        type=parse_ids,
        default=list(range(100)),
        help="Comma/space-separated query ids (default: 0-99).",
    )
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument(
        "--namespace",
        default="auto",
        help="Score namespace without traj_tracin_ prefix; auto prefers aligned100x1.",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.topk <= 0:
        parser.error("--topk must be positive")
    records = json.loads(args.query_file.read_text())["queries"]
    if min(args.query_ids) < 0 or max(args.query_ids) >= len(records):
        parser.error(f"query ids must be within [0, {len(records) - 1}]")
    if len(args.query_ids) < 2:
        parser.error("at least two query ids are required")

    result_root = SHAPES_ROOT / "result" / args.experiment
    namespace = choose_namespace(
        args.namespace,
        result_root,
        args.train_seed,
        records,
        args.query_ids,
    )
    output_dir = args.output_dir or (
        result_root
        / "analysis"
        / f"endpoint_pollute_query_top{args.topk}_overlap"
        / namespace
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    top_sets: dict[str, dict[int, frozenset[int]]] = {}
    candidate_indices: dict[int, np.ndarray] = {}
    print(f"[namespace] traj_tracin_{namespace}", flush=True)
    print("[ranking] stored score descending (endpoint-pollute P1)", flush=True)
    for variant, directory in VARIANTS:
        top_sets[variant] = {}
        for position, query_id in enumerate(args.query_ids, start=1):
            root = score_root(
                result_root, args.train_seed, records[query_id], namespace
            )
            selected, indices = load_topk(root / directory, args.topk)
            top_sets[variant][query_id] = selected
            if query_id not in candidate_indices:
                candidate_indices[query_id] = indices
            elif not np.array_equal(candidate_indices[query_id], indices):
                raise ValueError(
                    f"candidate indices differ between variants for Q{query_id}"
                )
            if position % 10 == 0 or position == len(args.query_ids):
                print(
                    f"[load] {variant}: {position}/{len(args.query_ids)} queries",
                    flush=True,
                )

    query_pairs = list(combinations(args.query_ids, 2))
    pair_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    report_lines = [
        "SHAPE3D ADAMW-FULL ENDPOINT-POLLUTE — CROSS-QUERY TOP-K OVERLAP",
        f"namespace: traj_tracin_{namespace}",
        "ranking: stored score descending (P1)",
        f"queries: {len(args.query_ids)}; pairs: {len(query_pairs)}; topk: {args.topk}",
        "overlap = |topK(Qa) ∩ topK(Qb)| / K",
        "",
    ]
    for variant, _ in VARIANTS:
        rows_for_variant = []
        for left, right in query_pairs:
            intersection = len(top_sets[variant][left] & top_sets[variant][right])
            overlap = intersection / args.topk
            jaccard = intersection / (2 * args.topk - intersection)
            row = {
                "variant": variant,
                "query_left": left,
                "query_right": right,
                "intersection": intersection,
                "overlap_fraction": overlap,
                "jaccard": jaccard,
            }
            pair_rows.append(row)
            rows_for_variant.append(row)
        values = np.asarray(
            [row["overlap_fraction"] for row in rows_for_variant],
            dtype=np.float64,
        )
        stats = summarize(values)
        lowest = min(rows_for_variant, key=lambda row: row["overlap_fraction"])
        highest = max(rows_for_variant, key=lambda row: row["overlap_fraction"])
        summary_rows.append(
            {
                "variant": variant,
                "queries": len(args.query_ids),
                "pairs": len(query_pairs),
                **stats,
                "lowest_pair": f"Q{lowest['query_left']}/Q{lowest['query_right']}",
                "highest_pair": f"Q{highest['query_left']}/Q{highest['query_right']}",
            }
        )
        report_lines.extend(
            [
                variant.upper(),
                (
                    f"  mean={stats['mean']:.2%}  median={stats['median']:.2%}  "
                    f"std={stats['std']:.2%}"
                ),
                (
                    f"  p10-p90=[{stats['p10']:.2%}, {stats['p90']:.2%}]  "
                    f"min-max=[{stats['min']:.2%}, {stats['max']:.2%}]"
                ),
                (
                    f"  lowest=Q{lowest['query_left']}/Q{lowest['query_right']} "
                    f"({float(lowest['overlap_fraction']):.2%}); "
                    f"highest=Q{highest['query_left']}/Q{highest['query_right']} "
                    f"({float(highest['overlap_fraction']):.2%})"
                ),
                "",
            ]
        )

    matched_variant_rows: list[dict[str, object]] = []
    report_lines.append("SAME-QUERY CROSS-VARIANT TOP-K OVERLAP")
    for (left_variant, _), (right_variant, _) in combinations(VARIANTS, 2):
        values = np.asarray(
            [
                len(
                    top_sets[left_variant][query_id]
                    & top_sets[right_variant][query_id]
                )
                / args.topk
                for query_id in args.query_ids
            ],
            dtype=np.float64,
        )
        stats = summarize(values)
        matched_variant_rows.append(
            {
                "variant_left": left_variant,
                "variant_right": right_variant,
                "queries": len(args.query_ids),
                **stats,
            }
        )
        report_lines.append(
            f"  {left_variant:14s} vs {right_variant:14s}: "
            f"mean={stats['mean']:.2%} median={stats['median']:.2%} "
            f"min-max=[{stats['min']:.2%}, {stats['max']:.2%}]"
        )

    reference = next(iter(candidate_indices.values()))
    all_candidate_sets_equal = all(
        np.array_equal(reference, values) for values in candidate_indices.values()
    )
    if all_candidate_sets_equal:
        random_overlap = args.topk / len(reference)
        report_lines.extend(
            [
                "",
                f"candidate universe: identical across queries, N={len(reference)}",
                f"random independent top-k overlap expectation: {random_overlap:.2%}",
            ]
        )
    else:
        report_lines.extend(
            [
                "",
                "candidate universe: differs across queries; no single random baseline",
            ]
        )

    write_csv(output_dir / "cross_query_pairwise_overlap.csv", pair_rows)
    write_csv(output_dir / "cross_query_overlap_summary.csv", summary_rows)
    write_csv(
        output_dir / "same_query_cross_variant_overlap_summary.csv",
        matched_variant_rows,
    )
    report = "\n".join(report_lines) + "\n"
    report_path = output_dir / "topk_overlap_report.txt"
    report_path.write_text(report)
    print()
    print(report, end="")
    print(f"[saved] {report_path}")


if __name__ == "__main__":
    main()
