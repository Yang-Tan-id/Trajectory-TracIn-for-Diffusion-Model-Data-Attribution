#!/usr/bin/env python3
"""Measure cross-query top-k overlap for the paper AdamW-full ReTrac score."""

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


NAMESPACE = "paper_retrac_adamw_full_exact4_endpoint100x1_q0_99"
SCORE_DIRECTORY = "score_query_train_l2_normalized"


def parse_ids(text: str) -> list[int]:
    values = [int(token) for token in text.replace(",", " ").split()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("query ids must be nonempty and unique")
    return values


def score_directory(
    result_root: Path,
    train_seed: int,
    record: dict[str, object],
) -> Path:
    seed = int(record.get("initial_seed", record.get("seed")))
    return (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
        / f"query_{_prompt_tag(str(record['prompt']))}"
        / f"initial_seed_{seed}"
        / f"traj_tracin_{NAMESPACE}"
        / SCORE_DIRECTORY
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
    # ReTrac removal uses ranking_score = -stored_score.  Therefore the most
    # influential points are the smallest finite stored scores.
    ranking = np.where(finite, scores, np.inf)
    order = np.argsort(ranking, kind="stable")[:topk]
    return frozenset(int(value) for value in indices[order]), indices


def stats(values: np.ndarray) -> dict[str, float]:
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    output_dir = args.output_dir or (
        result_root
        / "analysis"
        / f"retrac_query_top{args.topk}_overlap"
        / NAMESPACE
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    top_sets: dict[int, frozenset[int]] = {}
    candidate_indices: dict[int, np.ndarray] = {}
    for position, query_id in enumerate(args.query_ids, start=1):
        root = score_directory(
            result_root, args.train_seed, records[query_id]
        )
        if not (root / "scores.npy").is_file():
            raise FileNotFoundError(root / "scores.npy")
        selected, indices = load_topk(root, args.topk)
        top_sets[query_id] = selected
        candidate_indices[query_id] = indices
        if position % 10 == 0 or position == len(args.query_ids):
            print(f"[load] {position}/{len(args.query_ids)} queries", flush=True)

    rows: list[dict[str, object]] = []
    for left, right in combinations(args.query_ids, 2):
        intersection = len(top_sets[left] & top_sets[right])
        rows.append(
            {
                "query_left": left,
                "query_right": right,
                "intersection": intersection,
                "overlap_fraction": intersection / args.topk,
                "jaccard": intersection / (2 * args.topk - intersection),
            }
        )
    values = np.asarray(
        [row["overlap_fraction"] for row in rows], dtype=np.float64
    )
    summary = stats(values)
    lowest = min(rows, key=lambda row: row["overlap_fraction"])
    highest = max(rows, key=lambda row: row["overlap_fraction"])

    reference = next(iter(candidate_indices.values()))
    universes_equal = all(
        np.array_equal(reference, indices)
        for indices in candidate_indices.values()
    )
    report_lines = [
        "SHAPE3D PAPER RETRAC ADAMW-FULL — CROSS-QUERY TOP-K OVERLAP",
        f"namespace: traj_tracin_{NAMESPACE}",
        "variant: query_train_l2 (fixed by standard ReTrac definition)",
        "ranking: -stored_score, equivalently stored score ascending",
        (
            f"queries: {len(args.query_ids)}; pairs: {len(rows)}; "
            f"topk: {args.topk}"
        ),
        "overlap = |topK(Qa) ∩ topK(Qb)| / K",
        "",
        (
            f"mean={summary['mean']:.2%}  median={summary['median']:.2%}  "
            f"std={summary['std']:.2%}"
        ),
        (
            f"p10-p90=[{summary['p10']:.2%}, {summary['p90']:.2%}]  "
            f"min-max=[{summary['min']:.2%}, {summary['max']:.2%}]"
        ),
        (
            f"lowest=Q{lowest['query_left']}/Q{lowest['query_right']} "
            f"({float(lowest['overlap_fraction']):.2%}); "
            f"highest=Q{highest['query_left']}/Q{highest['query_right']} "
            f"({float(highest['overlap_fraction']):.2%})"
        ),
    ]
    if universes_equal:
        report_lines.extend(
            [
                "",
                f"candidate universe: identical across queries, N={len(reference)}",
                (
                    "random independent top-k overlap expectation: "
                    f"{args.topk / len(reference):.2%}"
                ),
            ]
        )
    else:
        report_lines.extend(
            [
                "",
                "candidate universe differs across queries; no single random baseline",
            ]
        )

    write_csv(output_dir / "cross_query_pairwise_overlap.csv", rows)
    write_csv(
        output_dir / "cross_query_overlap_summary.csv",
        [{"variant": "query_train_l2", "queries": len(args.query_ids),
          "pairs": len(rows), **summary,
          "lowest_pair": f"Q{lowest['query_left']}/Q{lowest['query_right']}",
          "highest_pair": f"Q{highest['query_left']}/Q{highest['query_right']}"}],
    )
    report = "\n".join(report_lines) + "\n"
    report_path = output_dir / "topk_overlap_report.txt"
    report_path.write_text(report)
    print()
    print(report, end="")
    print(f"[saved] {report_path}")


if __name__ == "__main__":
    main()
