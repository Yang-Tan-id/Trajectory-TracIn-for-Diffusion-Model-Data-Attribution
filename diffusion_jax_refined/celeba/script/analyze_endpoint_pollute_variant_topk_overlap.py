#!/usr/bin/env python3
"""Compare CelebA endpoint-pollute top-k sets across normalization variants."""

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


NAMESPACE = (
    "celeba_endpoint_pollute_adamw_full_train1x50_query1x50_"
    "timestamp_sum_squared_shape3d_semantics"
)
VARIANTS = (
    ("raw", "score"),
    ("query_l2", "score_query_normalized"),
    ("train_l2", "score_train_l2_normalized"),
    ("query_train_l2", "score_query_train_l2_normalized"),
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
    return int(record["initial_seed"] if "initial_seed" in record else record["seed"])


def score_root(result_root: Path, train_seed: int, record: dict) -> Path:
    return (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
        / f"query_{_prompt_path_tag(str(record['prompt']))}"
        / f"initial_seed_{query_seed(record)}"
        / f"traj_tracin_{NAMESPACE}"
    )


def topk(score_dir: Path, k: int) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(np.load(score_dir / "scores.npy"), dtype=np.float64).reshape(-1)
    indices = np.asarray(
        np.load(score_dir / "score_indices.npy"), dtype=np.int64
    ).reshape(-1)
    if scores.shape != indices.shape:
        raise ValueError(f"score/index shape mismatch under {score_dir}")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"duplicate candidate indices under {score_dir}")
    ranking = np.where(np.isfinite(scores), scores, -np.inf)
    order = np.argsort(-ranking, kind="stable")[:k]
    if len(order) != k or np.any(~np.isfinite(ranking[order])):
        raise ValueError(f"fewer than {k} finite scores under {score_dir}")
    return indices, indices[order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    query_ids = parse_ids(args.query_ids)
    records = load_queries(args.query_file)
    result_root = CELEBA_ROOT / "result" / args.experiment
    output_dir = args.output_dir or (
        result_root / "analysis" / f"endpoint_pollute_variant_top{args.topk}_overlap"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_names = [name for name, _ in VARIANTS]
    rows: list[dict[str, object]] = []
    universe_size: int | None = None
    for query_id in query_ids:
        record = records[query_id]
        root = score_root(result_root, args.train_seed, record)
        universes: dict[str, np.ndarray] = {}
        selections: dict[str, set[int]] = {}
        for name, directory in VARIANTS:
            universe, selected = topk(root / directory, args.topk)
            universes[name] = universe
            selections[name] = set(map(int, selected))
            universe_size = len(universe)
        reference = set(map(int, universes[variant_names[0]]))
        for name in variant_names[1:]:
            if set(map(int, universes[name])) != reference:
                raise ValueError(f"Q{query_id}: candidate universe differs for {name}")
        for left, right in combinations(variant_names, 2):
            count = len(selections[left] & selections[right])
            rows.append(
                {
                    "query_id": query_id,
                    "left": left,
                    "right": right,
                    "topk": args.topk,
                    "overlap_count": count,
                    "overlap_fraction": count / args.topk,
                }
            )

    lines = [
        "CELEBA ENDPOINT-POLLUTE ADAMW 50x1 TIMESTAMP-SUM-SQUARED",
        f"SAME-QUERY CROSS-VARIANT TOP-{args.topk} OVERLAP",
        "ranking: stored score descending",
        f"queries: {query_ids}",
        "",
        "PAIR                              MEAN    MEDIAN       STD       MIN       MAX",
        "-" * 82,
    ]
    for left, right in combinations(variant_names, 2):
        values = [
            float(row["overlap_fraction"])
            for row in rows
            if row["left"] == left and row["right"] == right
        ]
        lines.append(
            f"{left + ' vs ' + right:31s} "
            f"{statistics.fmean(values):8.2%} "
            f"{statistics.median(values):9.2%} "
            f"{statistics.pstdev(values):9.2%} "
            f"{min(values):9.2%} "
            f"{max(values):9.2%}"
        )
    random_overlap = args.topk / int(universe_size or 1)
    lines.extend(
        [
            "",
            f"candidate universe: N={universe_size}, identical across all variants/queries",
            f"random independent top-k overlap expectation: {random_overlap:.2%}",
            "",
            "PER QUERY",
            "-" * 82,
        ]
    )
    for query_id in query_ids:
        values = [row for row in rows if row["query_id"] == query_id]
        lines.append(
            f"Q{query_id}: "
            + "; ".join(
                f"{row['left']}/{row['right']}={float(row['overlap_fraction']):.2%}"
                for row in values
            )
        )

    report = "\n".join(lines) + "\n"
    print(report, end="")
    txt_path = output_dir / f"top{args.topk}_overlap_report.txt"
    csv_path = output_dir / f"top{args.topk}_overlap_per_query.csv"
    txt_path.write_text(report)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {txt_path}")
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
