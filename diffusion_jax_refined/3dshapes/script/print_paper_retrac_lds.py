#!/usr/bin/env python3
"""Print Paper ReTrac LDS results without expensive recursive directory scans."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from dataset_config import _prompt_tag


TARGETS = (
    "endpoint_contarfactual",
    "traj_contarfactual",
    "simple_loss",
    "noise_trajectory",
)
STANDARD_LDS_GROUP = (
    "m_64_k_2500_subset_seed_0__"
    "m_64_k_2500_subset_seed_1__"
    "m_64_k_2500_subset_seed_2"
)


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument(
        "--query-file",
        type=Path,
        default=SHAPES_ROOT / "queries_in_distribution_plus_zero_seed_100_219.json",
    )
    parser.add_argument("--query-ids", default=",".join(map(str, range(100))))
    parser.add_argument("--query-timestamp-count", type=int, choices=(10, 100), default=10)
    parser.add_argument("--prediction-sign", choices=("p1", "m1"), default="m1")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    records = json.loads(args.query_file.read_text())["queries"]
    query_ids = parse_ints(args.query_ids)
    namespace = (
        "traj_tracin_paper_retrac_adamw_full_exact4_"
        f"endpoint{args.query_timestamp_count}x1_q0_99_query_train_l2"
    )
    eval_root = SHAPES_ROOT / "result" / args.experiment / "eval" / "prompted_solo"

    def read_one(task: tuple[int, int]) -> tuple[int, int, float | None, Path]:
        query_id, target_index = task
        record = records[query_id]
        seed = int(record.get("initial_seed", record.get("seed")))
        path = (
            eval_root
            / f"query_{_prompt_tag(str(record['prompt']))}"
            / f"initial_seed_{seed}"
            / "lds"
            / namespace
            / TARGETS[target_index]
            / f"pred_kept_sign_{args.prediction_sign}"
            / STANDARD_LDS_GROUP
            / "lds_summary.json"
        )
        if not path.is_file():
            return query_id, target_index, None, path
        return query_id, target_index, float(json.loads(path.read_text())["lds_percent"]), path

    tasks = [(query_id, index) for query_id in query_ids for index in range(len(TARGETS))]
    values = {query_id: [None] * len(TARGETS) for query_id in query_ids}
    missing: list[tuple[int, int, Path]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for query_id, target_index, value, path in pool.map(read_one, tasks):
            values[query_id][target_index] = value
            if value is None:
                missing.append((query_id, target_index, path))

    print(
        "PAPER RETRAC ADAMW FULL — EXACT4 / "
        f"ENDPOINT{args.query_timestamp_count}x1 — QUERY_TRAIN_L2 — "
        f"{args.prediction_sign.upper()}"
    )
    print(
        f"{'QUERY':7s}{'ENDPOINT':>12s}{'TRAJ-CF':>12s}"
        f"{'SIMPLE':>12s}{'NOISE':>12s}"
    )
    print("-" * 55)
    for query_id in query_ids:
        cells = [
            f"{value:+11.3f}%" if value is not None else f"{'MISSING':>12s}"
            for value in values[query_id]
        ]
        print(f"Q{query_id:<6d}" + "".join(cells))

    print("-" * 55)
    means: list[str] = []
    counts: list[int] = []
    for target_index in range(len(TARGETS)):
        column = [
            values[query_id][target_index]
            for query_id in query_ids
            if values[query_id][target_index] is not None
        ]
        counts.append(len(column))
        means.append(
            f"{statistics.fmean(column):+11.3f}%" if column else f"{'MISSING':>12s}"
        )
    print(f"{'MEAN':7s}" + "".join(means))
    print(
        "COMPLETE: "
        + " ".join(
            f"{label}={count}/{len(query_ids)}"
            for label, count in zip(("endpoint", "traj", "simple", "noise"), counts)
        )
        + f" total={sum(counts)}/{len(query_ids) * len(TARGETS)}"
    )
    if missing:
        print("\nFirst 20 missing results:")
        for query_id, target_index, path in missing[:20]:
            print(f"Q{query_id:02d} {TARGETS[target_index]}: {path}")


if __name__ == "__main__":
    main()
