#!/usr/bin/env python3
"""Print mean LDS for each endpoint-pollute 10x1 timestamp-square score."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from dataset_config import _prompt_tag  # noqa: E402


TIMESTAMPS = (0, 111, 222, 333, 444, 555, 666, 777, 888, 999)
VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")
TARGETS = (
    "endpoint_contarfactual",
    "traj_contarfactual",
    "simple_loss",
    "noise_trajectory",
)
GROUP = (
    "m_64_k_2500_subset_seed_0__"
    "m_64_k_2500_subset_seed_1__"
    "m_64_k_2500_subset_seed_2"
)


def parse_ids(text: str) -> list[int]:
    return [int(token) for token in text.replace(",", " ").split()]


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def recover_from_csv(summary_path: Path) -> float:
    csv_path = summary_path.with_name("lds_results.csv")
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    prediction = np.asarray([float(row["pred_sum_tau"]) for row in rows])
    truth = np.asarray([float(row["true_f"]) for row in rows])
    if len(prediction) < 2:
        raise ValueError(f"not enough rows in {csv_path}")
    corr = float(
        np.corrcoef(average_ranks(prediction), average_ranks(truth))[0, 1]
    )
    return 100.0 * corr


def read_lds(path: Path, retries: int = 5) -> tuple[float, bool]:
    error: Exception | None = None
    for attempt in range(retries):
        try:
            return float(json.loads(path.read_text())["lds_percent"]), False
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            error = exc
            time.sleep(0.05 * (attempt + 1))
    try:
        return recover_from_csv(path), True
    except Exception as csv_error:
        raise RuntimeError(
            f"could not read {path} ({error}); CSV recovery failed: {csv_error}"
        ) from csv_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument(
        "--query-file",
        type=Path,
        default=SHAPES_ROOT / "queries_in_distribution_plus_zero_seed_100_219.json",
    )
    parser.add_argument("--query-ids", default=",".join(map(str, range(100))))
    parser.add_argument("--prediction-sign", choices=("p1", "m1"), default="p1")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--per-query", action="store_true")
    args = parser.parse_args()

    records = json.loads(args.query_file.read_text())["queries"]
    query_ids = parse_ids(args.query_ids)
    eval_root = SHAPES_ROOT / "result" / args.experiment / "eval" / "prompted_solo"
    tasks = []
    for timestep in TIMESTAMPS:
        base_namespace = (
            "traj_tracin_recreate_adamw_full_polluted_endpoint_"
            "delta_l2normalized_timestamp_aware_square_"
            f"t{timestep:03d}_q0_99"
        )
        for variant in VARIANTS:
            namespace = f"{base_namespace}_{variant}"
            for target in TARGETS:
                for query_id in query_ids:
                    record = records[query_id]
                    seed = int(record.get("initial_seed", record.get("seed")))
                    path = (
                        eval_root
                        / f"query_{_prompt_tag(record['prompt'])}"
                        / f"initial_seed_{seed}"
                        / "lds"
                        / namespace
                        / target
                        / f"pred_kept_sign_{args.prediction_sign}"
                        / GROUP
                        / "lds_summary.json"
                    )
                    tasks.append((timestep, variant, target, query_id, path))

    def load(task):
        timestep, variant, target, query_id, path = task
        value, recovered = read_lds(path)
        return timestep, variant, target, query_id, value, recovered

    values = {}
    recovered_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for timestep, variant, target, query_id, value, recovered in executor.map(
            load, tasks
        ):
            values[timestep, variant, target, query_id] = value
            recovered_count += int(recovered)

    for variant in VARIANTS:
        print(
            "ADAMW FULL ENDPOINT-POLLUTE 10x1 — PER-TIMESTAMP "
            f"TIMESTAMP-AWARE SQUARE — {variant.upper()} — "
            f"{args.prediction_sign.upper()}"
        )
        print(
            f"{'TIMESTAMP':>9s} {'ENDPOINT':>12s} {'TRAJ-CF':>12s} "
            f"{'SIMPLE':>12s} {'NOISE':>12s}"
        )
        print("-" * 63)
        for timestep in TIMESTAMPS:
            means = [
                fmean(values[timestep, variant, target, qid] for qid in query_ids)
                for target in TARGETS
            ]
            print(f"{timestep:9d}" + "".join(f" {value:+11.3f}%" for value in means))
        print()

        if args.per_query:
            for timestep in TIMESTAMPS:
                print(f"{variant.upper()} t={timestep}")
                print(
                    f"{'QUERY':7s} {'ENDPOINT':>12s} {'TRAJ-CF':>12s} "
                    f"{'SIMPLE':>12s} {'NOISE':>12s}"
                )
                for query_id in query_ids:
                    row = [
                        values[timestep, variant, target, query_id]
                        for target in TARGETS
                    ]
                    print(
                        f"Q{query_id:<6d}"
                        + "".join(f" {value:+11.3f}%" for value in row)
                    )
                print()

    print(
        f"COMPLETE: {len(tasks)}/{len(tasks)} LDS summaries; "
        f"CSV_RECOVERED={recovered_count}"
    )


if __name__ == "__main__":
    main()
