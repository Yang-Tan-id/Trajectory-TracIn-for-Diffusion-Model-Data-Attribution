#!/usr/bin/env python3
"""Print the completed 3D-Shapes D-TRAK lambda sweep without globbing Lustre."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from script.score_dtrak_three_objectives_100x1 import (  # noqa: E402
    DEFAULT_LAMBDAS,
    lambda_tag,
)


OBJECTIVES = ("simple_loss", "square", "average")
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


def recover_from_csv(path: Path) -> float:
    csv_path = path.with_name("lds_results.csv")
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    prediction = np.asarray([float(row["pred_sum_tau"]) for row in rows])
    truth = np.asarray([float(row["true_f"]) for row in rows])
    if len(prediction) < 2:
        raise ValueError(f"not enough rows in {csv_path}")
    corr = float(np.corrcoef(average_ranks(prediction), average_ranks(truth))[0, 1])
    return 100.0 * corr


def read_lds(path: Path, retries: int = 5) -> tuple[float, bool]:
    error: Exception | None = None
    for attempt in range(retries):
        try:
            payload = json.loads(path.read_text())
            return float(payload["lds_percent"]), False
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            error = exc
            time.sleep(0.05 * (attempt + 1))
    try:
        return recover_from_csv(path), True
    except Exception as csv_error:
        raise RuntimeError(
            f"could not read {path} ({error}); CSV recovery also failed: {csv_error}"
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
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--objectives",
        default=",".join(OBJECTIVES),
        help="Comma/space-separated subset of simple_loss,square,average.",
    )
    parser.add_argument(
        "--lambdas",
        default=",".join(f"{value:g}" for value in DEFAULT_LAMBDAS),
        help="Comma/space-separated subset of completed lambda values.",
    )
    parser.add_argument(
        "--per-query",
        action="store_true",
        help="Also print one four-target row per query (best used with one objective/lambda).",
    )
    args = parser.parse_args()

    records = json.loads(args.query_file.read_text())["queries"]
    query_ids = parse_ids(args.query_ids)
    objectives = tuple(args.objectives.replace(",", " ").split())
    invalid_objectives = sorted(set(objectives) - set(OBJECTIVES))
    if invalid_objectives:
        raise ValueError(f"unknown objectives: {invalid_objectives}")
    lambdas = tuple(float(token) for token in args.lambdas.replace(",", " ").split())
    invalid_lambdas = [value for value in lambdas if value not in DEFAULT_LAMBDAS]
    if invalid_lambdas:
        raise ValueError(f"lambda values were not run: {invalid_lambdas}")
    eval_root = SHAPES_ROOT / "result" / args.experiment / "eval" / "prompted_solo"

    tasks = []
    for objective in objectives:
        for damping in lambdas:
            namespace = (
                f"dtrak_{objective}_train100x1_query100x1_q0_99_"
                f"lambda_{lambda_tag(damping)}_raw"
            )
            for target in TARGETS:
                for query_id in query_ids:
                    query = records[query_id]
                    seed = int(query.get("initial_seed", query.get("seed")))
                    path = (
                        eval_root
                        / f"query_{_prompt_tag(query['prompt'])}"
                        / f"initial_seed_{seed}"
                        / "lds"
                        / namespace
                        / target
                        / f"pred_kept_sign_{args.prediction_sign}"
                        / GROUP
                        / "lds_summary.json"
                    )
                    tasks.append((objective, damping, target, query_id, path))

    def load(task):
        objective, damping, target, query_id, path = task
        value, recovered = read_lds(path)
        return objective, damping, target, query_id, value, recovered

    values = {}
    recovered_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for objective, damping, target, query_id, value, recovered in executor.map(load, tasks):
            values[objective, damping, target, query_id] = value
            recovered_count += int(recovered)

    if args.per_query:
        for objective in objectives:
            for damping in lambdas:
                print(
                    f"D-TRAK {objective} lambda={damping:g} — RAW — "
                    f"{args.prediction_sign.upper()}"
                )
                print(
                    f"{'QUERY':7s} {'ENDPOINT':>12s} {'TRAJ-CF':>12s} "
                    f"{'SIMPLE':>12s} {'NOISE':>12s}"
                )
                print("-" * 61)
                for query_id in query_ids:
                    row = [values[objective, damping, target, query_id] for target in TARGETS]
                    print(f"Q{query_id:<6d}" + "".join(f" {value:+11.3f}%" for value in row))
                means_row = [
                    fmean(values[objective, damping, target, qid] for qid in query_ids)
                    for target in TARGETS
                ]
                print("-" * 61)
                print(f"{'MEAN':7s}" + "".join(f" {value:+11.3f}%" for value in means_row))
                print()

    print(
        f"D-TRAK 100x1 x 100x1 — RAW — {args.prediction_sign.upper()}\n"
        f"{'OBJECTIVE':15s} {'LAMBDA':>9s}"
        f" {'ENDPOINT':>12s} {'TRAJ-CF':>12s} {'SIMPLE':>12s} {'NOISE':>12s}"
    )
    print("-" * 80)
    means = {}
    for objective in objectives:
        for damping in lambdas:
            row = []
            for target in TARGETS:
                mean = fmean(values[objective, damping, target, qid] for qid in query_ids)
                means[objective, damping, target] = mean
                row.append(mean)
            print(
                f"{objective:15s} {damping:9g}"
                + "".join(f" {value:+11.3f}%" for value in row)
            )

    print("\nBEST LAMBDA PER OBJECTIVE/TARGET")
    print(f"{'OBJECTIVE':15s} {'TARGET':24s} {'LAMBDA':>10s} {'MEAN':>12s}")
    print("-" * 66)
    for objective in objectives:
        for target in TARGETS:
            best = max(lambdas, key=lambda damping: means[objective, damping, target])
            print(
                f"{objective:15s} {target:24s} {best:10g} "
                f"{means[objective, best, target]:+11.3f}%"
            )

    expected = len(tasks)
    print(f"\nCOMPLETE: {expected}/{expected}; recovered_from_csv={recovered_count}")


if __name__ == "__main__":
    main()
