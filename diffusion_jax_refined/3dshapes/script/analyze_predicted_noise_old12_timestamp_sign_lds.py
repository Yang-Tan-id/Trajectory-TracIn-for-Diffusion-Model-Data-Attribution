#!/usr/bin/env python3
"""Evaluate a 12-probe bank after splitting timestamps by score-mean sign."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_predicted_noise_probe8_choose4 import (
    TARGETS,
    cache_group,
    load_target_data,
    spearman,
    write_csv,
)


SHAPES_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--bank", choices=("old12", "fresh12"), default="old12")
    parser.add_argument("--prediction-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    args = parser.parse_args()

    result_root = SHAPES_ROOT / "result" / args.experiment
    source = (
        result_root
        / "eval"
        / "predicted_noise_old_fresh12_grouped_signs"
        / f"run_{args.source_run_id}"
        / "grouped_scores.npz"
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as payload:
        timestamp_scores = np.asarray(
            payload["old_timestamp" if args.bank == "old12" else "fresh_timestamp"],
            dtype=np.float64,
        )
        query_ids = np.asarray(payload["query_ids"], dtype=np.int32)
        timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        score_indices = np.asarray(payload["score_indices"], dtype=np.int64)
    expected_shape = (len(query_ids), 10, len(score_indices))
    if timestamp_scores.shape != expected_shape:
        raise ValueError(
            f"{args.bank} timestamp scores expected {expected_shape}, "
            f"got {timestamp_scores.shape}"
        )

    full_method = f"full_{args.bank}"
    methods = (
        "positive_timestamp_means",
        "negative_timestamp_means",
        "negative_timestamp_means_flipped",
        full_method,
    )

    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []

    print(
        f"{args.bank.upper()} TIMESTAMP-MEAN SIGN SPLIT — BOTH-L2, "
        f"prediction sign {args.prediction_sign:+g}"
    )
    print(
        f"{'Q':>2s} {'METHOD':35s} {'ENDPOINT':>10s} {'TRAJ':>10s} "
        f"{'CF JOINT':>10s} {'NOISE':>10s} {'SIMPLE':>10s}  TIMESTAMPS"
    )
    print("-" * 132)

    for qslot, query_id_value in enumerate(query_ids):
        query_id = int(query_id_value)
        record = records[query_id]
        prompt = str(record["prompt"])
        prompt_tag = prompt.replace(",", "_")
        eval_root = (
            result_root
            / "eval"
            / "prompted_solo"
            / f"query_{prompt_tag}"
            / f"initial_seed_{int(record['initial_seed'])}"
        )
        incidence, true_values = load_target_data(cache_group(eval_root), score_indices)

        timestamp_means = timestamp_scores[qslot].mean(axis=1)
        positive = timestamp_means > 0.0
        negative = timestamp_means < 0.0
        if not positive.any() or not negative.any():
            raise ValueError(f"query {query_id} does not have both timestamp-mean signs")
        positive_ts = [int(value) for value in timesteps[positive]]
        negative_ts = [int(value) for value in timesteps[negative]]
        scores = {
            "positive_timestamp_means": timestamp_scores[qslot, positive].sum(axis=0),
            "negative_timestamp_means": timestamp_scores[qslot, negative].sum(axis=0),
            "negative_timestamp_means_flipped": -timestamp_scores[
                qslot, negative
            ].sum(axis=0),
            full_method: timestamp_scores[qslot].sum(axis=0),
        }
        method_timestamps = {
            "positive_timestamp_means": positive_ts,
            "negative_timestamp_means": negative_ts,
            "negative_timestamp_means_flipped": negative_ts,
            full_method: [int(value) for value in timesteps],
        }
        for slot, timestep in enumerate(timesteps):
            split_rows.append(
                {
                    "query": query_id,
                    "timestep": int(timestep),
                    "bank_mean": float(timestamp_means[slot]),
                    "mean_sign_group": "positive" if positive[slot] else "negative",
                    "prompt": prompt_tag,
                }
            )

        query_lookup: dict[tuple[str, str], float] = {}
        for method in methods:
            prediction = args.prediction_sign * scores[method] @ incidence.T
            for target in TARGETS:
                value = 100.0 * spearman(prediction, true_values[target])
                query_lookup[(method, target)] = value
                rows.append(
                    {
                        "query": query_id,
                        "method": method,
                        "prediction_sign": args.prediction_sign,
                        "target": target,
                        "lds_percent": value,
                        "timestamps": ",".join(map(str, method_timestamps[method])),
                        "prompt": prompt_tag,
                    }
                )
            endpoint = query_lookup[(method, "endpoint_contarfactual")]
            trajectory = query_lookup[(method, "traj_contarfactual")]
            print(
                f"{query_id:2d} {method:35s} "
                f"{endpoint:9.3f}% {trajectory:9.3f}% "
                f"{0.5 * (endpoint + trajectory):9.3f}% "
                f"{query_lookup[(method, 'noise_trajectory')]:9.3f}% "
                f"{query_lookup[(method, 'simple_loss')]:9.3f}%  "
                f"{','.join(map(str, method_timestamps[method]))}"
            )

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["target"]))].append(
            float(row["lds_percent"])
        )
    summary_rows: list[dict[str, object]] = []
    print("\n10-query mean")
    print(
        f"{'METHOD':35s} {'ENDPOINT':>10s} {'TRAJ':>10s} "
        f"{'CF JOINT':>10s} {'NOISE':>10s} {'SIMPLE':>10s}"
    )
    print("-" * 94)
    for method in methods:
        means = {
            target: statistics.mean(grouped[(method, target)]) for target in TARGETS
        }
        joint = 0.5 * (
            means["endpoint_contarfactual"] + means["traj_contarfactual"]
        )
        print(
            f"{method:35s} "
            f"{means['endpoint_contarfactual']:9.3f}% "
            f"{means['traj_contarfactual']:9.3f}% {joint:9.3f}% "
            f"{means['noise_trajectory']:9.3f}% {means['simple_loss']:9.3f}%"
        )
        for target in TARGETS:
            summary_rows.append(
                {
                    "method": method,
                    "target": target,
                    "mean_lds_percent": means[target],
                    "std_lds_percent": statistics.stdev(grouped[(method, target)]),
                }
            )

    output = (
        result_root
        / "eval"
        / f"predicted_noise_{args.bank}_timestamp_mean_sign_lds"
        / f"source_run_{args.source_run_id}"
    )
    write_csv(output / "per_query_lds.csv", rows)
    write_csv(output / "timestamp_groups.csv", split_rows)
    write_csv(output / "ten_query_summary.csv", summary_rows)
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
