#!/usr/bin/env python3
"""Exhaustively evaluate all 2^10 timestamp signs for old12 and fresh12."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_predicted_noise_old_fresh_term_signs import write_csv
from analyze_predicted_noise_probe12_sign_flips import rowwise_spearman, sign_matrix
from analyze_predicted_noise_probe8_choose4 import (
    TARGETS,
    cache_group,
    load_target_data,
)


SHAPES_ROOT = Path(__file__).resolve().parents[1]
BANK_KEYS = {"old12": "old_timestamp", "fresh12": "fresh_timestamp"}


def sign_label(signs: np.ndarray, timesteps: np.ndarray) -> str:
    return ",".join(
        f"{int(timestep)}:{'+' if sign > 0 else '-'}"
        for timestep, sign in zip(timesteps, signs)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--source-run-id", required=True)
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
        banks = {
            bank: np.asarray(payload[key], dtype=np.float64)
            for bank, key in BANK_KEYS.items()
        }
        query_ids = np.asarray(payload["query_ids"], dtype=np.int32)
        timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        score_indices = np.asarray(payload["score_indices"], dtype=np.int64)
    expected_shape = (len(query_ids), 10, len(score_indices))
    for bank, values in banks.items():
        if values.shape != expected_shape:
            raise ValueError(f"{bank} expected {expected_shape}, got {values.shape}")

    signs = sign_matrix(10)
    labels = [sign_label(row, timesteps) for row in signs]
    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    lds = {
        bank: {
            target: np.empty((len(signs), len(query_ids)), dtype=np.float64)
            for target in TARGETS
        }
        for bank in banks
    }

    for qslot, query_id_value in enumerate(query_ids):
        query_id = int(query_id_value)
        record = records[query_id]
        prompt_tag = str(record["prompt"]).replace(",", "_")
        eval_root = (
            result_root
            / "eval"
            / "prompted_solo"
            / f"query_{prompt_tag}"
            / f"initial_seed_{int(record['initial_seed'])}"
        )
        incidence, true_values = load_target_data(cache_group(eval_root), score_indices)
        for bank, values in banks.items():
            timestamp_predictions = values[qslot] @ incidence.T
            predictions = signs @ timestamp_predictions
            for target in TARGETS:
                lds[bank][target][:, qslot] = 100.0 * rowwise_spearman(
                    predictions, true_values[target]
                )
        print(f"[query {query_id}] evaluated 1024 timestamp signs for both banks")

    assignment_rows: list[dict[str, object]] = []
    shared_rows: list[dict[str, object]] = []
    per_query_rows: list[dict[str, object]] = []

    print("\nSHARED TIMESTAMP SIGN ASSIGNMENT — 10-QUERY MEAN")
    print(
        f"{'BANK':8s} {'TARGET':24s} {'ALL+':>9s} {'MIN':>9s} {'MAX':>9s} "
        f"{'MIN SIGNS':>35s} {'MAX SIGNS':>35s}"
    )
    print("-" * 140)
    for bank in banks:
        means = {target: values.mean(axis=1) for target, values in lds[bank].items()}
        joint = 0.5 * (
            means["endpoint_contarfactual"] + means["traj_contarfactual"]
        )
        metrics = {**means, "cf_joint": joint}
        for target, values in metrics.items():
            minimum = int(np.argmin(values))
            maximum = int(np.argmax(values))
            pair_error = float(np.max(np.abs(values + values[::-1])))
            shared_rows.append(
                {
                    "bank": bank,
                    "target": target,
                    "all_plus_lds_percent": float(values[0]),
                    "min_lds_percent": float(values[minimum]),
                    "min_mask": minimum,
                    "min_signs": labels[minimum],
                    "max_lds_percent": float(values[maximum]),
                    "max_mask": maximum,
                    "max_signs": labels[maximum],
                    "all_plus_percentile": float(100.0 * np.mean(values <= values[0])),
                    "max_sign_pair_symmetry_error": pair_error,
                }
            )
            print(
                f"{bank:8s} {target:24s} {values[0]:8.3f}% "
                f"{values[minimum]:8.3f}% {values[maximum]:8.3f}% "
                f"{labels[minimum]:>35s} {labels[maximum]:>35s}"
            )
        for mask, label in enumerate(labels):
            assignment_rows.append(
                {
                    "bank": bank,
                    "mask": mask,
                    "signs": label,
                    "endpoint_mean_lds_percent": float(
                        means["endpoint_contarfactual"][mask]
                    ),
                    "traj_mean_lds_percent": float(
                        means["traj_contarfactual"][mask]
                    ),
                    "cf_joint_mean_lds_percent": float(joint[mask]),
                    "noise_mean_lds_percent": float(means["noise_trajectory"][mask]),
                    "simple_mean_lds_percent": float(means["simple_loss"][mask]),
                }
            )

    print("\nQUERY-SPECIFIC CF-JOINT ORACLE EXTREMA")
    print(
        f"{'BANK':8s} {'Q':>2s} {'ALL+':>9s} {'MIN':>9s} {'MAX':>9s} "
        f"{'MIN SIGNS':>35s} {'MAX SIGNS':>35s}"
    )
    print("-" * 132)
    for bank in banks:
        joint = 0.5 * (
            lds[bank]["endpoint_contarfactual"]
            + lds[bank]["traj_contarfactual"]
        )
        for qslot, query_id_value in enumerate(query_ids):
            query_values = joint[:, qslot]
            minimum = int(np.argmin(query_values))
            maximum = int(np.argmax(query_values))
            row = {
                "bank": bank,
                "query": int(query_id_value),
                "all_plus_cf_joint_lds_percent": float(query_values[0]),
                "min_cf_joint_lds_percent": float(query_values[minimum]),
                "min_mask": minimum,
                "min_signs": labels[minimum],
                "max_cf_joint_lds_percent": float(query_values[maximum]),
                "max_mask": maximum,
                "max_signs": labels[maximum],
                "all_plus_percentile": float(
                    100.0 * np.mean(query_values <= query_values[0])
                ),
            }
            per_query_rows.append(row)
            print(
                f"{bank:8s} {int(query_id_value):2d} {query_values[0]:8.3f}% "
                f"{query_values[minimum]:8.3f}% {query_values[maximum]:8.3f}% "
                f"{labels[minimum]:>35s} {labels[maximum]:>35s}"
            )

    output = (
        result_root
        / "eval"
        / "predicted_noise_old_fresh12_timestamp_sign_flips"
        / f"source_run_{args.source_run_id}"
    )
    write_csv(output / "all_shared_assignments.csv", assignment_rows)
    write_csv(output / "shared_extrema.csv", shared_rows)
    write_csv(output / "per_query_cf_joint_extrema.csv", per_query_rows)
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
