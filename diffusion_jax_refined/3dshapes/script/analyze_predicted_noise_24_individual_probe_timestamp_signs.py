#!/usr/bin/env python3
"""Evaluate timestamp-sign assignments for 24 predicted-noise probes separately."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
BANKS = ("old12", "fresh12")


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
        / "predicted_noise_old_fresh12_per_probe_timestamp_orientation"
        / f"run_{args.source_run_id}"
        / "per_probe_timestamp_scores.npz"
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as payload:
        bank_values = {
            bank: np.asarray(payload[bank], dtype=np.float64) for bank in BANKS
        }
        query_ids = np.asarray(payload["query_ids"], dtype=np.int32)
        timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        score_indices = np.asarray(payload["score_indices"], dtype=np.int64)
    expected = (12, 10, len(query_ids), len(score_indices))
    for bank, values in bank_values.items():
        if values.shape != expected:
            raise ValueError(f"{bank}: expected {expected}, got {values.shape}")

    signs = sign_matrix(len(timesteps))
    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    target_cache: dict[int, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for query_id_value in query_ids:
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
        target_cache[query_id] = load_target_data(
            cache_group(eval_root), score_indices
        )

    num_global_probes = 24
    best_signs = np.empty(
        (num_global_probes, len(query_ids), len(timesteps)), dtype=np.int8
    )
    best_binary = np.empty(
        (num_global_probes, len(query_ids), len(timesteps), len(score_indices)),
        dtype=np.bool_,
    )
    extrema_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    summary_values: dict[tuple[int, str], list[float]] = defaultdict(list)

    print("24 INDIVIDUAL PROBES — QUERY-SPECIFIC TIMESTAMP-SIGN ORACLE")
    print(
        f"{'PROBE':>5s} {'BANK':8s} {'Q':>2s} {'ALL+ CF':>10s} "
        f"{'MAX CF':>10s} {'GAIN':>9s} {'SIGNS'}"
    )
    print("-" * 105)
    for bank_slot, bank in enumerate(BANKS):
        for probe_slot in range(12):
            global_probe = bank_slot * 12 + probe_slot + 1
            for qslot, query_id_value in enumerate(query_ids):
                query_id = int(query_id_value)
                components = bank_values[bank][probe_slot, :, qslot, :]
                incidence, true_values = target_cache[query_id]
                timestamp_predictions = components @ incidence.T
                predictions = signs @ timestamp_predictions
                lds = {
                    target: 100.0 * rowwise_spearman(predictions, true_values[target])
                    for target in TARGETS
                }
                cf = 0.5 * (
                    lds["endpoint_contarfactual"]
                    + lds["traj_contarfactual"]
                )
                maximum = int(np.argmax(cf))
                chosen_signs = signs[maximum]
                best_signs[global_probe - 1, qslot] = chosen_signs.astype(np.int8)
                best_binary[global_probe - 1, qslot] = (
                    chosen_signs[:, None] * components > 0.0
                )
                row: dict[str, object] = {
                    "global_probe": global_probe,
                    "bank": bank,
                    "probe_in_bank": probe_slot + 1,
                    "query": query_id,
                    "all_plus_cf_joint_lds_percent": float(cf[0]),
                    "max_cf_joint_lds_percent": float(cf[maximum]),
                    "gain_cf_joint_lds_percent": float(cf[maximum] - cf[0]),
                    "max_mask": maximum,
                    "max_signs": sign_label(chosen_signs, timesteps),
                }
                for target in TARGETS:
                    row[f"all_plus_{target}_lds_percent"] = float(lds[target][0])
                    row[f"max_{target}_lds_percent"] = float(lds[target][maximum])
                extrema_rows.append(row)
                summary_values[(global_probe, "all_plus")].append(float(cf[0]))
                summary_values[(global_probe, "max")].append(float(cf[maximum]))
                print(
                    f"{global_probe:5d} {bank:8s} {query_id:2d} "
                    f"{cf[0]:9.3f}% {cf[maximum]:9.3f}% "
                    f"{cf[maximum] - cf[0]:8.3f}% "
                    f"{sign_label(chosen_signs, timesteps)}"
                )
                for tslot, timestep in enumerate(timesteps):
                    values = components[tslot]
                    oriented = chosen_signs[tslot] * values
                    distribution_rows.append(
                        {
                            "global_probe": global_probe,
                            "bank": bank,
                            "probe_in_bank": probe_slot + 1,
                            "query": query_id,
                            "timestep": int(timestep),
                            "selected_sign": int(chosen_signs[tslot]),
                            "original_positive_fraction": float(np.mean(values > 0)),
                            "selected_positive_fraction": float(np.mean(oriented > 0)),
                            "original_mean": float(np.mean(values)),
                            "original_std": float(np.std(values)),
                        }
                    )

    summary_rows: list[dict[str, object]] = []
    print("\n10-query mean by individual probe")
    print(
        f"{'PROBE':>5s} {'BANK':8s} {'ALL+ CF':>10s} {'MAX CF':>10s} {'GAIN':>9s}"
    )
    print("-" * 52)
    for global_probe in range(1, num_global_probes + 1):
        bank = BANKS[(global_probe - 1) // 12]
        probe_slot = (global_probe - 1) % 12 + 1
        original = float(np.mean(summary_values[(global_probe, "all_plus")]))
        maximum = float(np.mean(summary_values[(global_probe, "max")]))
        summary_rows.append(
            {
                "global_probe": global_probe,
                "bank": bank,
                "probe_in_bank": probe_slot,
                "all_plus_ten_query_mean_cf_joint_lds_percent": original,
                "query_specific_max_ten_query_mean_cf_joint_lds_percent": maximum,
                "mean_gain_cf_joint_lds_percent": maximum - original,
                "max_lds_query_std_percent": float(
                    np.std(summary_values[(global_probe, "max")], ddof=1)
                ),
            }
        )
        print(
            f"{global_probe:5d} {bank:8s} {original:9.3f}% "
            f"{maximum:9.3f}% {maximum - original:8.3f}%"
        )

    output = (
        result_root
        / "eval"
        / "predicted_noise_24_individual_probe_timestamp_signs"
        / f"source_run_{args.source_run_id}"
    )
    write_csv(output / "per_probe_query_extrema.csv", extrema_rows)
    write_csv(output / "per_probe_timestamp_sign_distribution.csv", distribution_rows)
    write_csv(output / "per_probe_ten_query_summary.csv", summary_rows)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "individual_probe_best_binary_vectors.npz",
        query_ids=query_ids,
        timesteps=timesteps,
        score_indices=score_indices,
        global_probes=np.arange(1, 25, dtype=np.int32),
        best_signs=best_signs,
        best_binary=best_binary,
    )
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
