#!/usr/bin/env python3
"""Align old/fresh timestamp-component signs, then search common sign flips."""

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


def best_relative_fresh_signs(
    old: np.ndarray, fresh: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-timestamp fresh flips and raw/aligned sign agreement.

    Both inputs have shape (timestamps, datapoints). A single sign is chosen for
    each timestamp; no datapoint-specific sign choice is allowed.
    """
    if old.shape != fresh.shape or old.ndim != 2:
        raise ValueError("old and fresh must share shape (timestamps, datapoints)")
    raw = np.mean(np.signbit(old) == np.signbit(fresh), axis=1)
    relative = np.where(raw >= 0.5, 1.0, -1.0)
    aligned = np.mean(
        np.signbit(old) == np.signbit(relative[:, None] * fresh), axis=1
    )
    return relative, raw, aligned


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
        old = np.asarray(payload["old_timestamp"], dtype=np.float64)
        fresh = np.asarray(payload["fresh_timestamp"], dtype=np.float64)
        query_ids = np.asarray(payload["query_ids"], dtype=np.int32)
        timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        score_indices = np.asarray(payload["score_indices"], dtype=np.int64)
    expected = (len(query_ids), len(timesteps), len(score_indices))
    if old.shape != expected or fresh.shape != expected:
        raise ValueError(
            f"expected old/fresh shape {expected}, got {old.shape}/{fresh.shape}"
        )

    common_signs = sign_matrix(len(timesteps))
    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    alignment_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    selected_common = np.empty((len(query_ids), len(timesteps)), dtype=np.int8)
    selected_relative = np.empty_like(selected_common)
    old_binary = np.empty(expected, dtype=np.bool_)
    fresh_binary = np.empty(expected, dtype=np.bool_)

    print("MATCHED OLD12/FRESH12 TIMESTAMP COMPONENT SIGNS — QUERY-SPECIFIC")
    print(
        f"{'Q':>2s} {'MATCH':>7s} {'OBJECTIVE':16s} {'OLD CF':>9s} "
        f"{'FRESH CF':>9s} {'MEAN':>9s} {'MIN':>9s}"
    )
    print("-" * 78)

    for qslot, query_id_value in enumerate(query_ids):
        query_id = int(query_id_value)
        relative, raw_agreement, aligned_agreement = best_relative_fresh_signs(
            old[qslot], fresh[qslot]
        )
        selected_relative[qslot] = relative.astype(np.int8)
        for tslot, timestep in enumerate(timesteps):
            alignment_rows.append(
                {
                    "query": query_id,
                    "timestep": int(timestep),
                    "raw_same_sign_fraction": float(raw_agreement[tslot]),
                    "fresh_relative_sign": int(relative[tslot]),
                    "matched_same_sign_fraction": float(aligned_agreement[tslot]),
                }
            )

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
        old_timestamp_predictions = old[qslot] @ incidence.T
        fresh_timestamp_predictions = fresh[qslot] @ incidence.T
        old_predictions = common_signs @ old_timestamp_predictions
        fresh_signs = common_signs * relative[None, :]
        fresh_predictions = fresh_signs @ fresh_timestamp_predictions

        bank_lds: dict[str, dict[str, np.ndarray]] = {"old12": {}, "fresh12": {}}
        for target in TARGETS:
            bank_lds["old12"][target] = 100.0 * rowwise_spearman(
                old_predictions, true_values[target]
            )
            bank_lds["fresh12"][target] = 100.0 * rowwise_spearman(
                fresh_predictions, true_values[target]
            )
        old_cf = 0.5 * (
            bank_lds["old12"]["endpoint_contarfactual"]
            + bank_lds["old12"]["traj_contarfactual"]
        )
        fresh_cf = 0.5 * (
            bank_lds["fresh12"]["endpoint_contarfactual"]
            + bank_lds["fresh12"]["traj_contarfactual"]
        )
        objectives = {
            "max_bank_mean": 0.5 * (old_cf + fresh_cf),
            "max_worst_bank": np.minimum(old_cf, fresh_cf),
            "max_old12": old_cf,
            "max_fresh12": fresh_cf,
        }
        chosen: dict[str, int] = {
            name: int(np.argmax(values)) for name, values in objectives.items()
        }
        match_mean = float(np.mean(aligned_agreement))
        for objective, mask in chosen.items():
            row = {
                "query": query_id,
                "objective": objective,
                "matched_same_sign_fraction": match_mean,
                "mask": mask,
                "common_signs": sign_label(common_signs[mask], timesteps),
                "fresh_relative_signs": sign_label(relative, timesteps),
                "fresh_final_signs": sign_label(fresh_signs[mask], timesteps),
                "old_cf_joint_lds_percent": float(old_cf[mask]),
                "fresh_cf_joint_lds_percent": float(fresh_cf[mask]),
                "bank_mean_cf_joint_lds_percent": float(
                    0.5 * (old_cf[mask] + fresh_cf[mask])
                ),
                "worst_bank_cf_joint_lds_percent": float(
                    min(old_cf[mask], fresh_cf[mask])
                ),
            }
            for bank in ("old12", "fresh12"):
                for target in TARGETS:
                    row[f"{bank}_{target}_lds_percent"] = float(
                        bank_lds[bank][target][mask]
                    )
            result_rows.append(row)
            print(
                f"{query_id:2d} {match_mean:7.3f} {objective:16s} "
                f"{old_cf[mask]:8.3f}% {fresh_cf[mask]:8.3f}% "
                f"{0.5 * (old_cf[mask] + fresh_cf[mask]):8.3f}% "
                f"{min(old_cf[mask], fresh_cf[mask]):8.3f}%"
            )

        best = chosen["max_bank_mean"]
        selected_common[qslot] = common_signs[best].astype(np.int8)
        old_components = common_signs[best, :, None] * old[qslot]
        fresh_components = fresh_signs[best, :, None] * fresh[qslot]
        old_binary[qslot] = old_components > 0
        fresh_binary[qslot] = fresh_components > 0

    output = (
        result_root
        / "eval"
        / "predicted_noise_old_fresh12_matched_timestamp_signs"
        / f"source_run_{args.source_run_id}"
    )
    write_csv(output / "relative_sign_alignment.csv", alignment_rows)
    write_csv(output / "matched_sign_lds.csv", result_rows)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "max_bank_mean_binary_vectors.npz",
        query_ids=query_ids,
        timesteps=timesteps,
        score_indices=score_indices,
        common_signs=selected_common,
        fresh_relative_signs=selected_relative,
        old_binary=old_binary,
        fresh_binary=fresh_binary,
    )

    mean_rows = [row for row in result_rows if row["objective"] == "max_bank_mean"]
    robust_rows = [row for row in result_rows if row["objective"] == "max_worst_bank"]
    print("\n10-query means")
    for name, rows in (("max_bank_mean", mean_rows), ("max_worst_bank", robust_rows)):
        old_mean = np.mean([float(row["old_cf_joint_lds_percent"]) for row in rows])
        fresh_mean = np.mean([float(row["fresh_cf_joint_lds_percent"]) for row in rows])
        bank_mean = 0.5 * (old_mean + fresh_mean)
        worst_mean = np.mean(
            [float(row["worst_bank_cf_joint_lds_percent"]) for row in rows]
        )
        match = np.mean([float(row["matched_same_sign_fraction"]) for row in rows])
        print(
            f"{name:16s} match={match:.3f} old={old_mean:.3f}% "
            f"fresh={fresh_mean:.3f}% mean={bank_mean:.3f}% "
            f"mean-worst={worst_mean:.3f}%"
        )
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
