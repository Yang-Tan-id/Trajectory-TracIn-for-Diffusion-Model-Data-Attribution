#!/usr/bin/env python3
"""Compare checkpoint predicted-noise update directions across queries."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from analyze_predicted_noise_probe24_output_alignment import artifact_path, write_csv
from analyze_q1348_own_trajectory_product_square import parse_ints


def load_signs(path, variant):
    result = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["reduction"] == "linear"
                and row["variant"] == variant
                and row["prediction_sign"] == "p1"
            ):
                value = float(row["cf_joint_percent"])
                result[int(row["query"])] = "p1" if value >= 0.0 else "m1"
    return result


def cosine(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / max(float(denominator), 1e-12))


def ddim_noise_coefficient(timestep, alpha_bars):
    timestep = int(timestep)
    previous = timestep - 1
    alpha_t = float(alpha_bars[timestep])
    alpha_previous = 1.0 if previous < 0 else float(alpha_bars[previous])
    return float(
        np.sqrt(1.0 - alpha_previous)
        - np.sqrt(alpha_previous / alpha_t) * np.sqrt(1.0 - alpha_t)
    )


def mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--namespace", default="predicted_noise_cross_query_own_trajectory"
    )
    parser.add_argument("--score-results", type=Path, required=True)
    parser.add_argument("--variant", default="query_l2")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    query_ids = parse_ints(args.query_ids)
    signs = load_signs(args.score_results, args.variant)
    if set(signs) != set(query_ids):
        raise ValueError(f"missing query signs: {sorted(set(query_ids) - set(signs))}")

    banks = []
    timesteps = None
    alpha_bars = None
    for query in query_ids:
        path = artifact_path(
            args.experiment, args.train_seed, args.epochs, query, args.namespace
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            bank = np.asarray(
                payload["checkpoint_next_predicted_noise_deltas"],
                dtype=np.float64,
            )
            term_timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            query_alpha_bars = np.asarray(
                payload["diffusion_alphas_cumprod"], dtype=np.float64
            )
        snapshot_count = bank.shape[1]
        current_timesteps = term_timesteps[:snapshot_count]
        if timesteps is None:
            timesteps = current_timesteps
            alpha_bars = query_alpha_bars
        elif not np.array_equal(timesteps, current_timesteps):
            raise ValueError(f"Q{query} timestep mismatch")
        banks.append(bank)
    updates = np.stack(banks, axis=0)
    if updates.shape[:3] != (len(query_ids), 49, len(timesteps)):
        raise ValueError(f"unexpected update bank shape {updates.shape}")

    betas = np.linspace(1e-4, 0.02, 1000, dtype=np.float64)
    fallback_alpha_bars = np.cumprod(1.0 - betas)
    if alpha_bars is None or len(alpha_bars) != 1000:
        alpha_bars = fallback_alpha_bars
    coefficients = {
        int(timestep): ddim_noise_coefficient(int(timestep), alpha_bars)
        for timestep in timesteps
    }

    pair_rows = []
    summary_accumulator = defaultdict(lambda: defaultdict(list))
    for checkpoint in range(updates.shape[1]):
        for slot, timestep in enumerate(timesteps):
            vectors = updates[:, checkpoint, slot].reshape(len(query_ids), -1)
            common = vectors.mean(axis=0)
            residuals = vectors - common[None, :]
            vector_energy = np.mean(np.sum(np.square(vectors), axis=1))
            common_fraction = float(
                np.dot(common, common) / max(float(vector_energy), 1e-12)
            )
            coefficient = coefficients[int(timestep)]
            for left, right in combinations(range(len(query_ids)), 2):
                left_query = query_ids[left]
                right_query = query_ids[right]
                if signs[left_query] == signs[right_query] == "p1":
                    pair_type = "p1_p1"
                elif signs[left_query] == signs[right_query] == "m1":
                    pair_type = "m1_m1"
                else:
                    pair_type = "p1_m1"
                raw_cosine = cosine(vectors[left], vectors[right])
                residual_cosine = cosine(residuals[left], residuals[right])
                pair_rows.append(
                    {
                        "checkpoint": checkpoint + 1,
                        "epoch": 4 * (checkpoint + 1),
                        "checkpoint_bin": min(checkpoint // 10 + 1, 5),
                        "timestep": int(timestep),
                        "left_query": left_query,
                        "right_query": right_query,
                        "left_sign": signs[left_query],
                        "right_sign": signs[right_query],
                        "pair_type": pair_type,
                        "raw_update_cosine": raw_cosine,
                        "residual_update_cosine": residual_cosine,
                        "common_energy_fraction": common_fraction,
                        "ddim_noise_coefficient": coefficient,
                        "ddim_oriented_pair_cosine": raw_cosine,
                    }
                )
                for grouping, key in (
                    ("overall", (pair_type,)),
                    ("timestep", (pair_type, int(timestep))),
                    (
                        "checkpoint_bin",
                        (pair_type, min(checkpoint // 10 + 1, 5)),
                    ),
                ):
                    target = summary_accumulator[(grouping, key)]
                    target["raw"].append(raw_cosine)
                    target["residual"].append(residual_cosine)
                    target["common"].append(common_fraction)

    summary_rows = []
    for (grouping, key), values in sorted(summary_accumulator.items()):
        row = {
            "grouping": grouping,
            "pair_type": key[0],
            "timestep": key[1] if grouping == "timestep" else "",
            "checkpoint_bin": key[1] if grouping == "checkpoint_bin" else "",
            "pairs": len(values["raw"]),
            "raw_cosine_mean": mean_or_nan(values["raw"]),
            "raw_cosine_positive_fraction": mean_or_nan(
                [value > 0.0 for value in values["raw"]]
            ),
            "residual_cosine_mean": mean_or_nan(values["residual"]),
            "residual_cosine_positive_fraction": mean_or_nan(
                [value > 0.0 for value in values["residual"]]
            ),
            "common_energy_fraction_mean": mean_or_nan(values["common"]),
        }
        summary_rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_pair_checkpoint_timestep.csv", pair_rows)
    write_csv(args.out_dir / "summary.csv", summary_rows)
    print(
        f"CROSS-QUERY OWN-TRAJECTORY PREDICTED-NOISE UPDATES — {args.variant.upper()} SIGNS"
    )
    print("PAIR    N RAW-COS RAW+ RESID-COS RESID+ COMMON-ENERGY")
    print("-" * 66)
    for row in summary_rows:
        if row["grouping"] != "overall":
            continue
        print(
            f"{row['pair_type']:<7s} {int(row['pairs']):5d} "
            f"{row['raw_cosine_mean']:+7.4f} "
            f"{row['raw_cosine_positive_fraction']:5.3f} "
            f"{row['residual_cosine_mean']:+9.4f} "
            f"{row['residual_cosine_positive_fraction']:6.3f} "
            f"{row['common_energy_fraction_mean']:13.4f}"
        )
    print("\nBY TIMESTAMP")
    print("PAIR       T RAW-COS RESID-COS COMMON-ENERGY")
    for row in summary_rows:
        if row["grouping"] != "timestep":
            continue
        print(
            f"{row['pair_type']:<7s} {int(row['timestep']):4d} "
            f"{row['raw_cosine_mean']:+7.4f} "
            f"{row['residual_cosine_mean']:+9.4f} "
            f"{row['common_energy_fraction_mean']:13.4f}"
        )
    print("\nBY CHECKPOINT BIN")
    print("PAIR    BIN RAW-COS RESID-COS COMMON-ENERGY")
    for row in summary_rows:
        if row["grouping"] != "checkpoint_bin":
            continue
        print(
            f"{row['pair_type']:<7s} {int(row['checkpoint_bin']):3d} "
            f"{row['raw_cosine_mean']:+7.4f} "
            f"{row['residual_cosine_mean']:+9.4f} "
            f"{row['common_energy_fraction_mean']:13.4f}"
        )
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
