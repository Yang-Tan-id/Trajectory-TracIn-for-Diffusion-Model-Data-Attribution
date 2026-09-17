#!/usr/bin/env python3
"""Summarize Q8 query-loss derivatives along actual checkpoint updates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from analyze_predicted_noise_probe24_output_alignment import artifact_path, write_csv
from analyze_q8_checkpoint_sign_predicted_noise_geometry import majority_signs


def aggregate(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for values, group in sorted(groups.items()):
        result = dict(zip(keys, values))
        result["terms"] = len(group)
        for metric in (
            "directional_derivative",
            "derivative_per_update_norm",
            "gradient_update_cosine",
        ):
            samples = np.asarray([row[metric] for row in group], dtype=np.float64)
            result[f"{metric}_mean"] = float(np.mean(samples))
            result[f"{metric}_std"] = float(np.std(samples))
            result[f"{metric}_positive_fraction"] = float(np.mean(samples > 0.0))
        output.append(result)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-id", type=int, default=8)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--checkpoint-crossfit-dir", type=Path, required=True)
    parser.add_argument("--variant", default="query_train_l2")
    parser.add_argument("--method", default="five_bins")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    signs, stability, split_count = majority_signs(
        args.checkpoint_crossfit_dir / "per_split.csv",
        args.query_id,
        args.variant,
        args.method,
        "checkpoint",
    )
    path = artifact_path(
        args.experiment, args.train_seed, args.epochs, args.query_id, args.namespace
    )
    with np.load(path, allow_pickle=False) as payload:
        checkpoints = np.asarray(payload["ckpt_indices"], dtype=np.int32)
        timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        derivatives = np.asarray(
            payload["parameter_directional_derivatives"], dtype=np.float64
        )
        normalized = np.asarray(
            payload["parameter_directional_derivatives_per_update_norm"],
            dtype=np.float64,
        )
        cosines = np.asarray(
            payload["parameter_update_query_gradient_cosines"], dtype=np.float64
        )
        update_norms = np.asarray(payload["parameter_update_norms"], dtype=np.float64)

    if not (
        len(checkpoints)
        == len(timesteps)
        == len(derivatives)
        == len(normalized)
        == len(cosines)
        == len(update_norms)
    ):
        raise ValueError("directional-derivative artifact arrays are not aligned")

    rows = []
    for index, checkpoint in enumerate(checkpoints):
        checkpoint_index = int(checkpoint)
        rows.append(
            {
                "checkpoint": checkpoint_index + 1,
                "epoch": 4 * (checkpoint_index + 1),
                "checkpoint_bin": min(checkpoint_index // 10 + 1, 5),
                "flip_sign": int(signs[checkpoint_index]),
                "flip_stability": float(stability[checkpoint_index]),
                "timestep": int(timesteps[index]),
                "directional_derivative": float(derivatives[index]),
                "derivative_per_update_norm": float(normalized[index]),
                "gradient_update_cosine": float(cosines[index]),
                "parameter_update_norm": float(update_norms[index]),
            }
        )

    by_bin = aggregate(rows, ("checkpoint_bin", "flip_sign"))
    by_bin_timestep = aggregate(rows, ("checkpoint_bin", "flip_sign", "timestep"))
    by_sign = aggregate(rows, ("flip_sign",))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_checkpoint_timestep.csv", rows)
    write_csv(args.out_dir / "by_checkpoint_bin.csv", by_bin)
    write_csv(args.out_dir / "by_checkpoint_bin_timestep.csv", by_bin_timestep)
    write_csv(args.out_dir / "by_flip_sign.csv", by_sign)

    print(
        f"Q{args.query_id} QUERY-LOSS GRADIENT vs PARAMETER UPDATE; "
        f"signs from {split_count} splits"
    )
    print("D = <grad_theta loss_q(c,t), theta_(c+1)-theta_c>")
    print("D<0 predicts that the finite checkpoint update locally lowers query loss.\n")
    print("BY CHECKPOINT BIN")
    print("BIN FLIP TERMS       D MEAN    D>0   D/NORM MEAN  COS MEAN  COS>0")
    for row in by_bin:
        print(
            f"{int(row['checkpoint_bin']):3d} {int(row['flip_sign']):+4d} "
            f"{int(row['terms']):5d} "
            f"{row['directional_derivative_mean']:+12.5e} "
            f"{row['directional_derivative_positive_fraction']:6.3f} "
            f"{row['derivative_per_update_norm_mean']:+12.5e} "
            f"{row['gradient_update_cosine_mean']:+9.5f} "
            f"{row['gradient_update_cosine_positive_fraction']:6.3f}"
        )

    print("\nBY CHECKPOINT BIN AND TIMESTAMP")
    print("BIN FLIP    T   N       D MEAN    D>0   COS MEAN  COS>0")
    for row in sorted(
        by_bin_timestep,
        key=lambda item: (item["checkpoint_bin"], -item["timestep"]),
    ):
        print(
            f"{int(row['checkpoint_bin']):3d} {int(row['flip_sign']):+4d} "
            f"{int(row['timestep']):4d} {int(row['terms']):3d} "
            f"{row['directional_derivative_mean']:+12.5e} "
            f"{row['directional_derivative_positive_fraction']:6.3f} "
            f"{row['gradient_update_cosine_mean']:+9.5f} "
            f"{row['gradient_update_cosine_positive_fraction']:6.3f}"
        )
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
