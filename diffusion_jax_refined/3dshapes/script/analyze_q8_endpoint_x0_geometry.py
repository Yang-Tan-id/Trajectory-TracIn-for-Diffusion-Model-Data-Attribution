#!/usr/bin/env python3
"""Compare checkpoint predicted-noise changes with the actual sampled endpoint x0."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from analyze_predicted_noise_probe24_output_alignment import artifact_path, write_csv
from analyze_q8_checkpoint_sign_predicted_noise_geometry import majority_signs


METRICS = (
    "current_cos_endpoint_direction",
    "next_cos_endpoint_direction",
    "step_cos_endpoint_direction",
    "current_cos_endpoint_implied_noise",
    "next_cos_endpoint_implied_noise",
    "step_cos_endpoint_implied_noise",
)


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    axes = tuple(range(2, left.ndim))
    dot = np.sum(left * right, axis=axes)
    denominator = np.sqrt(
        np.sum(np.square(left), axis=axes) * np.sum(np.square(right), axis=axes)
    )
    return np.clip(dot / np.maximum(denominator, 1e-12), -1.0, 1.0)


def summarize(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    output = []
    for value, group in sorted(grouped.items(), key=lambda item: item[0]):
        result = {key: value, "terms": len(group)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in group], dtype=np.float64)
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_mean_abs"] = float(np.mean(np.abs(values)))
            result[f"{metric}_positive_fraction"] = float(np.mean(values > 0.0))
        output.append(result)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-id", type=int, default=8)
    parser.add_argument("--variant", default="query_train_l2")
    parser.add_argument("--method", default="five_bins")
    parser.add_argument("--namespace", default="predicted_noise_endpoint_x0_original12")
    parser.add_argument("--checkpoint-crossfit-dir", type=Path, required=True)
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
        positions = np.asarray(payload["snapshot_positions"], dtype=np.int32)
        outputs = np.asarray(
            payload["all_checkpoint_predicted_noise_outputs"], dtype=np.float64
        )
        xt = np.asarray(payload["aligned_trajectory_xt"], dtype=np.float64)
        x0 = np.asarray(payload["aligned_trajectory_endpoint_x0"], dtype=np.float64)
        alphas_cumprod = np.asarray(
            payload["diffusion_alphas_cumprod"], dtype=np.float64
        )

    snapshot_count = outputs.shape[1]
    term_positions = positions[:snapshot_count]
    term_timesteps = timesteps[:snapshot_count]
    if outputs.shape[0] != 50 or xt.shape[0] != snapshot_count:
        raise ValueError(f"unexpected cached shapes: outputs={outputs.shape}, xt={xt.shape}")

    # [49, 10, ...]: the same x_t and endpoint target are used at every checkpoint.
    current = outputs[:-1]
    following = outputs[1:]
    step = following - current
    xt = xt[None, ...]
    endpoint_x0 = x0[None, None, ...]
    endpoint_direction = endpoint_x0 - xt
    abar = alphas_cumprod[term_timesteps].reshape(
        (1, snapshot_count) + (1,) * (xt.ndim - 2)
    )
    endpoint_noise = (xt - np.sqrt(abar) * endpoint_x0) / np.maximum(
        np.sqrt(1.0 - abar), 1e-12
    )

    metric_arrays = {
        "current_cos_endpoint_direction": cosine(current, endpoint_direction),
        "next_cos_endpoint_direction": cosine(following, endpoint_direction),
        "step_cos_endpoint_direction": cosine(step, endpoint_direction),
        "current_cos_endpoint_implied_noise": cosine(current, endpoint_noise),
        "next_cos_endpoint_implied_noise": cosine(following, endpoint_noise),
        "step_cos_endpoint_implied_noise": cosine(step, endpoint_noise),
    }

    rows = []
    for checkpoint in range(49):
        sign = signs[checkpoint]
        for slot in range(snapshot_count):
            row = {
                "query": args.query_id,
                "checkpoint_index": checkpoint,
                "checkpoint": checkpoint + 1,
                "epoch": 4 * (checkpoint + 1),
                "checkpoint_bin": min(checkpoint // 10 + 1, 5),
                "timestep": int(term_timesteps[slot]),
                "snapshot_position": int(term_positions[slot]),
                "majority_flip_sign": sign,
                "flip_sign_stability": stability[checkpoint],
            }
            for metric, values in metric_arrays.items():
                row[metric] = float(values[checkpoint, slot])
            rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_term.csv", rows)
    by_sign = summarize(rows, "majority_flip_sign")
    by_bin = summarize(rows, "checkpoint_bin")
    for row in rows:
        row["timestep_and_sign"] = f"{row['timestep']}:{row['majority_flip_sign']:+d}"
    by_timestep_sign = summarize(rows, "timestep_and_sign")
    write_csv(args.out_dir / "by_flip_sign.csv", by_sign)
    write_csv(args.out_dir / "by_checkpoint_bin.csv", by_bin)
    write_csv(args.out_dir / "by_timestep_and_flip_sign.csv", by_timestep_sign)

    print(f"Q{args.query_id} TRUE ENDPOINT x0 GEOMETRY; signs from {split_count} splits")
    print("BY CHECKPOINT FLIP SIGN")
    print("SIGN TERMS  CURR~(x0-xt) NEXT~(x0-xt) STEP~(x0-xt)  CURR~eps_x0 NEXT~eps_x0 STEP~eps_x0")
    for row in by_sign:
        print(
            f"{int(row['majority_flip_sign']):+4d} {int(row['terms']):5d} "
            f"{row['current_cos_endpoint_direction_mean']:+13.5f} "
            f"{row['next_cos_endpoint_direction_mean']:+13.5f} "
            f"{row['step_cos_endpoint_direction_mean']:+13.5f} "
            f"{row['current_cos_endpoint_implied_noise_mean']:+12.5f} "
            f"{row['next_cos_endpoint_implied_noise_mean']:+11.5f} "
            f"{row['step_cos_endpoint_implied_noise_mean']:+11.5f}"
        )
    print("\nBY FIVE CHECKPOINT BINS")
    print("BIN SIGN TERMS STEP~(x0-xt) STEP~eps_x0")
    for row in by_bin:
        checkpoint = min((int(row["checkpoint_bin"]) - 1) * 10, 48)
        print(
            f"{int(row['checkpoint_bin']):3d} {signs[checkpoint]:+4d} "
            f"{int(row['terms']):5d} "
            f"{row['step_cos_endpoint_direction_mean']:+13.5f} "
            f"{row['step_cos_endpoint_implied_noise_mean']:+11.5f}"
        )
    print("\nBY TIMESTAMP AND FLIP SIGN")
    print("T SIGN TERMS STEP~(x0-xt) STEP~eps_x0")
    for row in by_timestep_sign:
        timestep, sign = row["timestep_and_sign"].split(":")
        print(
            f"{int(timestep):4d} {int(sign):+4d} {int(row['terms']):5d} "
            f"{row['step_cos_endpoint_direction_mean']:+13.5f} "
            f"{row['step_cos_endpoint_implied_noise_mean']:+11.5f}"
        )
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
