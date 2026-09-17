#!/usr/bin/env python3
"""Relate Q8 checkpoint signs to current/next/reference noise geometry."""

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


METRICS = (
    "cosine_current_next",
    "cosine_current_reference",
    "cosine_next_reference",
    "cosine_current_endpoint_direction",
    "cosine_next_endpoint_direction",
    "cosine_step_endpoint_direction",
    "current_to_reference_l2",
    "next_to_reference_l2",
    "next_minus_current_reference_l2",
)


def parse_signs(text: str) -> dict[int, int]:
    result = {}
    for item in text.split(","):
        checkpoint, sign = item.split(":", 1)
        result[int(checkpoint)] = 1 if sign.strip() == "+" else -1
    return result


def majority_checkpoint_signs(path: Path, query: int, variant: str, method: str):
    counts: dict[int, list[int]] = defaultdict(list)
    with path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["query"]) == query
            and row["variant"] == variant
            and row["method"] == method
        ]
    if not rows:
        raise ValueError(f"no Q{query} {variant} {method} rows in {path}")
    if "selected_signs" not in rows[0]:
        raise ValueError(f"{path} does not contain selected_signs")
    for row in rows:
        for checkpoint, sign in parse_signs(row["selected_signs"]).items():
            counts[checkpoint].append(sign)
    signs = {
        checkpoint: 1 if np.mean(values) >= 0.0 else -1
        for checkpoint, values in counts.items()
    }
    stability = {
        checkpoint: float(max(np.mean(np.asarray(values) > 0), np.mean(np.asarray(values) < 0)))
        for checkpoint, values in counts.items()
    }
    return signs, stability, len(rows)


def safe_cosine_from_lengths(left, right, difference):
    denominator = 2.0 * left * right
    return np.clip(
        (left * left + right * right - difference * difference)
        / np.maximum(denominator, 1e-12),
        -1.0,
        1.0,
    )


def summarize(rows, group_key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)
    result = []
    for group, values in sorted(groups.items(), key=lambda item: item[0]):
        output = {group_key: group, "terms": len(values)}
        for metric in METRICS:
            array = np.asarray([row[metric] for row in values], dtype=np.float64)
            output[f"{metric}_mean"] = float(array.mean())
            output[f"{metric}_mean_abs"] = float(np.abs(array).mean())
            output[f"{metric}_positive_fraction"] = float(np.mean(array > 0.0))
        result.append(output)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-id", type=int, default=8)
    parser.add_argument("--variant", default="query_train_l2")
    parser.add_argument("--method", default="five_bins")
    parser.add_argument(
        "--namespace", default="predicted_noise_output_reference_delta_original12"
    )
    parser.add_argument("--checkpoint-crossfit-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    signs, stability, split_count = majority_checkpoint_signs(
        args.checkpoint_crossfit_dir / "per_split.csv",
        args.query_id,
        args.variant,
        args.method,
    )
    path = artifact_path(
        args.experiment,
        args.train_seed,
        args.epochs,
        args.query_id,
        args.namespace,
    )
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as payload:
        checkpoints = np.asarray(payload["ckpt_indices"], dtype=np.int32)
        timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        current_norm = np.asarray(payload["predicted_noise_norms"], dtype=np.float64)
        next_norm = np.asarray(
            payload["next_predicted_noise_norms"], dtype=np.float64
        )
        next_delta_norm = np.asarray(
            payload["next_checkpoint_delta_norms"], dtype=np.float64
        )
        current_reference_cosine = np.asarray(
            payload["current_to_reference_predicted_noise_cosines"],
            dtype=np.float64,
        )
        next_reference_cosine = np.asarray(
            payload["next_to_reference_predicted_noise_cosines"],
            dtype=np.float64,
        )
        current_reference_l2 = np.asarray(
            payload["current_to_reference_predicted_noise_l2"], dtype=np.float64
        )
        next_reference_l2 = np.asarray(
            payload["next_to_reference_predicted_noise_l2"], dtype=np.float64
        )

    cosine_current_next = safe_cosine_from_lengths(
        current_norm, next_norm, next_delta_norm
    )
    endpoint_norm = np.sqrt(
        np.maximum(2.0 - 2.0 * current_reference_cosine, 1e-12)
    )
    cosine_current_endpoint = (
        current_reference_cosine - 1.0
    ) / endpoint_norm
    cosine_next_endpoint = (
        next_reference_cosine - cosine_current_next
    ) / endpoint_norm
    step_norm = np.sqrt(np.maximum(2.0 - 2.0 * cosine_current_next, 1e-12))
    cosine_step_endpoint = (
        next_reference_cosine
        - current_reference_cosine
        - cosine_current_next
        + 1.0
    ) / np.maximum(step_norm * endpoint_norm, 1e-12)
    cosine_next_endpoint = np.clip(cosine_next_endpoint, -1.0, 1.0)
    cosine_step_endpoint = np.clip(cosine_step_endpoint, -1.0, 1.0)

    rows = []
    for term, (checkpoint, timestep) in enumerate(zip(checkpoints, timesteps)):
        checkpoint = int(checkpoint)
        if checkpoint not in signs:
            continue
        display_checkpoint = checkpoint + 1
        rows.append(
            {
                "query": args.query_id,
                "checkpoint_index": checkpoint,
                "checkpoint": display_checkpoint,
                "epoch": 4 * display_checkpoint,
                "checkpoint_bin": min(checkpoint // 10 + 1, 5),
                "timestep": int(timestep),
                "majority_flip_sign": signs[checkpoint],
                "flip_sign_stability": stability[checkpoint],
                "cosine_current_next": float(cosine_current_next[term]),
                "cosine_current_reference": float(
                    current_reference_cosine[term]
                ),
                "cosine_next_reference": float(next_reference_cosine[term]),
                "cosine_current_endpoint_direction": float(
                    cosine_current_endpoint[term]
                ),
                "cosine_next_endpoint_direction": float(cosine_next_endpoint[term]),
                "cosine_step_endpoint_direction": float(cosine_step_endpoint[term]),
                "current_to_reference_l2": float(current_reference_l2[term]),
                "next_to_reference_l2": float(next_reference_l2[term]),
                "next_minus_current_reference_l2": float(
                    next_reference_l2[term] - current_reference_l2[term]
                ),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_term.csv", rows)
    by_sign = summarize(rows, "majority_flip_sign")
    by_bin = summarize(rows, "checkpoint_bin")
    by_timestep = summarize(rows, "timestep")
    for row in rows:
        row["timestep_flip_sign"] = (
            f"{int(row['timestep'])}:{int(row['majority_flip_sign']):+d}"
        )
    by_timestep_and_sign = summarize(rows, "timestep_flip_sign")
    write_csv(args.out_dir / "by_flip_sign.csv", by_sign)
    write_csv(args.out_dir / "by_checkpoint_bin.csv", by_bin)
    write_csv(args.out_dir / "by_timestep.csv", by_timestep)
    write_csv(
        args.out_dir / "by_timestep_and_flip_sign.csv", by_timestep_and_sign
    )

    print(
        f"Q{args.query_id} {args.variant} {args.method}; "
        f"majority signs from {split_count} splits"
    )
    print("BY CHECKPOINT FLIP SIGN")
    print("SIGN TERMS CURR~NEXT NEXT~END STEP~END DELTA_REF_L2")
    for row in by_sign:
        print(
            f"{int(row['majority_flip_sign']):+4d} "
            f"{int(row['terms']):5d} "
            f"{row['cosine_current_next_mean']:+9.5f} "
            f"{row['cosine_next_endpoint_direction_mean']:+9.5f} "
            f"{row['cosine_step_endpoint_direction_mean']:+9.5f} "
            f"{row['next_minus_current_reference_l2_mean']:+12.6f}"
        )
    print("\nBY FIVE CHECKPOINT BINS")
    print("BIN SIGN TERMS CURR~NEXT NEXT~END STEP~END DELTA_REF_L2")
    for row in by_bin:
        checkpoint_bin = int(row["checkpoint_bin"])
        bin_sign = signs[min((checkpoint_bin - 1) * 10, 48)]
        print(
            f"{checkpoint_bin:3d} {bin_sign:+4d} "
            f"{int(row['terms']):5d} "
            f"{row['cosine_current_next_mean']:+9.5f} "
            f"{row['cosine_next_endpoint_direction_mean']:+9.5f} "
            f"{row['cosine_step_endpoint_direction_mean']:+9.5f} "
            f"{row['next_minus_current_reference_l2_mean']:+12.6f}"
        )
    print("\nBY TIMESTAMP AND CHECKPOINT FLIP SIGN")
    print("T SIGN TERMS CURR~REF NEXT~REF STEP~END DELTA_REF_L2")
    for row in by_timestep_and_sign:
        timestep, flip_sign = row["timestep_flip_sign"].split(":")
        print(
            f"{int(timestep):4d} {int(flip_sign):+4d} "
            f"{int(row['terms']):5d} "
            f"{row['cosine_current_reference_mean']:+9.5f} "
            f"{row['cosine_next_reference_mean']:+9.5f} "
            f"{row['cosine_step_endpoint_direction_mean']:+9.5f} "
            f"{row['next_minus_current_reference_l2_mean']:+12.6f}"
        )
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
