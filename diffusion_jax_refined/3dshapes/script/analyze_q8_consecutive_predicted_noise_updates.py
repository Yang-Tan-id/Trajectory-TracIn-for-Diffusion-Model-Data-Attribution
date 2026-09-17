#!/usr/bin/env python3
"""Compare consecutive checkpoint predicted-noise update directions for Q8."""

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


def summarize(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    output = []
    for value, group in sorted(groups.items(), key=lambda item: str(item[0])):
        cosines = np.asarray([row["consecutive_update_cosine"] for row in group])
        current_norms = np.asarray([row["current_update_norm"] for row in group])
        next_norms = np.asarray([row["next_update_norm"] for row in group])
        output.append(
            {
                key: value,
                "terms": len(group),
                "cosine_mean": float(np.mean(cosines)),
                "cosine_mean_abs": float(np.mean(np.abs(cosines))),
                "opposite_fraction": float(np.mean(cosines < 0.0)),
                "strongly_opposite_fraction": float(np.mean(cosines < -0.25)),
                "current_update_norm_mean": float(np.mean(current_norms)),
                "next_update_norm_mean": float(np.mean(next_norms)),
                "next_over_current_norm_mean": float(
                    np.mean(next_norms / np.maximum(current_norms, 1e-12))
                ),
            }
        )
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
        outputs = np.asarray(
            payload["all_checkpoint_predicted_noise_outputs"], dtype=np.float64
        )
        timesteps = np.asarray(payload["timesteps"], dtype=np.int32)

    updates = outputs[1:] - outputs[:-1]  # 49 transitions x 10 timestamps
    first = updates[:-1]
    second = updates[1:]
    axes = tuple(range(2, updates.ndim))
    first_norm = np.sqrt(np.sum(np.square(first), axis=axes))
    second_norm = np.sqrt(np.sum(np.square(second), axis=axes))
    dot = np.sum(first * second, axis=axes)
    cosines = np.clip(
        dot / np.maximum(first_norm * second_norm, 1e-12), -1.0, 1.0
    )

    snapshot_count = outputs.shape[1]
    term_timesteps = timesteps[:snapshot_count]
    rows = []
    for checkpoint in range(48):
        current_sign = signs[checkpoint]
        next_sign = signs[checkpoint + 1]
        transition = ("+" if current_sign > 0 else "-") + (
            "+" if next_sign > 0 else "-"
        )
        for slot, timestep in enumerate(term_timesteps):
            rows.append(
                {
                    "query": args.query_id,
                    "checkpoint": checkpoint + 1,
                    "epoch": 4 * (checkpoint + 1),
                    "next_checkpoint": checkpoint + 2,
                    "next_epoch": 4 * (checkpoint + 2),
                    "checkpoint_bin": min(checkpoint // 10 + 1, 5),
                    "timestep": int(timestep),
                    "current_flip_sign": current_sign,
                    "next_flip_sign": next_sign,
                    "flip_sign_transition": transition,
                    "current_flip_stability": stability[checkpoint],
                    "next_flip_stability": stability[checkpoint + 1],
                    "consecutive_update_cosine": float(cosines[checkpoint, slot]),
                    "current_update_norm": float(first_norm[checkpoint, slot]),
                    "next_update_norm": float(second_norm[checkpoint, slot]),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_term.csv", rows)
    summaries = {
        "by_current_flip_sign.csv": summarize(rows, "current_flip_sign"),
        "by_flip_sign_transition.csv": summarize(rows, "flip_sign_transition"),
        "by_checkpoint_bin.csv": summarize(rows, "checkpoint_bin"),
        "by_timestep.csv": summarize(rows, "timestep"),
    }
    for filename, values in summaries.items():
        write_csv(args.out_dir / filename, values)

    def display(title, key, values):
        print(f"\n{title}")
        print(f"{key.upper():>10s} TERMS  COS MEAN   |COS|  OPPOSITE  COS<-0.25  NEXT/CURR NORM")
        for row in values:
            print(
                f"{str(row[key]):>10s} {int(row['terms']):5d} "
                f"{row['cosine_mean']:+9.5f} {row['cosine_mean_abs']:8.5f} "
                f"{row['opposite_fraction']:9.3f} "
                f"{row['strongly_opposite_fraction']:10.3f} "
                f"{row['next_over_current_norm_mean']:14.5f}"
            )

    print(
        f"Q{args.query_id} CONSECUTIVE PREDICTED-NOISE UPDATES; "
        f"flip signs from {split_count} splits"
    )
    display("BY CURRENT CHECKPOINT FLIP SIGN", "current_flip_sign", summaries["by_current_flip_sign.csv"])
    display("BY FLIP-SIGN TRANSITION", "flip_sign_transition", summaries["by_flip_sign_transition.csv"])
    display("BY CHECKPOINT BIN", "checkpoint_bin", summaries["by_checkpoint_bin.csv"])
    display("BY TIMESTAMP", "timestep", summaries["by_timestep.csv"])
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
