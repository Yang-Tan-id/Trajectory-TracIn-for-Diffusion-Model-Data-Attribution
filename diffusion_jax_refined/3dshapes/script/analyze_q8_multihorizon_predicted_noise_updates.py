#!/usr/bin/env python3
"""Measure whether 2/4-checkpoint deltas suppress one-checkpoint zig-zag."""

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


def cosine_and_norm(left, right):
    axes = tuple(range(2, left.ndim))
    left_norm = np.sqrt(np.sum(np.square(left), axis=axes))
    right_norm = np.sqrt(np.sum(np.square(right), axis=axes))
    dot = np.sum(left * right, axis=axes)
    cosine = np.clip(dot / np.maximum(left_norm * right_norm, 1e-12), -1.0, 1.0)
    return cosine, left_norm, right_norm


def summarize(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for values, group in sorted(groups.items()):
        cosines = np.asarray([row["successive_window_cosine"] for row in group])
        first_norm = np.asarray([row["first_delta_norm"] for row in group])
        second_norm = np.asarray([row["second_delta_norm"] for row in group])
        result = {key: value for key, value in zip(keys, values)}
        result.update(
            terms=len(group),
            cosine_mean=float(np.mean(cosines)),
            cosine_mean_abs=float(np.mean(np.abs(cosines))),
            positive_fraction=float(np.mean(cosines > 0.0)),
            opposite_fraction=float(np.mean(cosines < 0.0)),
            strongly_opposite_fraction=float(np.mean(cosines < -0.25)),
            first_delta_norm_mean=float(np.mean(first_norm)),
            second_delta_norm_mean=float(np.mean(second_norm)),
            second_over_first_norm_mean=float(
                np.mean(second_norm / np.maximum(first_norm, 1e-12))
            ),
        )
        output.append(result)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-id", type=int, default=8)
    parser.add_argument("--namespace", default="predicted_noise_endpoint_x0_original12")
    parser.add_argument("--horizons", default="1,2,4")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    path = artifact_path(
        args.experiment, args.train_seed, args.epochs, args.query_id, args.namespace
    )
    with np.load(path, allow_pickle=False) as payload:
        outputs = np.asarray(
            payload["all_checkpoint_predicted_noise_outputs"], dtype=np.float64
        )
        timesteps = np.asarray(payload["timesteps"], dtype=np.int32)

    snapshot_count = outputs.shape[1]
    term_timesteps = timesteps[:snapshot_count]
    rows = []
    for horizon in [int(value) for value in args.horizons.split(",")]:
        # Compare adjacent non-overlapping windows:
        # eps[c+h]-eps[c] versus eps[c+2h]-eps[c+h].
        first = outputs[horizon:-horizon] - outputs[:-2 * horizon]
        second = outputs[2 * horizon:] - outputs[horizon:-horizon]
        cosines, first_norm, second_norm = cosine_and_norm(first, second)
        for checkpoint in range(first.shape[0]):
            for slot, timestep in enumerate(term_timesteps):
                rows.append(
                    {
                        "query": args.query_id,
                        "horizon": horizon,
                        "start_checkpoint": checkpoint + 1,
                        "middle_checkpoint": checkpoint + horizon + 1,
                        "end_checkpoint": checkpoint + 2 * horizon + 1,
                        "start_epoch": 4 * (checkpoint + 1),
                        "middle_epoch": 4 * (checkpoint + horizon + 1),
                        "end_epoch": 4 * (checkpoint + 2 * horizon + 1),
                        "timestep": int(timestep),
                        "successive_window_cosine": float(cosines[checkpoint, slot]),
                        "first_delta_norm": float(first_norm[checkpoint, slot]),
                        "second_delta_norm": float(second_norm[checkpoint, slot]),
                    }
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_term.csv", rows)
    overall = summarize(rows, ["horizon"])
    by_timestep = summarize(rows, ["horizon", "timestep"])
    write_csv(args.out_dir / "by_horizon.csv", overall)
    write_csv(args.out_dir / "by_horizon_and_timestep.csv", by_timestep)

    print("Q8 MULTI-HORIZON PREDICTED-NOISE UPDATE STABILITY")
    print("H TERMS  COS MEAN   |COS|   COS+  OPPOSITE COS<-0.25  NEXT/CURR NORM")
    for row in overall:
        print(
            f"{int(row['horizon']):1d} {int(row['terms']):5d} "
            f"{row['cosine_mean']:+9.5f} {row['cosine_mean_abs']:8.5f} "
            f"{row['positive_fraction']:6.3f} {row['opposite_fraction']:9.3f} "
            f"{row['strongly_opposite_fraction']:9.3f} "
            f"{row['second_over_first_norm_mean']:14.5f}"
        )
    print("\nBY TIMESTAMP")
    print("H    T TERMS  COS MEAN   |COS|   COS+  OPPOSITE")
    for row in by_timestep:
        print(
            f"{int(row['horizon']):1d} {int(row['timestep']):4d} "
            f"{int(row['terms']):5d} {row['cosine_mean']:+9.5f} "
            f"{row['cosine_mean_abs']:8.5f} {row['positive_fraction']:6.3f} "
            f"{row['opposite_fraction']:9.3f}"
        )
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
