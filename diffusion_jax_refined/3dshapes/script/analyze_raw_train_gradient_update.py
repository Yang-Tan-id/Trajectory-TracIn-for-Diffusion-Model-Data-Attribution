#!/usr/bin/env python3
"""Summarize projected raw train-gradient alignment with checkpoint updates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from analyze_reference_probe_delta_alignment import artifact_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-ids", default="0")
    parser.add_argument(
        "--geometry-namespace", default="raw_train_gradient_update_sanity_reference_v2"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    query_id = int(args.query_ids.split(",")[0])
    path = artifact_path(args, query_id)
    with np.load(path, allow_pickle=False) as payload:
        ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
        timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        cosines = np.asarray(payload["raw_train_update_cosines"], dtype=np.float64)
        ratios = np.asarray(payload["raw_train_update_norm_ratios"], dtype=np.float64)
        timestamp_mean_cosines = np.asarray(
            payload["raw_train_update_timestamp_mean_cosines"], dtype=np.float64
        )
        timestamp_mean_ratios = np.asarray(
            payload["raw_train_update_timestamp_mean_norm_ratios"], dtype=np.float64
        )

    if not (len(ckpts) == len(timesteps) == len(cosines) == len(ratios) == 490):
        raise ValueError(
            f"expected 490 aligned terms, got "
            f"{len(ckpts)}, {len(timesteps)}, {len(cosines)}, {len(ratios)}"
        )

    rows = []
    for c, t, cosine, ratio in zip(ckpts, timesteps, cosines, ratios):
        rows.append(
            {
                "checkpoint": int(c) + 1,
                "epoch": 4 * (int(c) + 1),
                "timestep": int(t),
                "cosine": float(cosine),
                "projected_mean_gradient_to_delta_norm_ratio": float(ratio),
            }
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "per_checkpoint_timestamp.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("RAW TRAIN-LOSS GRADIENT SUM vs NEXT-CHECKPOINT PARAMETER UPDATE")
    print("cosine(-mean_i P[g_i,c,t], P[theta[c+1]-theta[c]])")
    print(f"overall: mean={np.mean(cosines):+.4f} median={np.median(cosines):+.4f} "
          f">0={np.mean(cosines > 0):.3f}")
    print("\nCHECKPOINT-WISE MEAN OVER THE 10 TIMESTAMPS")
    print(
        f"mean={np.mean(timestamp_mean_cosines):+.4f} "
        f"median={np.median(timestamp_mean_cosines):+.4f} "
        f">0={np.mean(timestamp_mean_cosines > 0):.3f} "
        f"median-ratio={np.median(timestamp_mean_ratios):.6g}"
    )
    checkpoint_rows = [
        {
            "checkpoint": checkpoint + 1,
            "epoch": 4 * (checkpoint + 1),
            "timestamp_mean_cosine": float(timestamp_mean_cosines[checkpoint]),
            "timestamp_mean_norm_ratio": float(timestamp_mean_ratios[checkpoint]),
        }
        for checkpoint in range(len(timestamp_mean_cosines))
    ]
    with (args.out_dir / "per_checkpoint_timestamp_mean.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(checkpoint_rows[0]))
        writer.writeheader()
        writer.writerows(checkpoint_rows)
    print(f"{'CKPT':>4s} {'EPOCH':>5s} {'COS':>9s} {'RATIO':>10s}")
    print("-" * 34)
    for row in checkpoint_rows:
        print(
            f"{row['checkpoint']:4d} {row['epoch']:5d} "
            f"{row['timestamp_mean_cosine']:+9.4f} "
            f"{row['timestamp_mean_norm_ratio']:10.4g}"
        )
    print("\nPER-TIMESTAMP COMPARISON")
    print(f"{'T':>4s} {'N':>3s} {'MEAN':>9s} {'MEDIAN':>9s} {'>0':>7s} {'RATIO':>10s}")
    print("-" * 52)
    for timestep in sorted(set(timesteps.tolist()), reverse=True):
        mask = timesteps == timestep
        print(
            f"{timestep:4d} {int(mask.sum()):3d} {np.mean(cosines[mask]):+9.4f} "
            f"{np.median(cosines[mask]):+9.4f} {np.mean(cosines[mask] > 0):7.3f} "
            f"{np.median(ratios[mask]):10.4g}"
        )
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
