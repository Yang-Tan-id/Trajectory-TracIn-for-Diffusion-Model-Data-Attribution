#!/usr/bin/env python3
"""Summarize AdamW history-only direction against adjacent checkpoint deltas."""

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
        "--geometry-namespace", default="optimizer_history_update_sanity_reference"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    path = artifact_path(args, int(args.query_ids.split(",")[0]))
    with np.load(path, allow_pickle=False) as payload:
        cosines = np.asarray(
            payload["optimizer_history_update_cosines"], dtype=np.float64
        )
        ratios = np.asarray(
            payload["optimizer_history_update_norm_ratios"], dtype=np.float64
        )
    if len(cosines) != 49 or len(ratios) != 49:
        raise ValueError(f"expected 49 checkpoint intervals, got {len(cosines)}")

    rows = [
        {
            "checkpoint": index + 1,
            "epoch": 4 * (index + 1),
            "cosine": float(cosines[index]),
            "one_step_to_four_epoch_norm_ratio": float(ratios[index]),
        }
        for index in range(49)
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "per_checkpoint.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("ADAMW CHECKPOINT HISTORY-ONLY UPDATE vs NEXT-CHECKPOINT DELTA")
    print("update = optimizer.update(zero_gradient, checkpoint.opt_state, params)")
    print(
        f"mean={np.mean(cosines):+.4f} median={np.median(cosines):+.4f} "
        f">0={np.mean(cosines > 0):.3f} >.5={np.mean(cosines > .5):.3f}"
    )
    print(f"{'CKPT':>4s} {'EPOCH':>5s} {'COS':>9s} {'RATIO':>11s}")
    print("-" * 35)
    for row in rows:
        print(
            f"{row['checkpoint']:4d} {row['epoch']:5d} {row['cosine']:+9.4f} "
            f"{row['one_step_to_four_epoch_norm_ratio']:11.5g}"
        )
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
