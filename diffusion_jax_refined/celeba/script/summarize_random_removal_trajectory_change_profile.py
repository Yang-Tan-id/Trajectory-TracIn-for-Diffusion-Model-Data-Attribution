#!/usr/bin/env python3
"""Summarize per-timestep CelebA random-removal trajectory changes."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def cumulative_step(profile: np.ndarray, threshold: float) -> int:
    cumulative = np.cumsum(profile) / np.sum(profile)
    return int(np.searchsorted(cumulative, threshold, side="left"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    lookup = {}
    timesteps = {}
    for path in sorted(args.root.glob("shard_*/trajectory_state_mse_per_timestep.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (
                    int(row["evaluation_query_id"]),
                    int(row["random_model_id"]),
                    int(row["step"]),
                )
                lookup[key] = float(row["trajectory_state_mse"])
                timesteps[key[2]] = int(row["timestep"])
    if not lookup:
        raise SystemExit(f"no shard CSV files under {args.root}")

    pairs = sorted({(query, model) for query, model, _ in lookup})
    missing = [pair for pair in pairs if any((*pair, step) not in lookup for step in range(50))]
    if missing:
        raise RuntimeError(f"incomplete pairs: {missing}")
    matrix = np.asarray([
        [lookup[(query, model, step)] for step in range(50)]
        for query, model in pairs
    ], dtype=np.float64)
    query_ids = sorted({query for query, _ in pairs})

    raw_mean = matrix.mean(axis=0)
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    equal_pair_share = normalized.mean(axis=0)
    peak_steps = np.argmax(matrix, axis=1)
    peak_counts = np.bincount(peak_steps, minlength=50)

    lines = [
        "CELEBA RANDOM TOP-1000 — TRAJECTORY CHANGE CONCENTRATION",
        "=" * 104,
        f"queries={len(query_ids)} pairs={len(pairs)} steps=50",
        "RAW SHARE weights high-change pairs more; EQUAL-PAIR SHARE normalizes every pair first.",
        "",
        f"{'STEP':>5}{'TIME':>7}{'MEAN MSE':>15}{'RAW SHARE':>13}"
        f"{'PAIR SHARE':>13}{'PEAK PAIRS':>13}{'CUM SHARE':>13}",
        "-" * 104,
    ]
    raw_share = raw_mean / raw_mean.sum()
    cumulative = np.cumsum(equal_pair_share)
    for step in range(50):
        lines.append(
            f"{step:>5}{timesteps[step]:>7}{raw_mean[step]:>15.8g}"
            f"{raw_share[step]:>13.4%}{equal_pair_share[step]:>13.4%}"
            f"{peak_counts[step]:>10}/{len(pairs):<2}{cumulative[step]:>13.4%}"
        )

    lines.extend(["", "FIVE 10-STEP SEGMENTS", "-" * 104])
    for start in range(0, 50, 10):
        stop = start + 9
        lines.append(
            f"steps {start:02d}-{stop:02d}  "
            f"t={timesteps[start]:4d}..{timesteps[stop]:4d}  "
            f"raw_share={raw_share[start:stop + 1].sum():7.3%}  "
            f"equal_pair_share={equal_pair_share[start:stop + 1].sum():7.3%}"
        )

    lines.extend(["", "CUMULATIVE CHANGE FROM THE NOISY START", "-" * 104])
    for threshold in (0.25, 0.50, 0.75, 0.90):
        step = cumulative_step(equal_pair_share, threshold)
        lines.append(
            f"{threshold:>5.0%} reached at step={step:02d}, timestep={timesteps[step]}"
        )

    lines.extend(["", "PER-QUERY FIVE-SEGMENT EQUAL-PAIR SHARES", "-" * 104])
    by_query = defaultdict(list)
    for index, (query, _model) in enumerate(pairs):
        by_query[query].append(normalized[index])
    for query in query_ids:
        profile = np.mean(by_query[query], axis=0)
        parts = [profile[start:start + 10].sum() for start in range(0, 50, 10)]
        lines.append(
            f"Q{query}: " + " ".join(
                f"S{slot}={value:7.3%}" for slot, value in enumerate(parts)
            )
        )

    report = "\n".join(lines) + "\n"
    output = args.root / "trajectory_change_concentration_report.txt"
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
