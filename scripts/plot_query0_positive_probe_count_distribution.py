#!/usr/bin/env python3
"""Plot positive-probe count distributions over 5,000 datapoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--query", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as payload:
        binary = np.asarray(payload["best_binary"], dtype=bool)
        query_ids = np.asarray(payload["query_ids"], dtype=np.int32)
        timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
    matches = np.flatnonzero(query_ids == args.query)
    if len(matches) != 1:
        raise ValueError(f"query {args.query} appears {len(matches)} times")
    qslot = int(matches[0])
    # Shape: (timestamp, datapoint), values are integer positive-probe counts 0..24.
    counts = binary[:, qslot].sum(axis=0)
    if counts.shape != (len(timesteps), 5000):
        raise ValueError(f"unexpected count shape: {counts.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    rows = []
    for tslot, timestep in enumerate(timesteps):
        frequencies = np.bincount(counts[tslot], minlength=25)
        for positive_count, frequency in enumerate(frequencies):
            rows.append(
                {
                    "query": args.query,
                    "timestep": int(timestep),
                    "positive_probe_count": positive_count,
                    "datapoint_count": int(frequency),
                    "datapoint_fraction": float(frequency / 5000.0),
                }
            )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 5, figsize=(17.5, 8.5), sharex=True, sharey=True)
    x = np.arange(25)
    for axis, timestep, values in zip(axes.flat, timesteps, counts):
        frequencies = np.bincount(values, minlength=25) / 5000.0
        axis.bar(x, 100.0 * frequencies, width=0.86, color="#4C78A8", alpha=0.88)
        mean = float(values.mean())
        median = float(np.median(values))
        axis.axvline(12, color="#666666", linestyle=":", linewidth=1.2)
        axis.axvline(mean, color="#D1495B", linewidth=1.7)
        axis.set_title(f"t={int(timestep)}  mean={mean:.2f}, median={median:.0f}")
        axis.set_xlim(-0.5, 24.5)
        axis.set_xticks([0, 4, 8, 12, 16, 20, 24])
        axis.grid(axis="y", color="#D9E2EC", linewidth=0.65, alpha=0.65)
    for axis in axes[1]:
        axis.set_xlabel("Positive probes out of 24")
    for axis in axes[:, 0]:
        axis.set_ylabel("Datapoints (%)")
    fig.suptitle(
        f"Query {args.query}: positive-probe count after best-LDS timestamp flips (5,000 datapoints)",
        fontsize=14.5,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(args.output)
    print(csv_path)


if __name__ == "__main__":
    main()
