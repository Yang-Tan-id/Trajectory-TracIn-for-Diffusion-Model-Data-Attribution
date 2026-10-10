#!/usr/bin/env python3
"""Plot 24-probe component-score signs for 20 datapoints and 10 timestamps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--query", type=int, default=0)
    parser.add_argument("--num-datapoints", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as payload:
        binary = np.asarray(payload["best_binary"], dtype=bool)
        query_ids = np.asarray(payload["query_ids"], dtype=np.int32)
        timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        score_indices = np.asarray(payload["score_indices"], dtype=np.int64)
    matches = np.flatnonzero(query_ids == args.query)
    if len(matches) != 1:
        raise ValueError(f"query {args.query} appears {len(matches)} times")
    qslot = int(matches[0])
    n = min(args.num_datapoints, len(score_indices))
    chosen_indices = score_indices[:n]
    # Stored as (probe, query, timestamp, datapoint); display as (timestamp, datapoint, probe).
    signs = np.transpose(binary[:, qslot, :, :n], (1, 2, 0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    rows = []
    for tslot, timestep in enumerate(timesteps):
        for dslot, score_index in enumerate(chosen_indices):
            values = signs[tslot, dslot]
            positive_count = int(values.sum())
            row = {
                "query": args.query,
                "score_index": int(score_index),
                "timestep": int(timestep),
                "positive_probe_count": positive_count,
                "negative_probe_count": 24 - positive_count,
                "consensus_probe_count": max(positive_count, 24 - positive_count),
                "consensus_sign": "+" if positive_count > 12 else "-" if positive_count < 12 else "tie",
            }
            row.update(
                {f"probe_{probe + 1}_sign": "+" if value else "-" for probe, value in enumerate(values)}
            )
            rows.append(row)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10.5,
        }
    )
    fig, axes = plt.subplots(2, 5, figsize=(18, 10), constrained_layout=True)
    cmap = ListedColormap(["#D96C75", "#3F88C5"])
    for tslot, (axis, timestep) in enumerate(zip(axes.flat, timesteps)):
        panel = signs[tslot].astype(int)
        axis.imshow(panel, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        axis.set_title(
            f"t={int(timestep)}  |  mean positive probes={panel.sum(axis=1).mean():.2f}/24"
        )
        axis.set_xticks(np.arange(24))
        axis.set_xticklabels(np.arange(1, 25), fontsize=6.5)
        axis.set_yticks(np.arange(n))
        axis.set_yticklabels(chosen_indices, fontsize=6.5)
        axis.set_xlabel("Probe")
        axis.set_ylabel("Datapoint score_index")
        axis.axvline(11.5, color="black", linewidth=1.4)
        axis.set_xticks(np.arange(-0.5, 24, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, n, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.22, alpha=0.65)
        axis.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle(
        f"Query {args.query}: component-score signs for 20 datapoints across 24 probes",
        fontsize=15,
        fontweight="semibold",
    )
    fig.legend(
        handles=[
            Patch(facecolor="#3F88C5", label="positive score"),
            Patch(facecolor="#D96C75", label="negative score"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(args.output)
    print(csv_path)


if __name__ == "__main__":
    main()
