#!/usr/bin/env python3
"""Plot per-query, per-probe proper-score sign and location summaries."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT = Path("analysis_outputs/probe24_proper_score_signs/per_probe_query_proper_score_summary.csv")
OUTDIR = Path("analysis_outputs/probe24_proper_score_signs")


def main() -> None:
    with INPUT.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    positive = np.full((10, 24), np.nan)
    mean = np.full((10, 24), np.nan)
    standardized_mean = np.full((10, 24), np.nan)
    for row in rows:
        q = int(row["query"])
        r = int(row["global_probe"]) - 1
        positive[q, r] = float(row["positive_fraction"])
        mean[q, r] = float(row["mean"])
        standardized_mean[q, r] = mean[q, r] / float(row["std"])

    fig, ax = plt.subplots(figsize=(18, 7), constrained_layout=True)
    image = ax.imshow(positive, vmin=0, vmax=1, cmap="RdBu", aspect="auto")
    for q in range(10):
        for r in range(24):
            value = positive[q, r]
            color = "white" if value < 0.18 or value > 0.82 else "black"
            ax.text(r, q, f"{100 * value:.0f}", ha="center", va="center", fontsize=8, color=color)
    ax.axvline(11.5, color="black", linewidth=2)
    ax.set_xticks(range(24), [str(i) for i in range(1, 25)])
    ax.set_yticks(range(10), [f"Q{i}" for i in range(10)])
    ax.set_xlabel("global probe (1–12 old bank, 13–24 fresh bank)")
    ax.set_ylabel("query")
    ax.set_title("Positive fraction among 5000 proper scores after best timestamp flips (%)")
    colorbar = fig.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label("positive fraction")
    fig.savefig(OUTDIR / "per_probe_proper_score_positive_fraction_heatmap.png", dpi=200)
    fig.savefig(OUTDIR / "per_probe_proper_score_positive_fraction_heatmap.svg")
    plt.close(fig)

    max_abs = float(np.max(np.abs(standardized_mean)))
    fig, ax = plt.subplots(figsize=(18, 7), constrained_layout=True)
    image = ax.imshow(
        standardized_mean,
        vmin=-max_abs,
        vmax=max_abs,
        cmap="RdBu_r",
        aspect="auto",
    )
    for q in range(10):
        for r in range(24):
            value = standardized_mean[q, r]
            color = "white" if abs(value) > 0.65 * max_abs else "black"
            ax.text(r, q, f"{value:+.1f}", ha="center", va="center", fontsize=8, color=color)
    ax.axvline(11.5, color="black", linewidth=2)
    ax.set_xticks(range(24), [str(i) for i in range(1, 25)])
    ax.set_yticks(range(10), [f"Q{i}" for i in range(10)])
    ax.set_xlabel("global probe (1–12 old bank, 13–24 fresh bank)")
    ax.set_ylabel("query")
    ax.set_title("Proper-score mean divided by datapoint SD after best timestamp flips")
    colorbar = fig.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label("mean / SD")
    fig.savefig(OUTDIR / "per_probe_proper_score_standardized_mean_heatmap.png", dpi=200)
    fig.savefig(OUTDIR / "per_probe_proper_score_standardized_mean_heatmap.svg")
    plt.close(fig)

    print("Q  " + " ".join(f"P{r:02d}" for r in range(1, 25)))
    print("-" * 130)
    for q in range(10):
        print(f"{q}  " + " ".join(f"{100 * positive[q, r]:4.0f}" for r in range(24)))
    print(f"[saved] {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
