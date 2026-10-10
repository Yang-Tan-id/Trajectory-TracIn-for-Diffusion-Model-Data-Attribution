#!/usr/bin/env python3
"""Plot component-level datapoint sign distributions for 24 probes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


COLORS = {"all": "#243B53", "old12": "#2F80ED", "fresh12": "#E07A5F"}


def density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return gaussian_kde(values, bw_method=0.16)(grid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    required = {"bank", "original_positive_fraction"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    positive = frame["original_positive_fraction"].to_numpy(dtype=float)
    majority = np.maximum(positive, 1.0 - positive)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))

    ax = axes[0]
    bins = np.linspace(0.0, 1.0, 41)
    ax.hist(
        positive,
        bins=bins,
        density=True,
        color="#B8C4CE",
        alpha=0.52,
        edgecolor="white",
        linewidth=0.45,
        label="All components",
    )
    grid = np.linspace(0.0, 1.0, 500)
    ax.plot(grid, density(positive, grid), color=COLORS["all"], linewidth=2.4)
    for bank in ("old12", "fresh12"):
        values = frame.loc[
            frame["bank"] == bank, "original_positive_fraction"
        ].to_numpy(dtype=float)
        ax.plot(
            grid,
            density(values, grid),
            color=COLORS[bank],
            linewidth=1.7,
            linestyle="--",
            label=bank,
        )
    ax.axvline(0.1, color="#777777", linestyle=":", linewidth=1.1)
    ax.axvline(0.5, color="#777777", linestyle=":", linewidth=1.1)
    ax.axvline(0.9, color="#777777", linestyle=":", linewidth=1.1)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Fraction of 5,000 datapoint scores that are positive")
    ax.set_ylabel("Density")
    ax.set_title("A. Positive-sign fraction")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    bins = np.linspace(0.5, 1.0, 31)
    ax.hist(
        majority,
        bins=bins,
        density=True,
        color="#98C1A5",
        alpha=0.55,
        edgecolor="white",
        linewidth=0.45,
    )
    grid = np.linspace(0.5, 1.0, 500)
    ax.plot(grid, density(majority, grid), color="#245C3C", linewidth=2.4)
    ax.axvspan(0.9, 1.0, color="#E9C46A", alpha=0.22)
    ax.axvline(0.9, color="#9A6B00", linestyle="--", linewidth=1.2)
    ax.text(
        0.945,
        ax.get_ylim()[1] * 0.91,
        "53.25%\nabove 90%",
        ha="center",
        va="top",
        color="#6F4D00",
        fontsize=9.5,
    )
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Majority-sign fraction  max(p, 1 − p)")
    ax.set_ylabel("Density")
    ax.set_title("B. Same-sign concentration")

    fig.suptitle(
        "Datapoint score-sign distributions across individual components",
        fontsize=14,
        fontweight="semibold",
        y=1.01,
    )
    fig.text(
        0.5,
        -0.01,
        "24 probes × 10 queries × 10 timestamps = 2,400 components; "
        "each component contains 5,000 datapoint scores",
        ha="center",
        fontsize=9.5,
        color="#4A5568",
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
