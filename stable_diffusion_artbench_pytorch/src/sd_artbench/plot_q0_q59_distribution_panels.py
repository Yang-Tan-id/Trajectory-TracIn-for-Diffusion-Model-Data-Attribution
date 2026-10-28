#!/usr/bin/env python3
"""Plot Q1--Q3 distribution panels for ArtBench top-k removal metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import numpy as np


METHODS = (
    ("endpoint_pollute_query_train_l2", "PINT (Ours)", "#4C78A8"),
    ("endpoint_tracin_raw", "Endpoint TracIn", "#F2A65A"),
    ("endpoint_tracin_query_train_l2", "GAS", "#A995BD"),
    ("retrac_query_train_l2", "ReTrac", "#72B7A1"),
    ("random_full5000_seeded", "Random", "#B0B0B0"),
)

METHOD_ALIASES = {
    "endpoint_pollute_adamw_timestamp_both_l2_full10k_pos": (
        "endpoint_pollute_query_train_l2"
    ),
    "endpoint_tracin_adamw_linear_raw_full10k_neg": "endpoint_tracin_raw",
    "endpoint_tracin_adamw_linear_both_l2_full10k_neg": (
        "endpoint_tracin_query_train_l2"
    ),
    "retrac_adamw_query_train_l2_full10k_neg": "retrac_query_train_l2",
    "random_full10k_seeded": "random_full5000_seeded",
}

PERCEPTUAL_LABEL_TO_METHOD = {
    "PINT-BOTH-L2": "endpoint_pollute_query_train_l2",
    "TRACIN-RAW": "endpoint_tracin_raw",
    "TRACIN-BOTH-L2": "endpoint_tracin_query_train_l2",
    "RETRAC-BOTH-L2": "retrac_query_train_l2",
    "RANDOM": "random_full5000_seeded",
}

PERCEPTUAL_PATTERN = re.compile(
    r"^Q(?P<query>\d+)\s+"
    r"(?P<label>PINT-BOTH-L2|TRACIN-RAW|TRACIN-BOTH-L2|RETRAC-BOTH-L2|RANDOM)\s+"
    r"(?P<pixel>[+-]?\d+(?:\.\d+)?)\s+"
    r"(?P<clip>[+-]?\d+(?:\.\d+)?)\s+"
    r"(?P<lpips>[+-]?\d+(?:\.\d+)?)\s+"
    r"(?P<dino>[+-]?\d+(?:\.\d+)?)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-csv", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--perceptual-csv", type=Path)
    source.add_argument("--perceptual-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-queries", type=int, default=60)
    parser.add_argument("--dataset-title", default="ArtBench")
    parser.add_argument("--query-label", default="Q0–Q59")
    parser.add_argument("--output-prefix", default="artbench_q0_q59")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_perceptual_report(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in path.read_text(encoding="utf-8").splitlines():
        match = PERCEPTUAL_PATTERN.fullmatch(source.strip())
        if match is None:
            continue
        values = match.groupdict()
        rows.append(
            {
                "query_index": values["query"],
                "method": PERCEPTUAL_LABEL_TO_METHOD[values["label"]],
                "pixel_l2": values["pixel"],
                "clip_distance": values["clip"],
                "lpips": values["lpips"],
                "dino_distance": values["dino"],
            }
        )
    if not rows:
        raise ValueError(f"no per-query perceptual rows found in {path}")
    return rows


def values_by_method(
    rows: Iterable[dict[str, str]], metric: str, expected_queries: int
) -> dict[str, np.ndarray]:
    rows = list(rows)
    result: dict[str, np.ndarray] = {}
    for method, _, _ in METHODS:
        selected = sorted(
            (
                row
                for row in rows
                if METHOD_ALIASES.get(row["method"], row["method"]) == method
            ),
            key=lambda row: int(row.get("query_index", row.get("query_id", "-1"))),
        )
        query_ids = [
            int(row.get("query_index", row.get("query_id", "-1")))
            for row in selected
        ]
        if len(selected) != expected_queries:
            raise ValueError(
                f"{method}/{metric}: expected {expected_queries} rows, "
                f"found {len(selected)}"
            )
        if len(set(query_ids)) != expected_queries:
            raise ValueError(f"{method}/{metric}: duplicate query ids")
        source_metric = metric
        if metric == "pixel_l2" and selected and metric not in selected[0]:
            source_metric = "endpoint_rmse"
        result[method] = np.asarray(
            [float(row[source_metric]) for row in selected], dtype=np.float64
        )
    return result


def distribution(values: np.ndarray) -> tuple[float, float, float, float, float]:
    return (
        float(np.min(values)),
        float(np.quantile(values, 0.25)),
        float(np.mean(values)),
        float(np.quantile(values, 0.75)),
        float(np.max(values)),
    )


def draw_distribution_panel(
    axis: Axes,
    grouped: dict[str, np.ndarray],
    *,
    title: str,
    ylabel: str,
) -> None:
    all_values = np.concatenate(list(grouped.values()))
    maximum = float(np.max(all_values))
    upper = maximum * 1.18 if maximum > 0 else 1.0
    box_width = 0.58

    for position, (method, label, color) in enumerate(METHODS):
        minimum, q1, mean, q3, maximum_value = distribution(grouped[method])
        axis.vlines(position, minimum, maximum_value, color="#4A4D50", linewidth=1.5)
        axis.hlines(
            (minimum, maximum_value),
            position - 0.13,
            position + 0.13,
            color="#4A4D50",
            linewidth=1.5,
        )
        axis.add_patch(
            Rectangle(
                (position - box_width / 2, q1),
                box_width,
                q3 - q1,
                facecolor=color,
                edgecolor="#45484B",
                linewidth=1.25,
                alpha=0.78,
                zorder=3,
            )
        )
        axis.hlines(
            mean,
            position - box_width / 2 + 0.06,
            position + box_width / 2 - 0.06,
            color="#202124",
            linewidth=2.6,
            zorder=4,
        )
        axis.text(
            position,
            maximum_value + upper * 0.018,
            f"Mean {mean:.5f}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color="#202124",
        )
        axis.text(
            position,
            -0.115,
            f"Q1 {q1:.5f} · Q3 {q3:.5f}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=5.7,
            color="#4D5156",
            clip_on=False,
        )

    axis.set_title(title, fontsize=12.5, fontweight="semibold", pad=10)
    axis.set_ylabel(ylabel, fontsize=10)
    axis.set_ylim(0.0, upper)
    axis.set_xlim(-0.65, len(METHODS) - 0.35)
    axis.set_xticks(range(len(METHODS)))
    axis.set_xticklabels(
        [label for _, label, _ in METHODS], fontsize=8.7, fontweight="semibold"
    )
    axis.tick_params(axis="x", pad=8, length=0)
    axis.grid(axis="y", color="#E8EAED", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#5F6368")
    axis.spines["bottom"].set_color("#5F6368")


def save_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        path = stem.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[saved] {path}")


def main() -> None:
    args = parse_args()
    endpoint_rows = read_csv(args.endpoint_csv)
    if args.perceptual_csv is not None:
        perceptual_rows = read_csv(args.perceptual_csv)
    else:
        perceptual_rows = read_perceptual_report(args.perceptual_report)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": "#303236",
            "xtick.color": "#4E5155",
            "ytick.color": "#4E5155",
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(15.8, 6.2))
    draw_distribution_panel(
        axes[0],
        values_by_method(endpoint_rows, "endpoint_mse", args.expected_queries),
        title="Endpoint difference",
        ylabel="MSE",
    )
    draw_distribution_panel(
        axes[1],
        values_by_method(endpoint_rows, "trajectory_mse", args.expected_queries),
        title="Trajectory difference",
        ylabel="MSE",
    )
    figure.suptitle(
        f"{args.dataset_title} Top-1000 Removal ({args.query_label})",
        fontsize=15,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.25, wspace=0.16)
    save_figure(
        figure,
        args.output_dir / f"{args.output_prefix}_endpoint_trajectory_distribution",
    )
    plt.close(figure)

    panels = (
        ("pixel_l2", "Pixel L2", "Distance"),
        ("clip_distance", "CLIP Distance", "Cosine distance"),
        ("lpips", "LPIPS", "Distance"),
        ("dino_distance", "DINO Distance", "Cosine distance"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(15.8, 11.2))
    for axis, (metric, title, ylabel) in zip(axes.flat, panels, strict=True):
        draw_distribution_panel(
            axis,
            values_by_method(perceptual_rows, metric, args.expected_queries),
            title=title,
            ylabel=ylabel,
        )
    figure.suptitle(
        f"{args.dataset_title} Top-1000 Removal: Perceptual Metrics "
        f"({args.query_label})",
        fontsize=15,
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.89,
        bottom=0.13,
        hspace=0.39,
        wspace=0.13,
    )
    save_figure(
        figure,
        args.output_dir / f"{args.output_prefix}_perceptual_distribution_2x2",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
