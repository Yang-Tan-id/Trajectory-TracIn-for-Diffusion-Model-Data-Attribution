#!/usr/bin/env python3
"""Evaluate every nonempty subset of a saved per-probe score run."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_predicted_noise_probe8_choose4 import (
    SHAPES_ROOT,
    TARGETS,
    VARIANTS,
    cache_group,
    load_probe_scores,
    load_target_data,
    spearman,
    write_csv,
)


def all_probe_subsets(num_probes: int = 8) -> list[tuple[int, ...]]:
    return [
        combination
        for subset_size in range(1, num_probes + 1)
        for combination in itertools.combinations(range(num_probes), subset_size)
    ]


def standard_deviation(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def save_plot(
    path: Path,
    distribution_rows: list[dict[str, object]],
    num_probes: int,
    score_label: str = "linear",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warning] matplotlib unavailable; skipping plot: {exc}")
        return

    lookup = {
        (int(row["subset_size"]), str(row["target"]), str(row["variant"])): row
        for row in distribution_rows
    }
    subset_sizes = np.arange(1, num_probes + 1)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    styles = (
        ("endpoint_contarfactual", "Endpoint counterfactual", "#1565c0", "o"),
        ("traj_contarfactual", "Trajectory counterfactual", "#c62828", "s"),
    )
    for target, label, color, marker in styles:
        means = [
            float(lookup[(size, target, "query_train_l2")]["mean_lds_percent"])
            for size in subset_sizes
        ]
        stds = [
            float(lookup[(size, target, "query_train_l2")]["std_lds_percent"])
            for size in subset_sizes
        ]
        ax.errorbar(
            subset_sizes,
            means,
            yerr=stds,
            marker=marker,
            linewidth=2,
            capsize=4,
            color=color,
            label=label,
        )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xticks(subset_sizes)
    ax.set_xlabel("Number of probes in subset (k)")
    ax.set_ylabel("10-query mean LDS (%)")
    ax.set_title(f"{num_probes}-probe {score_label} score: all subsets, Both-L2")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def save_all_points_plot(
    path: Path,
    combination_rows: list[dict[str, object]],
    num_probes: int,
    score_label: str,
) -> None:
    """Plot every subset LDS and overlay Q1/median/mean/Q3 summaries."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        print(f"[warning] matplotlib unavailable; skipping all-points plot: {exc}")
        return

    target_labels = {
        "endpoint_contarfactual": "Endpoint counterfactual",
        "traj_contarfactual": "Trajectory counterfactual",
        "simple_loss": "Simple loss",
        "noise_trajectory": "Noise trajectory",
    }
    variant_labels = {
        "raw": "Raw",
        "query_l2": "Query-L2",
        "train_l2": "Train-L2",
        "query_train_l2": "Both-L2",
    }
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in combination_rows:
        grouped[(str(row["target"]), str(row["variant"]), int(row["subset_size"]))].append(
            float(row["mean_lds_percent"])
        )

    fig, axes = plt.subplots(
        len(TARGETS), len(VARIANTS), figsize=(18.5, 13.5), sharex=True
    )
    point_color = "#3977b5"
    quartile_color = "#111111"
    mean_color = "#d62728"
    for row_index, target in enumerate(TARGETS):
        for column_index, variant in enumerate(VARIANTS):
            ax = axes[row_index, column_index]
            for subset_size in range(1, num_probes + 1):
                values = np.asarray(grouped[(target, variant, subset_size)], dtype=np.float64)
                # Deterministic spreading makes every combination visible and guarantees,
                # for example, that k=11 visibly contains all twelve subsets.
                if values.size == 1:
                    x_values = np.asarray([float(subset_size)])
                else:
                    order = np.argsort(values, kind="stable")
                    offsets = np.linspace(-0.32, 0.32, values.size)
                    x_values = np.empty(values.size, dtype=np.float64)
                    x_values[order] = subset_size + offsets
                ax.scatter(
                    x_values,
                    values,
                    s=8 if values.size > 300 else 13,
                    color=point_color,
                    alpha=0.28 if values.size > 300 else 0.48,
                    linewidths=0,
                    rasterized=True,
                    zorder=1,
                )
                q1, median, q3 = np.percentile(values, [25.0, 50.0, 75.0])
                mean = float(np.mean(values))
                ax.vlines(subset_size, q1, q3, color=quartile_color, linewidth=2.0, zorder=3)
                ax.hlines(
                    [q1, q3], subset_size - 0.11, subset_size + 0.11,
                    color=quartile_color, linewidth=1.5, zorder=3,
                )
                ax.scatter(
                    [subset_size], [median], marker="_", s=95,
                    color=quartile_color, linewidths=2.2, zorder=4,
                )
                ax.scatter(
                    [subset_size], [mean], marker="D", s=29,
                    color=mean_color, edgecolors="white", linewidths=0.45, zorder=5,
                )
            ax.axhline(0.0, color="#777777", linewidth=0.7, alpha=0.7, zorder=0)
            ax.grid(axis="y", alpha=0.16)
            ax.set_xticks(range(1, num_probes + 1))
            if row_index == 0:
                ax.set_title(variant_labels.get(variant, variant))
            if column_index == 0:
                ax.set_ylabel(f"{target_labels.get(target, target)}\n10-query mean LDS (%)")
            if row_index == len(TARGETS) - 1:
                ax.set_xlabel("Subset size k")

    legend = (
        Line2D([], [], marker="o", linestyle="none", markersize=5, color=point_color,
               alpha=0.55, label="Every probe subset"),
        Line2D([], [], marker="D", linestyle="none", markersize=6, color=mean_color,
               label="Mean"),
        Line2D([], [], color=quartile_color, linewidth=2, label="Q1–Q3; center tick = median"),
    )
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(
        f"All {2 ** num_probes - 1:,} nonempty subsets of {num_probes} probes — {score_label}",
        y=0.999,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"), dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--num-probes", type=int, default=8)
    parser.add_argument("--prediction-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument(
        "--score-namespace",
        default="",
        help=(
            "Stream-score namespace containing per-probe shard arrays. Defaults to "
            "the historical final-post-square staging namespace."
        ),
    )
    parser.add_argument(
        "--analysis-label",
        default="",
        help="Output directory label; defaults to probeN_all_subset_sizes_linear.",
    )
    parser.add_argument("--score-label", default="linear")
    parser.add_argument(
        "--expected-terms",
        type=int,
        default=500,
        help="Expected accumulated checkpoint/timestamp terms across score shards.",
    )
    args = parser.parse_args()

    result_root = SHAPES_ROOT / "result" / args.experiment
    if args.num_probes <= 0:
        raise ValueError("--num-probes must be positive")
    score_namespace = args.score_namespace or (
        f"traj_tracin_predicted_noise_jvp_final_post_square_probe{args.num_probes}"
    )
    shard_dir = (
        result_root
        / "stream_score"
        / score_namespace
        / f"train_seed_{args.train_seed}"
        / f"run_{args.run_id}"
        / "shards"
    )
    probe_scores, score_indices = load_probe_scores(
        shard_dir,
        args.num_probes,
        expected_terms=args.expected_terms,
    )
    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    subsets = all_probe_subsets(args.num_probes)
    per_query_rows: list[dict[str, object]] = []

    for query_id, record in enumerate(records):
        prompt = str(record["prompt"])
        prompt_tag = prompt.replace(",", "_")
        eval_root = (
            result_root
            / "eval"
            / "prompted_solo"
            / f"query_{prompt_tag}"
            / f"initial_seed_{int(record['initial_seed'])}"
        )
        incidence, true_values = load_target_data(cache_group(eval_root), score_indices)
        for variant, values in probe_scores.items():
            per_probe_predictions = (
                args.prediction_sign * values[:, query_id, :] @ incidence.T
            )
            for subset in subsets:
                prediction = per_probe_predictions[list(subset)].mean(axis=0)
                subset_tag = ",".join(str(index + 1) for index in subset)
                for target in TARGETS:
                    per_query_rows.append(
                        {
                            "subset_size": len(subset),
                            "combination": subset_tag,
                            "query": query_id,
                            "target": target,
                            "variant": variant,
                            "lds_percent": 100.0 * spearman(prediction, true_values[target]),
                            "prompt": prompt_tag,
                        }
                    )
        print(
            f"[query {query_id}/9] evaluated {len(subsets)} nonempty subsets",
            flush=True,
        )

    grouped: dict[tuple[int, str, str, str], list[float]] = defaultdict(list)
    for row in per_query_rows:
        key = (
            int(row["subset_size"]),
            str(row["combination"]),
            str(row["target"]),
            str(row["variant"]),
        )
        grouped[key].append(float(row["lds_percent"]))
    combination_rows = [
        {
            "subset_size": subset_size,
            "combination": combination,
            "target": target,
            "variant": variant,
            "mean_lds_percent": statistics.mean(values),
        }
        for (subset_size, combination, target, variant), values in sorted(grouped.items())
    ]

    by_size: dict[tuple[int, str, str], list[tuple[float, str]]] = defaultdict(list)
    for row in combination_rows:
        by_size[(int(row["subset_size"]), str(row["target"]), str(row["variant"]))].append(
            (float(row["mean_lds_percent"]), str(row["combination"]))
        )
    distribution_rows = []
    for (subset_size, target, variant), items in sorted(by_size.items()):
        values = [value for value, _ in items]
        best_value, best_subset = max(items)
        distribution_rows.append(
            {
                "subset_size": subset_size,
                "num_combinations": len(items),
                "target": target,
                "variant": variant,
                "mean_lds_percent": statistics.mean(values),
                "std_lds_percent": standard_deviation(values),
                "q1_lds_percent": float(np.percentile(values, 25.0)),
                "median_lds_percent": float(np.percentile(values, 50.0)),
                "q3_lds_percent": float(np.percentile(values, 75.0)),
                "min_lds_percent": min(values),
                "max_lds_percent": max(values),
                "best_combination": best_subset,
                "best_lds_percent": best_value,
            }
        )

    analysis_label = args.analysis_label or (
        f"probe{args.num_probes}_all_subset_sizes_linear"
    )
    output_dir = (
        result_root
        / "eval"
        / analysis_label
        / f"run_{args.run_id}"
    )
    write_csv(output_dir / "per_query.csv", per_query_rows)
    write_csv(output_dir / "combination_ten_query_means.csv", combination_rows)
    write_csv(output_dir / "subset_size_distribution.csv", distribution_rows)
    save_plot(
        output_dir / "both_l2_counterfactual_by_subset_size.png",
        distribution_rows,
        args.num_probes,
        args.score_label,
    )
    save_all_points_plot(
        output_dir / "all_targets_variants_subset_scatter.png",
        combination_rows,
        args.num_probes,
        args.score_label,
    )

    mean_lookup = {
        (int(row["subset_size"]), str(row["combination"]), str(row["target"]), str(row["variant"])):
        float(row["mean_lds_percent"])
        for row in combination_rows
    }
    print("\nBOTH-L2 COUNTERFACTUAL BY PROBE COUNT")
    print(
        f"{'K':>2s} {'N':>4s} {'ENDPOINT MEAN±STD':>21s} "
        f"{'TRAJ MEAN±STD':>21s} {'CF MEAN±STD':>21s} {'BEST':>9s}"
    )
    print("-" * 108)
    for subset_size in range(1, args.num_probes + 1):
        subset_tags = [
            ",".join(str(index + 1) for index in item)
            for item in itertools.combinations(range(args.num_probes), subset_size)
        ]
        endpoints = [
            mean_lookup[(subset_size, tag, "endpoint_contarfactual", "query_train_l2")]
            for tag in subset_tags
        ]
        trajectories = [
            mean_lookup[(subset_size, tag, "traj_contarfactual", "query_train_l2")]
            for tag in subset_tags
        ]
        cf_values = [0.5 * (endpoint + trajectory) for endpoint, trajectory in zip(endpoints, trajectories)]
        best_index = int(np.argmax(cf_values))
        print(
            f"{subset_size:2d} {len(subset_tags):4d} "
            f"{statistics.mean(endpoints):8.3f}±{standard_deviation(endpoints):5.3f}% "
            f"{statistics.mean(trajectories):8.3f}±{standard_deviation(trajectories):5.3f}% "
            f"{statistics.mean(cf_values):8.3f}±{standard_deviation(cf_values):5.3f}% "
            f"{subset_tags[best_index]:>9s}"
        )

    print("\nALL TARGETS AND NORMALIZATIONS")
    print(f"{'K':>2s} {'N':>4s} {'TARGET':24s} {'VARIANT':15s} {'MEAN':>9s} {'STD':>9s}")
    print("-" * 72)
    for row in distribution_rows:
        print(
            f"{int(row['subset_size']):2d} {int(row['num_combinations']):4d} "
            f"{str(row['target']):24s} {str(row['variant']):15s} "
            f"{float(row['mean_lds_percent']):8.3f}% {float(row['std_lds_percent']):8.3f}%"
        )

    for name in (
        "per_query.csv",
        "combination_ten_query_means.csv",
        "subset_size_distribution.csv",
        "both_l2_counterfactual_by_subset_size.png",
        "both_l2_counterfactual_by_subset_size.svg",
        "all_targets_variants_subset_scatter.png",
        "all_targets_variants_subset_scatter.pdf",
    ):
        print(f"[saved] {output_dir / name}")


if __name__ == "__main__":
    main()
