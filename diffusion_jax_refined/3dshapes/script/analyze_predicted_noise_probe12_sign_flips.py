#!/usr/bin/env python3
"""Exhaustively evaluate all sign assignments of a saved 12-probe bank."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_predicted_noise_probe8_choose4 import (
    SHAPES_ROOT,
    TARGETS,
    average_tie_ranks,
    cache_group,
    load_probe_scores,
    load_target_data,
)


def sign_matrix(num_probes: int) -> np.ndarray:
    masks = np.arange(1 << num_probes, dtype=np.uint32)
    bits = (masks[:, None] >> np.arange(num_probes, dtype=np.uint32)) & 1
    return np.where(bits == 0, 1.0, -1.0)


def rowwise_spearman(predictions: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Spearman correlations for continuous prediction rows against one target."""
    predictions = np.asarray(predictions, dtype=np.float64)
    order = np.argsort(predictions, axis=1, kind="mergesort")
    ranks = np.empty_like(predictions)
    np.put_along_axis(
        ranks,
        order,
        np.arange(predictions.shape[1], dtype=np.float64)[None, :],
        axis=1,
    )
    ranks -= ranks.mean(axis=1, keepdims=True)
    target_ranks = average_tie_ranks(np.asarray(target, dtype=np.float64))
    target_ranks -= target_ranks.mean()
    denominator = np.sqrt(
        np.sum(ranks * ranks, axis=1) * np.sum(target_ranks * target_ranks)
    )
    return (ranks @ target_ranks) / denominator


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_histogram(path: Path, values: np.ndarray, original: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warning] matplotlib unavailable; skipping plot: {exc}")
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.hist(values, bins=60, color="#4472c4", alpha=0.82)
    ax.axvline(original, color="#c62828", linewidth=2.2, label=f"all-plus: {original:.3f}%")
    ax.axvline(0.0, color="black", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("Endpoint/trajectory joint 10-query mean LDS (%)")
    ax.set_ylabel("Number of sign assignments")
    ax.set_title("12-probe linear Both-L2: all 4096 sign assignments")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    num_probes = 12
    result_root = SHAPES_ROOT / "result" / args.experiment
    shard_dir = (
        result_root
        / "stream_score"
        / "traj_tracin_predicted_noise_jvp_final_post_square_probe12"
        / f"train_seed_{args.train_seed}"
        / f"run_{args.run_id}"
        / "shards"
    )
    probe_scores, score_indices = load_probe_scores(shard_dir, num_probes)
    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    signs = sign_matrix(num_probes)
    lds = {
        (target, variant): np.empty((len(signs), len(records)), dtype=np.float64)
        for target in TARGETS
        for variant in probe_scores
    }

    for query_id, record in enumerate(records):
        prompt_tag = str(record["prompt"]).replace(",", "_")
        eval_root = (
            result_root
            / "eval"
            / "prompted_solo"
            / f"query_{prompt_tag}"
            / f"initial_seed_{int(record['initial_seed'])}"
        )
        incidence, true_values = load_target_data(cache_group(eval_root), score_indices)
        for variant, values in probe_scores.items():
            per_probe_predictions = values[:, query_id, :] @ incidence.T
            predictions = (signs @ per_probe_predictions) / float(num_probes)
            for target in TARGETS:
                lds[(target, variant)][:, query_id] = (
                    100.0 * rowwise_spearman(predictions, true_values[target])
                )
        print(f"[query {query_id}/9] evaluated 4096 sign assignments", flush=True)

    output_dir = (
        result_root
        / "eval"
        / "probe12_all_sign_flips_linear"
        / f"run_{args.run_id}"
    )
    assignment_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    means_by_key = {key: values.mean(axis=1) for key, values in lds.items()}
    sign_labels = ["".join("+" if value > 0 else "-" for value in row) for row in signs]
    for (target, variant), means in sorted(means_by_key.items()):
        for mask, (label, value) in enumerate(zip(sign_labels, means)):
            assignment_rows.append(
                {
                    "mask": mask,
                    "signs_probe_1_to_12": label,
                    "target": target,
                    "variant": variant,
                    "ten_query_mean_lds_percent": float(value),
                }
            )
        original = float(means[0])
        pair_error = float(np.max(np.abs(means + means[::-1])))
        summary_rows.append(
            {
                "target": target,
                "variant": variant,
                "all_plus_lds_percent": original,
                "all_minus_lds_percent": float(means[-1]),
                "sign_mean_lds_percent": float(np.mean(means)),
                "sign_std_lds_percent": float(np.std(means, ddof=1)),
                "sign_min_lds_percent": float(np.min(means)),
                "sign_max_lds_percent": float(np.max(means)),
                "all_plus_percentile": float(100.0 * np.mean(means <= original)),
                "max_sign_pair_symmetry_error": pair_error,
            }
        )

    endpoint = means_by_key[("endpoint_contarfactual", "query_train_l2")]
    trajectory = means_by_key[("traj_contarfactual", "query_train_l2")]
    joint = 0.5 * (endpoint + trajectory)
    joint_rows = [
        {
            "mask": mask,
            "signs_probe_1_to_12": label,
            "endpoint_mean_lds_percent": float(endpoint[mask]),
            "trajectory_mean_lds_percent": float(trajectory[mask]),
            "joint_mean_lds_percent": float(joint[mask]),
        }
        for mask, label in enumerate(sign_labels)
    ]
    write_csv(output_dir / "all_sign_assignments.csv", assignment_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "both_l2_counterfactual_signs.csv", joint_rows)
    save_histogram(output_dir / "both_l2_counterfactual_sign_histogram.png", joint, float(joint[0]))

    print("\n12-PROBE FULL-BANK SIGN-FLIP TEST")
    print(
        f"{'TARGET':24s} {'VARIANT':15s} {'ALL+':>8s} {'ALL-':>8s} "
        f"{'STD':>8s} {'MIN':>8s} {'MAX':>8s} {'ALL+ PCTL':>10s}"
    )
    print("-" * 101)
    for row in summary_rows:
        print(
            f"{str(row['target']):24s} {str(row['variant']):15s} "
            f"{float(row['all_plus_lds_percent']):7.3f}% "
            f"{float(row['all_minus_lds_percent']):7.3f}% "
            f"{float(row['sign_std_lds_percent']):7.3f}% "
            f"{float(row['sign_min_lds_percent']):7.3f}% "
            f"{float(row['sign_max_lds_percent']):7.3f}% "
            f"{float(row['all_plus_percentile']):9.2f}%"
        )
    print("\nBOTH-L2 COUNTERFACTUAL JOINT")
    print(f"all-plus : {joint[0]:.4f}%")
    print(f"all-minus: {joint[-1]:.4f}%")
    print(f"mean     : {np.mean(joint):.4f}%")
    print(f"std      : {np.std(joint, ddof=1):.4f}%")
    print(f"min/max  : {np.min(joint):.4f}% / {np.max(joint):.4f}%")
    print(f"all+ pct : {100.0 * np.mean(joint <= joint[0]):.2f}%")
    print(f"pair err : {np.max(np.abs(joint + joint[::-1])):.3e}")
    for name in (
        "all_sign_assignments.csv",
        "summary.csv",
        "both_l2_counterfactual_signs.csv",
        "both_l2_counterfactual_sign_histogram.png",
        "both_l2_counterfactual_sign_histogram.svg",
    ):
        print(f"[saved] {output_dir / name}")


if __name__ == "__main__":
    main()
