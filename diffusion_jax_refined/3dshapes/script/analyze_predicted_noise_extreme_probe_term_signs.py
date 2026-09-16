#!/usr/bin/env python3
"""Audit 50x10 internal term signs for near-all-positive/negative proper scores."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_predicted_noise_old_fresh_term_signs import BANKS, write_csv
from run_predicted_noise_jvp_l2_squared import query_artifact_path, train_part_dir


SHAPES_ROOT = Path(__file__).resolve().parents[1]


def load_inputs(args: argparse.Namespace):
    result_root = SHAPES_ROOT / "result" / args.experiment
    grouped_path = (
        result_root
        / "eval"
        / "predicted_noise_old_fresh12_per_probe_timestamp_orientation"
        / f"run_{args.source_run_id}"
        / "per_probe_timestamp_scores.npz"
    )
    signs_path = (
        result_root
        / "eval"
        / "predicted_noise_24_individual_probe_timestamp_signs"
        / f"source_run_{args.source_run_id}"
        / "individual_probe_best_binary_vectors.npz"
    )
    if not grouped_path.is_file():
        raise FileNotFoundError(grouped_path)
    if not signs_path.is_file():
        raise FileNotFoundError(signs_path)
    with np.load(grouped_path, allow_pickle=False) as payload:
        grouped = np.concatenate(
            [
                np.asarray(payload["old12"], dtype=np.float64),
                np.asarray(payload["fresh12"], dtype=np.float64),
            ],
            axis=0,
        )
        query_ids = np.asarray(payload["query_ids"], dtype=np.int32)
        timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        score_indices = np.asarray(payload["score_indices"], dtype=np.int64)
    with np.load(signs_path, allow_pickle=False) as payload:
        best_signs = np.asarray(payload["best_signs"], dtype=np.int8)
        for key, reference in (
            ("query_ids", query_ids),
            ("timesteps", timesteps),
            ("score_indices", score_indices),
        ):
            if not np.array_equal(payload[key], reference):
                raise ValueError(f"{key} differs between grouped-score and sign caches")
    expected = (24, 10, len(query_ids), len(score_indices))
    if grouped.shape != expected:
        raise ValueError(f"expected grouped shape {expected}, got {grouped.shape}")
    if best_signs.shape != (24, len(query_ids), 10):
        raise ValueError(f"unexpected best_signs shape {best_signs.shape}")
    return grouped, best_signs, query_ids, timesteps, score_indices


def select_extremes(grouped, best_signs, query_ids):
    proper = np.sum(
        np.transpose(grouped, (0, 2, 1, 3)) * best_signs[..., None], axis=2
    )
    positive_fraction = np.mean(proper > 0.0, axis=-1)
    selections: list[dict[str, object]] = []
    for qslot, query_id_value in enumerate(query_ids):
        query_id = int(query_id_value)
        fractions = positive_fraction[:, qslot]
        for selection, target in (
            ("closest_to_0_percent_positive", float(np.min(fractions))),
            ("closest_to_100_percent_positive", float(np.max(fractions))),
        ):
            for probe_slot in np.flatnonzero(fractions == target):
                selections.append(
                    {
                        "selection": selection,
                        "query": query_id,
                        "query_slot": qslot,
                        "global_probe": int(probe_slot) + 1,
                        "probe_slot": int(probe_slot),
                        "bank": "old12" if probe_slot < 12 else "fresh12",
                        "probe_in_bank": int(probe_slot) + 1 if probe_slot < 12 else int(probe_slot) - 11,
                        "proper_positive_fraction": target,
                    }
                )
    return selections


def load_selected_query_features(args, selections):
    features = []
    reference = None
    for item in selections:
        pattern = BANKS[str(item["bank"])]["query_pattern"]
        path = query_artifact_path(
            args.experiment,
            args.train_seed,
            args.epochs,
            int(item["query"]),
            num_probes=12,
            probe_index=int(item["probe_in_bank"]) - 1,
            query_namespace_pattern=pattern,
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            value = np.asarray(payload["query_features"], dtype=np.float32)
            metadata = {
                "ckpt_indices": np.asarray(payload["ckpt_indices"], dtype=np.int32),
                "timesteps": np.asarray(payload["timesteps"], dtype=np.int32),
            }
        value /= np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-8)
        features.append(value)
        if reference is None:
            reference = metadata
        else:
            for key, expected in reference.items():
                if not np.array_equal(metadata[key], expected):
                    raise ValueError(f"query artifact metadata differs: {path}:{key}")
    assert reference is not None
    return np.stack(features, axis=0), reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--source-run-id", required=True)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    grouped, best_signs, query_ids, timesteps, score_indices = load_inputs(args)
    selections = select_extremes(grouped, best_signs, query_ids)
    query_features, metadata = load_selected_query_features(args, selections)
    lookup = {
        (int(checkpoint), int(timestep)): term
        for term, (checkpoint, timestep) in enumerate(
            zip(metadata["ckpt_indices"], metadata["timesteps"])
        )
    }
    timestep_slots = {int(value): slot for slot, value in enumerate(timesteps)}
    pair_count = len(selections)
    datapoint_count = len(score_indices)
    total_positive = np.zeros(pair_count, dtype=np.int64)
    total_negative = np.zeros(pair_count, dtype=np.int64)
    total_zero = np.zeros(pair_count, dtype=np.int64)
    term_mean_positive = np.zeros(pair_count, dtype=np.int64)
    term_mean_negative = np.zeros(pair_count, dtype=np.int64)
    per_term_positive_fraction: list[list[float]] = [[] for _ in selections]
    per_timestamp_counts = defaultdict(lambda: np.zeros((pair_count, 3), dtype=np.int64))
    reconstructed = np.zeros((pair_count, len(timesteps), datapoint_count), dtype=np.float64)
    term_count = np.zeros(pair_count, dtype=np.int64)
    term_positive_grid = np.full((pair_count, 50, len(timesteps)), np.nan, dtype=np.float32)
    term_mean_grid = np.full((pair_count, 50, len(timesteps)), np.nan, dtype=np.float32)
    term_std_grid = np.full((pair_count, 50, len(timesteps)), np.nan, dtype=np.float32)
    selected_signs = np.asarray(
        [
            best_signs[int(item["probe_slot"]), int(item["query_slot"])]
            for item in selections
        ],
        dtype=np.int8,
    )

    for checkpoint_slot in range(50):
        path = (
            train_part_dir(args.experiment, args.train_seed)
            / f"ckpt_{checkpoint_slot:04d}.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            train = np.asarray(payload["train_features"], dtype=np.float32)
            indices = np.asarray(payload["score_indices"], dtype=np.int64)
            checkpoints = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            local_timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            weights = np.asarray(payload["term_weights"], dtype=np.float64)
        if not np.array_equal(indices, score_indices):
            raise ValueError(f"score indices differ in {path}")
        for local_term, (checkpoint, timestep, weight) in enumerate(
            zip(checkpoints, local_timesteps, weights)
        ):
            term = lookup[(int(checkpoint), int(timestep))]
            train_device = jnp.asarray(train[local_term])
            train_unit = train_device / jnp.maximum(
                jnp.linalg.norm(train_device, axis=1, keepdims=True), 1e-8
            )
            query_device = jnp.asarray(query_features[:, term, :])
            values = np.asarray(
                jax.device_get(train_unit @ query_device.T), dtype=np.float32
            ).T
            timestamp_slot = timestep_slots[int(timestep)]
            oriented = selected_signs[:, timestamp_slot, None] * values
            positive = oriented > 0.0
            negative = oriented < 0.0
            zero = oriented == 0.0
            total_positive += positive.sum(axis=1)
            total_negative += negative.sum(axis=1)
            total_zero += zero.sum(axis=1)
            term_means = oriented.mean(axis=1)
            term_stds = oriented.std(axis=1)
            term_mean_positive += term_means > 0.0
            term_mean_negative += term_means < 0.0
            term_count += 1
            for pair in range(pair_count):
                per_term_positive_fraction[pair].append(float(positive[pair].mean()))
            per_timestamp_counts[int(timestep)][:, 0] += positive.sum(axis=1)
            per_timestamp_counts[int(timestep)][:, 1] += negative.sum(axis=1)
            per_timestamp_counts[int(timestep)][:, 2] += zero.sum(axis=1)
            reconstructed[:, timestamp_slot] += float(weight) * values
            term_positive_grid[:, checkpoint_slot, timestamp_slot] = positive.mean(axis=1)
            term_mean_grid[:, checkpoint_slot, timestamp_slot] = term_means
            term_std_grid[:, checkpoint_slot, timestamp_slot] = term_stds
        print(f"[term signs] checkpoint={checkpoint_slot + 1}/50", flush=True)

    summary_rows = []
    timestamp_rows = []
    for pair, item in enumerate(selections):
        probe_slot = int(item["probe_slot"])
        query_slot = int(item["query_slot"])
        expected = grouped[probe_slot, :, query_slot, :]
        error = float(np.max(np.abs(reconstructed[pair] - expected)))
        denominator = total_positive[pair] + total_negative[pair] + total_zero[pair]
        fractions = np.asarray(per_term_positive_fraction[pair], dtype=np.float64)
        row = {
            key: item[key]
            for key in (
                "selection",
                "query",
                "global_probe",
                "bank",
                "probe_in_bank",
                "proper_positive_fraction",
            )
        }
        row.update(
            {
                "component_count": int(denominator),
                "component_positive_fraction": float(total_positive[pair] / denominator),
                "component_negative_fraction": float(total_negative[pair] / denominator),
                "component_zero_fraction": float(total_zero[pair] / denominator),
                "term_count": int(term_count[pair]),
                "term_mean_positive_fraction": float(term_mean_positive[pair] / term_count[pair]),
                "term_mean_negative_fraction": float(term_mean_negative[pair] / term_count[pair]),
                "median_term_positive_fraction": float(np.median(fractions)),
                "min_term_positive_fraction": float(np.min(fractions)),
                "max_term_positive_fraction": float(np.max(fractions)),
                "grouped_reconstruction_max_abs_error": error,
            }
        )
        summary_rows.append(row)
        for timestep in timesteps:
            counts = per_timestamp_counts[int(timestep)][pair]
            total = int(counts.sum())
            timestamp_rows.append(
                {
                    "selection": item["selection"],
                    "query": item["query"],
                    "global_probe": item["global_probe"],
                    "timestep": int(timestep),
                    "selected_sign": int(selected_signs[pair, timestep_slots[int(timestep)]]),
                    "component_count": total,
                    "component_positive_fraction": float(counts[0] / total),
                    "component_negative_fraction": float(counts[1] / total),
                    "component_zero_fraction": float(counts[2] / total),
                }
            )

    out_dir = (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "eval"
        / "predicted_noise_extreme_probe_term_signs"
        / f"source_run_{args.source_run_id}"
    )
    write_csv(out_dir / "extreme_probe_term_sign_summary.csv", summary_rows)
    write_csv(out_dir / "extreme_probe_term_sign_by_timestamp.csv", timestamp_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "extreme_probe_term_grids.npz",
        query=np.asarray([int(item["query"]) for item in selections], dtype=np.int32),
        global_probe=np.asarray(
            [int(item["global_probe"]) for item in selections], dtype=np.int32
        ),
        selection=np.asarray([str(item["selection"]) for item in selections]),
        timesteps=timesteps,
        checkpoints=np.arange(1, 51, dtype=np.int32),
        selected_signs=selected_signs,
        positive_fraction=term_positive_grid,
        mean=term_mean_grid,
        std=term_std_grid,
    )

    # Q0/P1 is the closest-to-0%-positive probe; Q0/P20 is the
    # closest-to-100%-positive probe. Plot their full 50x10 term maps.
    plot_pairs = []
    for query_id, probe in ((0, 1), (0, 20)):
        matches = [
            pair
            for pair, item in enumerate(selections)
            if int(item["query"]) == query_id and int(item["global_probe"]) == probe
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one selected Q{query_id}/P{probe}, got {matches}")
        plot_pairs.append(matches[0])
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
        for column, (pair, probe) in enumerate(zip(plot_pairs, (1, 20))):
            positive_values = term_positive_grid[pair]
            image = axes[0, column].imshow(
                positive_values,
                vmin=0.35,
                vmax=0.65,
                cmap="RdBu",
                aspect="auto",
                origin="lower",
            )
            axes[0, column].set_title(
                f"Q0 P{probe}: component positive fraction"
            )
            axes[0, column].set_xlabel("timestamp")
            axes[0, column].set_ylabel("checkpoint")
            axes[0, column].set_xticks(range(len(timesteps)), timesteps)
            axes[0, column].set_yticks(range(0, 50, 5), range(1, 51, 5))
            fig.colorbar(image, ax=axes[0, column], pad=0.01)

            standardized = term_mean_grid[pair] / np.maximum(
                term_std_grid[pair], 1e-12
            )
            limit = max(0.5, float(np.nanmax(np.abs(standardized))))
            image = axes[1, column].imshow(
                standardized,
                vmin=-limit,
                vmax=limit,
                cmap="RdBu_r",
                aspect="auto",
                origin="lower",
            )
            axes[1, column].set_title(f"Q0 P{probe}: component mean / SD")
            axes[1, column].set_xlabel("timestamp")
            axes[1, column].set_ylabel("checkpoint")
            axes[1, column].set_xticks(range(len(timesteps)), timesteps)
            axes[1, column].set_yticks(range(0, 50, 5), range(1, 51, 5))
            fig.colorbar(image, ax=axes[1, column], pad=0.01)
        fig.suptitle(
            "Q0 extreme probes after best timestamp flips: 50 checkpoints x 10 timestamps"
        )
        fig.savefig(out_dir / "q0_probe1_probe20_term_maps.png", dpi=180)
        fig.savefig(out_dir / "q0_probe1_probe20_term_maps.svg")
        plt.close(fig)
    except ImportError:
        print("[warning] matplotlib unavailable; saved term grids without plots", flush=True)

    print("EXTREME PROPER-SCORE PROBES — INTERNAL 50x10 TERM SIGNS")
    print(
        f"{'TYPE':8s} {'Q':>2s} {'P':>2s} {'PROPER+':>8s} "
        f"{'ALL TERM+':>10s} {'TERM MEAN+':>11s} {'RECON ERR':>10s}"
    )
    print("-" * 78)
    for row in summary_rows:
        label = "near0" if str(row["selection"]).startswith("closest_to_0") else "near100"
        print(
            f"{label:8s} {int(row['query']):2d} {int(row['global_probe']):2d} "
            f"{float(row['proper_positive_fraction']):8.2%} "
            f"{float(row['component_positive_fraction']):10.2%} "
            f"{float(row['term_mean_positive_fraction']):11.2%} "
            f"{float(row['grouped_reconstruction_max_abs_error']):10.2e}"
        )
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
