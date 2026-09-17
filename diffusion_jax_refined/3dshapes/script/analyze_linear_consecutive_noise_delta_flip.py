#!/usr/bin/env python3
"""Flip linear checkpoint components when consecutive noise updates oppose."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from analyze_original_f_timestamp_sign_crossfit import TARGETS, lds, target_data
from analyze_predicted_noise_probe24_output_alignment import artifact_path, write_csv
from analyze_q1348_own_trajectory_product_square import parse_ints
from run_expected_residual_jacobian_scores import load_query_bank, train_part_dir


VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")


def checkpoint_flip_signs(args, query_ids, expected_timesteps):
    direct_signs = np.ones((len(query_ids), 49), dtype=np.float64)
    cumulative_signs = np.ones((len(query_ids), 49), dtype=np.float64)
    timestamp_cumulative_signs = np.ones(
        (len(query_ids), 49, len(expected_timesteps)), dtype=np.float64
    )
    first_checkpoint_signs = np.ones((len(query_ids), 49), dtype=np.float64)
    first_timestamp_signs = np.ones(
        (len(query_ids), 49, len(expected_timesteps)), dtype=np.float64
    )
    rows = []
    timestamp_rows = []
    for query_position, query in enumerate(query_ids):
        path = artifact_path(
            args.experiment,
            args.train_seed,
            args.epochs,
            query,
            args.delta_namespace,
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            updates = np.asarray(
                payload["checkpoint_next_predicted_noise_deltas"], dtype=np.float64
            )
            term_timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        timesteps = term_timesteps[: updates.shape[1]]
        if not np.array_equal(timesteps, expected_timesteps):
            raise ValueError(f"Q{query}: timestep mismatch")
        flattened = updates.reshape(updates.shape[0], updates.shape[1], -1)
        previous = flattened[:-1]
        current = flattened[1:]
        dot = np.sum(previous * current, axis=2)
        denominator = np.maximum(
            np.linalg.norm(previous, axis=2) * np.linalg.norm(current, axis=2),
            1e-12,
        )
        cosines = np.clip(dot / denominator, -1.0, 1.0)
        mean_cosines = np.mean(cosines, axis=1)
        transition_signs = np.where(mean_cosines < 0.0, -1.0, 1.0)
        direct_signs[query_position, 1:] = transition_signs
        cumulative_signs[query_position, 1:] = np.cumprod(transition_signs)
        timestamp_transition_signs = np.where(cosines < 0.0, -1.0, 1.0)
        timestamp_cumulative_signs[query_position, 1:] = np.cumprod(
            timestamp_transition_signs, axis=0
        )
        first = flattened[0:1]
        anchor_dot = np.sum(flattened * first, axis=2)
        anchor_denominator = np.maximum(
            np.linalg.norm(flattened, axis=2)
            * np.linalg.norm(first, axis=2),
            1e-12,
        )
        anchor_cosines = np.clip(
            anchor_dot / anchor_denominator, -1.0, 1.0
        )
        anchor_mean_cosines = np.mean(anchor_cosines, axis=1)
        first_checkpoint_signs[query_position] = np.where(
            anchor_mean_cosines < 0.0, -1.0, 1.0
        )
        first_timestamp_signs[query_position] = np.where(
            anchor_cosines < 0.0, -1.0, 1.0
        )
        rows.append(
            {
                "query": query,
                "checkpoint": 1,
                "epoch": 4,
                "previous_update_cosine_mean": "",
                "opposite_timestamp_fraction": "",
                "direct_multiplier": 1,
                "cumulative_multiplier": 1,
                "first_checkpoint_cosine_mean": float(anchor_mean_cosines[0]),
                "first_checkpoint_multiplier": int(
                    first_checkpoint_signs[query_position, 0]
                ),
            }
        )
        for checkpoint, (mean_cosine, values) in enumerate(
            zip(mean_cosines, cosines), start=2
        ):
            rows.append(
                {
                    "query": query,
                    "checkpoint": checkpoint,
                    "epoch": 4 * checkpoint,
                    "previous_update_cosine_mean": float(mean_cosine),
                    "opposite_timestamp_fraction": float(np.mean(values < 0.0)),
                    "direct_multiplier": int(
                        direct_signs[query_position, checkpoint - 1]
                    ),
                    "cumulative_multiplier": int(
                        cumulative_signs[query_position, checkpoint - 1]
                    ),
                    "first_checkpoint_cosine_mean": float(
                        anchor_mean_cosines[checkpoint - 1]
                    ),
                    "first_checkpoint_multiplier": int(
                        first_checkpoint_signs[
                            query_position, checkpoint - 1
                        ]
                    ),
                }
            )
        for slot, timestep in enumerate(expected_timesteps):
            timestamp_rows.append(
                {
                    "query": query,
                    "timestep": int(timestep),
                    "negative_transitions": int(
                        np.sum(timestamp_transition_signs[:, slot] < 0.0)
                    ),
                    "positive_transitions": int(
                        np.sum(timestamp_transition_signs[:, slot] > 0.0)
                    ),
                    "cumulative_flipped_checkpoints": int(
                        np.sum(
                            timestamp_cumulative_signs[query_position, :, slot]
                            < 0.0
                        )
                    ),
                    "transition_cosine_mean": float(np.mean(cosines[:, slot])),
                    "first_checkpoint_negative_checkpoints": int(
                        np.sum(anchor_cosines[:, slot] < 0.0)
                    ),
                    "first_checkpoint_positive_checkpoints": int(
                        np.sum(anchor_cosines[:, slot] >= 0.0)
                    ),
                    "first_checkpoint_cosine_mean": float(
                        np.mean(anchor_cosines[:, slot])
                    ),
                }
            )
    return (
        direct_signs,
        cumulative_signs,
        timestamp_cumulative_signs,
        first_checkpoint_signs,
        first_timestamp_signs,
        rows,
        timestamp_rows,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--num-checkpoints", type=int, default=50)
    parser.add_argument("--train-namespace", default="traj_tracin")
    parser.add_argument(
        "--query-namespace",
        default="loss_direction_original_f_checkpoint_own_trajectory",
    )
    parser.add_argument(
        "--delta-namespace",
        default="predicted_noise_cross_query_own_trajectory",
    )
    parser.add_argument(
        "--train-feature-semantics",
        default="raw_projected_expected_loss_gradient",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.skip_predicted = True
    args.shard_index = 0
    args.shard_count = 1
    args.run_id = "linear_consecutive_noise_delta_flip"
    query_ids = parse_ints(args.query_ids)

    query_bank, metadata = load_query_bank(
        args,
        args.query_namespace,
        "trajectory_next_checkpoint_noise_mse",
        query_ids=query_ids,
    )
    term_checkpoints = np.asarray(metadata["ckpt_indices"], dtype=np.int32)
    term_timesteps = np.asarray(metadata["timesteps"], dtype=np.int32)
    term_weights = np.asarray(metadata["term_weights"], dtype=np.float64)
    unique_checkpoints = np.unique(term_checkpoints)
    first_checkpoint = np.flatnonzero(term_checkpoints == unique_checkpoints[0])
    expected_timesteps = term_timesteps[first_checkpoint]
    (
        direct_signs,
        cumulative_signs,
        timestamp_cumulative_signs,
        first_checkpoint_signs,
        first_timestamp_signs,
        sign_rows,
        timestamp_sign_rows,
    ) = checkpoint_flip_signs(args, query_ids, expected_timesteps)
    timestep_position = {
        int(timestep): position
        for position, timestep in enumerate(expected_timesteps)
    }
    lookup = {
        (int(checkpoint), int(timestep)): term
        for term, (checkpoint, timestep) in enumerate(
            zip(term_checkpoints, term_timesteps)
        )
    }
    checkpoint_position = {
        int(checkpoint): position
        for position, checkpoint in enumerate(unique_checkpoints)
    }
    baseline = {
        variant: np.zeros((len(query_ids), 5000), dtype=np.float64)
        for variant in VARIANTS
    }
    flipped = {
        variant: np.zeros((len(query_ids), 5000), dtype=np.float64)
        for variant in VARIANTS
    }
    cumulative_flipped = {
        variant: np.zeros((len(query_ids), 5000), dtype=np.float64)
        for variant in VARIANTS
    }
    timestamp_cumulative_flipped = {
        variant: np.zeros((len(query_ids), 5000), dtype=np.float64)
        for variant in VARIANTS
    }
    first_checkpoint_flipped = {
        variant: np.zeros((len(query_ids), 5000), dtype=np.float64)
        for variant in VARIANTS
    }
    first_timestamp_flipped = {
        variant: np.zeros((len(query_ids), 5000), dtype=np.float64)
        for variant in VARIANTS
    }
    score_indices = None
    used_terms = 0

    import jax
    import jax.numpy as jnp

    for checkpoint in range(args.num_checkpoints):
        path = train_part_dir(args) / f"ckpt_{checkpoint:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            train_terms = np.asarray(payload["train_features"], dtype=np.float32)
            part_checkpoints = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            part_timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            indices = np.asarray(payload["score_indices"], dtype=np.int64)
            semantics = str(
                np.asarray(
                    payload.get(
                        "train_feature_semantics",
                        "raw_projected_expected_loss_gradient",
                    )
                ).item()
            )
        if semantics != args.train_feature_semantics:
            raise ValueError(f"{path}: unexpected semantics {semantics!r}")
        if score_indices is None:
            score_indices = indices
        elif not np.array_equal(score_indices, indices):
            raise ValueError(f"score indices mismatch: {path}")
        for local, (term_checkpoint, timestep) in enumerate(
            zip(part_checkpoints, part_timesteps)
        ):
            term = lookup.get((int(term_checkpoint), int(timestep)))
            if term is None:
                continue
            position = checkpoint_position[int(term_checkpoint)]
            timestamp_slot = timestep_position[int(timestep)]
            train = jax.device_put(jnp.asarray(train_terms[local]))
            query = jax.device_put(jnp.asarray(query_bank[:, term, :]))
            dots = train @ query.T
            train_norm = jnp.linalg.norm(train, axis=1) + 1e-8
            query_norm = jnp.linalg.norm(query, axis=1) + 1e-8
            values = {
                "raw": dots,
                "query_l2": dots / query_norm[None, :],
                "train_l2": dots / train_norm[:, None],
                "query_train_l2": dots
                / (train_norm[:, None] * query_norm[None, :]),
            }
            for variant, value in values.items():
                component = (
                    term_weights[term]
                    * np.asarray(jax.device_get(value), dtype=np.float64).T
                )
                # Original-f orientation is the negative accumulated product.
                baseline[variant] -= component
                flipped[variant] -= direct_signs[:, position, None] * component
                cumulative_flipped[variant] -= (
                    cumulative_signs[:, position, None] * component
                )
                timestamp_cumulative_flipped[variant] -= (
                    timestamp_cumulative_signs[
                        :, position, timestamp_slot, None
                    ]
                    * component
                )
                first_checkpoint_flipped[variant] -= (
                    first_checkpoint_signs[:, position, None] * component
                )
                first_timestamp_flipped[variant] -= (
                    first_timestamp_signs[:, position, timestamp_slot, None]
                    * component
                )
            used_terms += 1
        print(
            f"[linear delta flip] checkpoint={checkpoint + 1}/"
            f"{args.num_checkpoints} terms={used_terms}",
            flush=True,
        )
    if used_terms != len(term_checkpoints) or score_indices is None:
        raise ValueError(f"expected {len(term_checkpoints)} terms, found {used_terms}")

    rows = []
    for query_position, query in enumerate(query_ids):
        incidence, true = target_data(args, query, score_indices)
        endpoint = true[TARGETS[0]]
        trajectory = true[TARGETS[1]]
        direct_flipped_count = int(
            np.sum(direct_signs[query_position] < 0.0)
        )
        cumulative_flipped_count = int(
            np.sum(cumulative_signs[query_position] < 0.0)
        )
        timestamp_cumulative_flipped_count = int(
            np.sum(timestamp_cumulative_signs[query_position] < 0.0)
        )
        first_checkpoint_flipped_count = int(
            np.sum(first_checkpoint_signs[query_position] < 0.0)
        )
        first_timestamp_flipped_count = int(
            np.sum(first_timestamp_signs[query_position] < 0.0)
        )
        for variant in VARIANTS:
            for method, bank, flipped_count in (
                ("baseline", baseline, 0),
                ("delta_flip", flipped, direct_flipped_count),
                (
                    "cumulative_flip",
                    cumulative_flipped,
                    cumulative_flipped_count,
                ),
                (
                    "cumulative_timestamp_flip",
                    timestamp_cumulative_flipped,
                    timestamp_cumulative_flipped_count,
                ),
                (
                    "first_checkpoint_flip",
                    first_checkpoint_flipped,
                    first_checkpoint_flipped_count,
                ),
                (
                    "first_timestamp_flip",
                    first_timestamp_flipped,
                    first_timestamp_flipped_count,
                ),
            ):
                prediction = bank[variant][query_position] @ incidence.T
                endpoint_lds, trajectory_lds, joint_lds = lds(
                    prediction, endpoint, trajectory
                )
                rows.append(
                    {
                        "query": query,
                        "variant": variant,
                        "method": method,
                        "flipped_checkpoints": flipped_count,
                        "endpoint_percent": float(endpoint_lds[0]),
                        "trajectory_percent": float(trajectory_lds[0]),
                        "cf_joint_percent": float(joint_lds[0]),
                    }
                )

    mean_rows = []
    for variant in VARIANTS:
        for method in (
            "baseline",
            "delta_flip",
            "cumulative_flip",
            "cumulative_timestamp_flip",
            "first_checkpoint_flip",
            "first_timestamp_flip",
        ):
            selected = [
                row
                for row in rows
                if row["variant"] == variant and row["method"] == method
            ]
            mean_rows.append(
                {
                    "variant": variant,
                    "method": method,
                    "queries": len(selected),
                    "endpoint_percent_mean": float(
                        np.mean([row["endpoint_percent"] for row in selected])
                    ),
                    "trajectory_percent_mean": float(
                        np.mean([row["trajectory_percent"] for row in selected])
                    ),
                    "cf_joint_percent_mean": float(
                        np.mean([row["cf_joint_percent"] for row in selected])
                    ),
                    "positive_query_fraction": float(
                        np.mean([row["cf_joint_percent"] > 0.0 for row in selected])
                    ),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "checkpoint_signs.csv", sign_rows)
    write_csv(args.out_dir / "timestamp_sign_summary.csv", timestamp_sign_rows)
    write_csv(args.out_dir / "per_query_results.csv", rows)
    write_csv(args.out_dir / "mean_results.csv", mean_rows)
    print("LINEAR ORIGINAL-F — CONSECUTIVE PREDICTED-NOISE-DELTA CHECKPOINT FLIP")
    print("Q VARIANT             METHOD                       FLIPS ENDPOINT    TRAJ   JOINT")
    print("-" * 93)
    for row in rows:
        print(
            f"{int(row['query']):1d} {row['variant']:<19s} "
            f"{row['method']:<28s} {int(row['flipped_checkpoints']):5d} "
            f"{row['endpoint_percent']:+8.3f}% "
            f"{row['trajectory_percent']:+7.3f}% "
            f"{row['cf_joint_percent']:+7.3f}%"
        )
    print("\nTEN-QUERY MEAN")
    print("VARIANT             METHOD                       ENDPOINT    TRAJ   JOINT  Q>0")
    print("-" * 87)
    for row in mean_rows:
        print(
            f"{row['variant']:<19s} {row['method']:<28s} "
            f"{row['endpoint_percent_mean']:+8.3f}% "
            f"{row['trajectory_percent_mean']:+7.3f}% "
            f"{row['cf_joint_percent_mean']:+7.3f}% "
            f"{row['positive_query_fraction']:4.2f}"
        )
    print("\nPER-TIMESTAMP CUMULATIVE SIGN COUNTS")
    print(
        "Q    T NEG-TRANS POS-TRANS CUM-FLIP/49 TRANS-COS "
        "FIRST-NEG FIRST-POS FIRST-COS"
    )
    print("-" * 96)
    for row in timestamp_sign_rows:
        print(
            f"{int(row['query']):1d} {int(row['timestep']):4d} "
            f"{int(row['negative_transitions']):9d} "
            f"{int(row['positive_transitions']):9d} "
            f"{int(row['cumulative_flipped_checkpoints']):10d}/49 "
            f"{row['transition_cosine_mean']:+8.4f} "
            f"{int(row['first_checkpoint_negative_checkpoints']):9d} "
            f"{int(row['first_checkpoint_positive_checkpoints']):9d} "
            f"{row['first_checkpoint_cosine_mean']:+9.4f}"
        )
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
