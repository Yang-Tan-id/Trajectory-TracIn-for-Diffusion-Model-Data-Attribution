#!/usr/bin/env python3
"""Evaluate termwise squared gradient products for own-trajectory original-f."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from analyze_original_f_timestamp_sign_crossfit import TARGETS, lds, target_data
from analyze_predicted_noise_probe24_output_alignment import artifact_path
from run_expected_residual_jacobian_scores import load_query_bank, train_part_dir


VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")


def parse_ints(value):
    return [int(item) for item in value.replace(",", " ").split()]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ddim_coefficient_squared(timestep, previous_timestep, alpha_bars):
    alpha_bar_t = float(alpha_bars[int(timestep)])
    alpha_bar_previous = (
        1.0
        if int(previous_timestep) < 0
        else float(alpha_bars[int(previous_timestep)])
    )
    coefficient = (
        np.sqrt(1.0 - alpha_bar_previous)
        - np.sqrt(alpha_bar_previous / alpha_bar_t)
        * np.sqrt(1.0 - alpha_bar_t)
    )
    return coefficient * coefficient


def reweight_terms(
    weights,
    checkpoints,
    timesteps,
    mode,
    *,
    query_ids,
    args,
):
    weights = np.asarray(weights, dtype=np.float64)
    checkpoints = np.asarray(checkpoints, dtype=np.int32)
    timesteps = np.asarray(timesteps, dtype=np.int32)
    if mode == "uniform":
        return np.broadcast_to(weights[None, :], (len(query_ids), len(weights))).copy()
    if mode == "t0_only":
        output = np.zeros_like(weights)
        for checkpoint in np.unique(checkpoints):
            indices = np.flatnonzero(checkpoints == checkpoint)
            zero_indices = indices[timesteps[indices] == 0]
            if len(zero_indices) != 1:
                raise ValueError(
                    f"checkpoint {int(checkpoint)} has {len(zero_indices)} t=0 terms"
                )
            output[zero_indices[0]] = float(weights[indices].sum())
        return np.broadcast_to(
            output[None, :], (len(query_ids), len(output))
        ).copy()
    if mode in (
        "own_endpoint_inverse_rmse",
        "own_endpoint_inverse_sqrt_rmse",
    ):
        output = np.zeros((len(query_ids), len(weights)), dtype=np.float64)
        for query_position, query_id in enumerate(query_ids):
            path = artifact_path(
                args.experiment,
                args.train_seed,
                args.epochs,
                query_id,
                args.trajectory_state_namespace,
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as payload:
                state_timesteps = np.asarray(
                    payload["checkpoint_own_trajectory_state_timesteps"], dtype=np.int32
                )
                if "checkpoint_own_trajectory_endpoint_rmse" in payload:
                    endpoint_rmse = np.asarray(
                        payload["checkpoint_own_trajectory_endpoint_rmse"],
                        dtype=np.float64,
                    )
                    states = None
                    endpoints = None
                else:
                    states = np.asarray(
                        payload["checkpoint_own_trajectory_states"], dtype=np.float64
                    )
                    endpoints = np.asarray(
                        payload["checkpoint_own_trajectory_endpoints"], dtype=np.float64
                    )
                    endpoint_rmse = None
            state_index = {
                int(timestep): index for index, timestep in enumerate(state_timesteps)
            }
            unique_checkpoints = np.unique(checkpoints)
            geometry_shape = (
                endpoint_rmse.shape
                if endpoint_rmse is not None
                else states.shape[:2]
            )
            if geometry_shape != (len(unique_checkpoints), len(state_timesteps)):
                raise ValueError(
                    f"Q{query_id} geometry shape mismatch: {geometry_shape} != "
                    f"{(len(unique_checkpoints), len(state_timesteps))}"
                )
            if endpoints is not None and endpoints.shape[0] != len(unique_checkpoints):
                raise ValueError(
                    f"Q{query_id} endpoint count mismatch: {endpoints.shape[0]} != "
                    f"{len(unique_checkpoints)}"
                )
            for checkpoint_position, checkpoint in enumerate(unique_checkpoints):
                indices = np.flatnonzero(checkpoints == checkpoint)
                distances = []
                for index in indices:
                    timestep = int(timesteps[index])
                    if timestep not in state_index:
                        raise ValueError(
                            f"Q{query_id} checkpoint {int(checkpoint)} missing timestep {timestep}"
                        )
                    if endpoint_rmse is not None:
                        distance = endpoint_rmse[
                            checkpoint_position, state_index[timestep]
                        ]
                    else:
                        difference = (
                            states[checkpoint_position, state_index[timestep]]
                            - endpoints[checkpoint_position]
                        )
                        distance = np.sqrt(np.mean(np.square(difference)))
                    distances.append(float(distance))
                distance_power = (
                    1.0 if mode == "own_endpoint_inverse_rmse" else 0.5
                )
                inverse_distance = np.power(
                    np.maximum(np.asarray(distances, dtype=np.float64), 1e-8),
                    -distance_power,
                )
                output[query_position, indices] = (
                    float(weights[indices].sum())
                    * inverse_distance
                    / float(inverse_distance.sum())
                )
        return output
    betas = np.linspace(1e-4, 0.02, 1000, dtype=np.float64)
    alpha_bars = np.cumprod(1.0 - betas)
    output = np.zeros_like(weights)
    for checkpoint in np.unique(checkpoints):
        indices = np.flatnonzero(checkpoints == checkpoint)
        checkpoint_timesteps = timesteps[indices]
        if mode == "local_ddim_step_squared":
            step_weights = np.asarray(
                [
                    ddim_coefficient_squared(timestep, int(timestep) - 1, alpha_bars)
                    if int(timestep) > 0
                    else ddim_coefficient_squared(timestep, -1, alpha_bars)
                    for timestep in checkpoint_timesteps
                ],
                dtype=np.float64,
            )
        elif mode == "snapshot_interval_ddim_step_squared":
            order = np.argsort(-checkpoint_timesteps)
            ordered_timesteps = checkpoint_timesteps[order]
            previous = np.concatenate(
                [ordered_timesteps[1:], np.asarray([-1], dtype=np.int32)]
            )
            ordered_weights = np.asarray(
                [
                    ddim_coefficient_squared(timestep, prior, alpha_bars)
                    for timestep, prior in zip(ordered_timesteps, previous)
                ],
                dtype=np.float64,
            )
            step_weights = np.empty_like(ordered_weights)
            step_weights[order] = ordered_weights
        else:
            raise ValueError(f"unknown timestep weighting mode {mode!r}")
        denominator = float(step_weights.sum())
        if denominator <= 0.0:
            raise ValueError(f"checkpoint {int(checkpoint)} has zero timestep weight")
        output[indices] = float(weights[indices].sum()) * step_weights / denominator
    return np.broadcast_to(output[None, :], (len(query_ids), len(output))).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-ids", default="1,3,4,8")
    parser.add_argument("--num-checkpoints", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260916)
    parser.add_argument(
        "--timestep-weighting",
        choices=(
            "uniform",
            "local_ddim_step_squared",
            "snapshot_interval_ddim_step_squared",
            "own_endpoint_inverse_rmse",
            "own_endpoint_inverse_sqrt_rmse",
            "t0_only",
        ),
        default="uniform",
    )
    parser.add_argument("--train-namespace", default="traj_tracin")
    parser.add_argument(
        "--original-query-namespace",
        default="loss_direction_original_f_checkpoint_own_trajectory",
    )
    parser.add_argument(
        "--trajectory-state-namespace",
        default="loss_direction_original_f_checkpoint_own_trajectory_endpoints_all10",
    )
    parser.add_argument(
        "--train-feature-semantics",
        default="raw_projected_expected_loss_gradient",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    query_ids = parse_ints(args.query_ids)
    args.skip_predicted = True
    args.shard_index = 0
    args.shard_count = 1
    args.run_id = "q1348_own_trajectory_product_square"

    query_bank, metadata = load_query_bank(
        args,
        args.original_query_namespace,
        "trajectory_next_checkpoint_noise_mse",
        query_ids=query_ids,
    )
    term_checkpoints = np.asarray(metadata["ckpt_indices"], dtype=np.int32)
    term_timesteps = np.asarray(metadata["timesteps"], dtype=np.int32)
    original_term_weights = np.asarray(metadata["term_weights"], dtype=np.float64)
    effective_term_weights = reweight_terms(
        original_term_weights,
        term_checkpoints,
        term_timesteps,
        args.timestep_weighting,
        query_ids=query_ids,
        args=args,
    )
    lookup = {
        (int(checkpoint), int(timestep)): term
        for term, (checkpoint, timestep) in enumerate(
            zip(term_checkpoints, term_timesteps)
        )
    }
    scores = {
        variant: np.zeros((len(query_ids), 5000), dtype=np.float64)
        for variant in VARIANTS
    }
    linear_scores = {
        variant: np.zeros((len(query_ids), 5000), dtype=np.float64)
        for variant in VARIANTS
    }
    root_scores = {
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
            weights = np.asarray(payload["term_weights"], dtype=np.float64)
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

        for local, (term_checkpoint, timestep, weight) in enumerate(
            zip(part_checkpoints, part_timesteps, weights)
        ):
            term = lookup.get((int(term_checkpoint), int(timestep)))
            if term is None:
                continue
            if not np.isclose(
                float(weight), original_term_weights[term], rtol=1e-6, atol=1e-12
            ):
                raise ValueError(f"term-weight mismatch for term {term}")
            score_weight = effective_term_weights[:, term, None]
            train = jax.device_put(jnp.asarray(train_terms[local]))
            query = jax.device_put(jnp.asarray(query_bank[:, term, :]))
            dots = train @ query.T
            train_norm = jnp.linalg.norm(train, axis=1) + 1e-8
            query_norm = jnp.linalg.norm(query, axis=1) + 1e-8
            values = {
                "raw": dots,
                "query_l2": dots / query_norm[None, :],
                "train_l2": dots / train_norm[:, None],
                "query_train_l2": dots / (train_norm[:, None] * query_norm[None, :]),
            }
            for variant, value in values.items():
                linear = np.asarray(jax.device_get(value), dtype=np.float64)
                squared = np.square(linear)
                linear_scores[variant] += score_weight * linear.T
                scores[variant] += score_weight * squared.T
                root_scores[variant] += score_weight * np.abs(linear).T
            used_terms += 1
        print(
            f"[product square] checkpoint={checkpoint + 1}/50 terms={used_terms}",
            flush=True,
        )

    if used_terms != 490 or score_indices is None:
        raise ValueError(f"expected 490 terms, found {used_terms}")

    rows = []
    crossfit_rows = []
    crossfit_summary = []
    linear_guided_rows = []
    linear_guided_summary = []
    for query_slot, query_id in enumerate(query_ids):
        incidence, true = target_data(args, query_id, score_indices)
        endpoint = true[TARGETS[0]]
        trajectory = true[TARGETS[1]]
        rng = np.random.default_rng(args.random_seed)
        folds = []
        for _ in range(args.repeats):
            permutation = rng.permutation(len(endpoint))
            folds.append((permutation[::2], permutation[1::2]))
        for variant in VARIANTS:
            subset_sum = scores[variant][query_slot] @ incidence.T
            # Original-f uses the negative accumulated gradient product.
            linear_subset_prediction = -linear_scores[variant][query_slot] @ incidence.T
            for sign_name, multiplier in (("m1", -1.0), ("p1", 1.0)):
                endpoint_values, trajectory_values, joint_values = lds(
                    multiplier * subset_sum, endpoint, trajectory
                )
                rows.append(
                    {
                        "query": query_id,
                        "variant": variant,
                        "prediction_sign": sign_name,
                        "endpoint_percent": float(endpoint_values[0]),
                        "trajectory_percent": float(trajectory_values[0]),
                        "cf_joint_percent": float(joint_values[0]),
                        "score_mean": float(np.mean(scores[variant][query_slot])),
                        "score_std": float(np.std(scores[variant][query_slot])),
                    }
                )
            repeat_values = []
            selected_p1 = []
            for repeat, pair in enumerate(folds):
                fold_values = []
                for train_fold in range(2):
                    train_ids = pair[train_fold]
                    heldout_ids = pair[1 - train_fold]
                    _, _, train_p1 = lds(
                        subset_sum[train_ids],
                        endpoint[train_ids],
                        trajectory[train_ids],
                    )
                    multiplier = 1.0 if float(train_p1[0]) >= 0.0 else -1.0
                    sign_name = "p1" if multiplier > 0 else "m1"
                    heldout_end, heldout_traj, heldout_joint = lds(
                        multiplier * subset_sum[heldout_ids],
                        endpoint[heldout_ids],
                        trajectory[heldout_ids],
                    )
                    fold_values.append(float(heldout_joint[0]))
                    selected_p1.append(multiplier > 0)
                    crossfit_rows.append(
                        {
                            "query": query_id,
                            "variant": variant,
                            "repeat": repeat,
                            "train_fold": train_fold,
                            "selected_sign": sign_name,
                            "train_p1_cf_joint_percent": float(train_p1[0]),
                            "heldout_endpoint_percent": float(heldout_end[0]),
                            "heldout_trajectory_percent": float(heldout_traj[0]),
                            "heldout_cf_joint_percent": float(heldout_joint[0]),
                        }
                    )
                repeat_values.append(float(np.mean(fold_values)))
            repeat_values = np.asarray(repeat_values, dtype=np.float64)
            full_p1 = next(
                row["cf_joint_percent"]
                for row in rows
                if row["query"] == query_id
                and row["variant"] == variant
                and row["prediction_sign"] == "p1"
            )
            crossfit_summary.append(
                {
                    "query": query_id,
                    "variant": variant,
                    "full_p1_cf_joint_percent": float(full_p1),
                    "full_oracle_sign": "p1" if full_p1 >= 0.0 else "m1",
                    "full_oracle_cf_joint_percent": float(abs(full_p1)),
                    "crossfit_cf_joint_mean_percent": float(repeat_values.mean()),
                    "crossfit_cf_joint_std_percent": float(
                        repeat_values.std(ddof=1)
                    ),
                    "crossfit_positive_repeat_fraction": float(
                        np.mean(repeat_values > 0.0)
                    ),
                    "selected_p1_split_fraction": float(np.mean(selected_p1)),
                }
            )

            guided_repeat_values = []
            guided_selected_p1 = []
            guided_linear_values = []
            for repeat, pair in enumerate(folds):
                fold_values = []
                for train_fold in range(2):
                    train_ids = pair[train_fold]
                    heldout_ids = pair[1 - train_fold]
                    _, _, train_linear_joint = lds(
                        linear_subset_prediction[train_ids],
                        endpoint[train_ids],
                        trajectory[train_ids],
                    )
                    # Prespecified hypothesis: square has the opposite orientation
                    # from the signed linear original-f signal.
                    multiplier = -1.0 if float(train_linear_joint[0]) >= 0.0 else 1.0
                    sign_name = "p1" if multiplier > 0 else "m1"
                    heldout_end, heldout_traj, heldout_joint = lds(
                        multiplier * subset_sum[heldout_ids],
                        endpoint[heldout_ids],
                        trajectory[heldout_ids],
                    )
                    fold_values.append(float(heldout_joint[0]))
                    guided_selected_p1.append(multiplier > 0)
                    guided_linear_values.append(float(train_linear_joint[0]))
                    linear_guided_rows.append(
                        {
                            "query": query_id,
                            "variant": variant,
                            "repeat": repeat,
                            "train_fold": train_fold,
                            "train_linear_cf_joint_percent": float(train_linear_joint[0]),
                            "selected_square_sign": sign_name,
                            "heldout_endpoint_percent": float(heldout_end[0]),
                            "heldout_trajectory_percent": float(heldout_traj[0]),
                            "heldout_cf_joint_percent": float(heldout_joint[0]),
                        }
                    )
                guided_repeat_values.append(float(np.mean(fold_values)))
            guided_repeat_values = np.asarray(guided_repeat_values, dtype=np.float64)
            linear_guided_summary.append(
                {
                    "query": query_id,
                    "variant": variant,
                    "train_linear_cf_joint_mean_percent": float(np.mean(guided_linear_values)),
                    "selected_square_p1_split_fraction": float(np.mean(guided_selected_p1)),
                    "crossfit_cf_joint_mean_percent": float(guided_repeat_values.mean()),
                    "crossfit_cf_joint_std_percent": float(guided_repeat_values.std(ddof=1)),
                    "crossfit_positive_repeat_fraction": float(
                        np.mean(guided_repeat_values > 0.0)
                    ),
                }
            )

    write_csv(args.out_dir / "results.csv", rows)
    write_csv(args.out_dir / "crossfit_per_split.csv", crossfit_rows)
    write_csv(args.out_dir / "crossfit_summary.csv", crossfit_summary)
    write_csv(args.out_dir / "linear_guided_crossfit_per_split.csv", linear_guided_rows)
    write_csv(args.out_dir / "linear_guided_crossfit_summary.csv", linear_guided_summary)
    reduction_rows = []
    reduction_banks = {
        # Preserve the historical original-f orientation.
        "linear": {variant: -values for variant, values in linear_scores.items()},
        "square": scores,
        # A single predicted-noise-difference direction has one scalar z per term,
        # so its component root is sqrt(z^2) = abs(z).
        "root": root_scores,
    }
    for reduction, bank in reduction_banks.items():
        for query_slot, query_id in enumerate(query_ids):
            incidence, true = target_data(args, query_id, score_indices)
            endpoint = true[TARGETS[0]]
            trajectory = true[TARGETS[1]]
            for variant in VARIANTS:
                subset_prediction = bank[variant][query_slot] @ incidence.T
                for sign_name, multiplier in (("p1", 1.0), ("m1", -1.0)):
                    endpoint_values, trajectory_values, joint_values = lds(
                        multiplier * subset_prediction,
                        endpoint,
                        trajectory,
                    )
                    reduction_rows.append(
                        {
                            "reduction": reduction,
                            "query": query_id,
                            "variant": variant,
                            "prediction_sign": sign_name,
                            "endpoint_percent": float(endpoint_values[0]),
                            "trajectory_percent": float(trajectory_values[0]),
                            "cf_joint_percent": float(joint_values[0]),
                        }
                    )
    write_csv(args.out_dir / "linear_square_root_results.csv", reduction_rows)
    np.savez_compressed(
        args.out_dir / "product_square_scores.npz",
        query_ids=np.asarray(query_ids, dtype=np.int32),
        score_indices=np.asarray(score_indices, dtype=np.int64),
        term_checkpoints=term_checkpoints,
        term_timesteps=term_timesteps,
        original_term_weights=original_term_weights,
        effective_term_weights=effective_term_weights,
        **{variant: values.astype(np.float32) for variant, values in scores.items()},
    )
    query_label = ",".join(f"Q{query_id}" for query_id in query_ids)
    print(
        f"{query_label} OWN-TRAJECTORY ORIGINAL-F — TERMWISE PRODUCT SQUARE"
    )
    print(f"timestep_weighting={args.timestep_weighting}")
    first_checkpoint = np.flatnonzero(term_checkpoints == np.min(term_checkpoints))
    for query_position, query_id in enumerate(query_ids):
        print(
            f"Q{query_id}_first_checkpoint_timestamp_weights="
            + ",".join(
                f"{int(term_timesteps[index])}:"
                f"{effective_term_weights[query_position, index] / effective_term_weights[query_position, first_checkpoint].sum():.6f}"
                for index in first_checkpoint
            )
        )
    print("S_i = sum_(c,t) weight_(c,t) * product_(c,t,i)^2")
    print("Q VARIANT             SIGN   ENDPOINT      TRAJ  CF JOINT")
    print("-" * 70)
    for row in rows:
        print(
            f"{int(row['query']):1d} {row['variant']:<19s} "
            f"{row['prediction_sign']:>4s} "
            f"{row['endpoint_percent']:+9.3f}% "
            f"{row['trajectory_percent']:+9.3f}% "
            f"{row['cf_joint_percent']:+9.3f}%"
        )

    print(f"\n{len(query_ids)}-QUERY MEAN")
    print("VARIANT             SIGN   ENDPOINT      TRAJ  CF JOINT")
    print("-" * 68)
    for variant in VARIANTS:
        for sign_name in ("m1", "p1"):
            selected = [
                row
                for row in rows
                if row["variant"] == variant
                and row["prediction_sign"] == sign_name
            ]
            print(
                f"{variant:<19s} {sign_name:>4s} "
                f"{np.mean([row['endpoint_percent'] for row in selected]):+9.3f}% "
                f"{np.mean([row['trajectory_percent'] for row in selected]):+9.3f}% "
                f"{np.mean([row['cf_joint_percent'] for row in selected]):+9.3f}%"
            )

    print("\nOVERALL-SIGN CROSSFIT — NO CHECKPOINT/TIMESTAMP FLIPS")
    print("Q VARIANT             FULL BEST SIGN    CV MEAN      STD   CV>0  P1 SELECT")
    print("-" * 82)
    for row in crossfit_summary:
        print(
            f"{int(row['query']):1d} {row['variant']:<19s} "
            f"{row['full_oracle_cf_joint_percent']:+9.3f}% "
            f"{row['full_oracle_sign']:>4s} "
            f"{row['crossfit_cf_joint_mean_percent']:+9.3f}% "
            f"{row['crossfit_cf_joint_std_percent']:8.3f}% "
            f"{row['crossfit_positive_repeat_fraction']:6.2f} "
            f"{row['selected_p1_split_fraction']:9.2f}"
        )

    print(f"\n{len(query_ids)}-QUERY CROSSFIT MEAN")
    print("VARIANT               CV MEAN")
    print("-" * 34)
    for variant in VARIANTS:
        selected = [row for row in crossfit_summary if row["variant"] == variant]
        print(
            f"{variant:<19s} "
            f"{np.mean([row['crossfit_cf_joint_mean_percent'] for row in selected]):+9.3f}%"
        )

    print("\nLINEAR-SIGN-GUIDED SQUARE CROSSFIT — OPPOSITE ORIENTATION")
    print("Q VARIANT             TRAIN LINEAR  SQ P1    CV MEAN      STD   CV>0")
    print("-" * 78)
    for row in linear_guided_summary:
        print(
            f"{int(row['query']):1d} {row['variant']:<19s} "
            f"{row['train_linear_cf_joint_mean_percent']:+11.3f}% "
            f"{row['selected_square_p1_split_fraction']:6.2f} "
            f"{row['crossfit_cf_joint_mean_percent']:+9.3f}% "
            f"{row['crossfit_cf_joint_std_percent']:8.3f}% "
            f"{row['crossfit_positive_repeat_fraction']:6.2f}"
        )

    print(f"\n{len(query_ids)}-QUERY LINEAR-GUIDED CROSSFIT MEAN")
    print("VARIANT               CV MEAN")
    print("-" * 34)
    for variant in VARIANTS:
        selected = [row for row in linear_guided_summary if row["variant"] == variant]
        print(
            f"{variant:<19s} "
            f"{np.mean([row['crossfit_cf_joint_mean_percent'] for row in selected]):+9.3f}%"
        )

    print("\nOWN-TRAJECTORY PREDICTED-NOISE-DIFFERENCE — LINEAR / SQUARE / ROOT")
    print("Q VARIANT             REDUCTION BEST SIGN   ENDPOINT      TRAJ  CF JOINT")
    print("-" * 82)
    for query_id in query_ids:
        for variant in VARIANTS:
            for reduction in ("linear", "square", "root"):
                p1 = next(
                    row
                    for row in reduction_rows
                    if row["query"] == query_id
                    and row["variant"] == variant
                    and row["reduction"] == reduction
                    and row["prediction_sign"] == "p1"
                )
                multiplier = 1.0 if p1["cf_joint_percent"] >= 0.0 else -1.0
                sign_name = "p1" if multiplier > 0.0 else "m1"
                print(
                    f"{query_id:1d} {variant:<19s} {reduction:<9s} {sign_name:>4s} "
                    f"{multiplier * p1['endpoint_percent']:+9.3f}% "
                    f"{multiplier * p1['trajectory_percent']:+9.3f}% "
                    f"{multiplier * p1['cf_joint_percent']:+9.3f}%"
                )
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
