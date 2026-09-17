#!/usr/bin/env python3
"""Diagnose norm weighting and per-component LDS for an own-trajectory query."""

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
from run_expected_residual_jacobian_scores import load_query_bank, train_part_dir


VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def joint(prediction, endpoint, trajectory) -> float:
    return float(lds(prediction, endpoint, trajectory)[2][0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-id", type=int, default=3)
    parser.add_argument("--num-checkpoints", type=int, default=50)
    parser.add_argument("--train-namespace", default="traj_tracin")
    parser.add_argument(
        "--original-query-namespace",
        default="loss_direction_original_f_checkpoint_own_trajectory",
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
    args.run_id = "q3_component_norm_lds"

    query_bank, metadata = load_query_bank(
        args,
        args.original_query_namespace,
        "trajectory_next_checkpoint_noise_mse",
        query_ids=[args.query_id],
    )
    term_checkpoints = np.asarray(metadata["ckpt_indices"], dtype=np.int32)
    term_timesteps = np.asarray(metadata["timesteps"], dtype=np.int32)
    term_weights = np.asarray(metadata["term_weights"], dtype=np.float64)
    lookup = {
        (int(checkpoint), int(timestep)): term
        for term, (checkpoint, timestep) in enumerate(
            zip(term_checkpoints, term_timesteps)
        )
    }
    num_terms = len(term_checkpoints)
    if num_terms != 490:
        raise ValueError(f"expected 490 query terms, found {num_terms}")

    score_indices = None
    incidence = endpoint = trajectory = None
    component_predictions = None
    square_component_predictions = None
    query_norms = np.full(num_terms, np.nan, dtype=np.float64)
    train_norm_means = np.full(num_terms, np.nan, dtype=np.float64)
    train_norm_medians = np.full(num_terms, np.nan, dtype=np.float64)

    import jax
    import jax.numpy as jnp

    used_terms = 0
    for checkpoint in range(args.num_checkpoints):
        path = train_part_dir(args) / f"ckpt_{checkpoint:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            train_terms = np.asarray(payload["train_features"], dtype=np.float32)
            part_checkpoints = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            part_timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            part_weights = np.asarray(payload["term_weights"], dtype=np.float64)
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
            incidence, true = target_data(args, args.query_id, score_indices)
            endpoint = true[TARGETS[0]]
            trajectory = true[TARGETS[1]]
            component_predictions = {
                variant: np.zeros((num_terms, incidence.shape[0]), dtype=np.float64)
                for variant in VARIANTS
            }
            square_component_predictions = {
                variant: np.zeros((num_terms, incidence.shape[0]), dtype=np.float64)
                for variant in VARIANTS
            }
        elif not np.array_equal(score_indices, indices):
            raise ValueError(f"score indices mismatch: {path}")

        for local, (part_checkpoint, timestep, weight) in enumerate(
            zip(part_checkpoints, part_timesteps, part_weights)
        ):
            term = lookup.get((int(part_checkpoint), int(timestep)))
            if term is None:
                continue
            if not np.isclose(weight, term_weights[term], rtol=1e-6, atol=1e-12):
                raise ValueError(f"term-weight mismatch for term {term}")
            train = jax.device_put(jnp.asarray(train_terms[local]))
            query = jax.device_put(jnp.asarray(query_bank[0, term]))
            dots = train @ query
            train_norm = jnp.linalg.norm(train, axis=1) + 1e-8
            query_norm = jnp.linalg.norm(query) + 1e-8
            values = {
                "raw": dots,
                "query_l2": dots / query_norm,
                "train_l2": dots / train_norm,
                "query_train_l2": dots / (train_norm * query_norm),
            }
            query_norms[term] = float(jax.device_get(query_norm))
            host_train_norm = np.asarray(jax.device_get(train_norm), dtype=np.float64)
            train_norm_means[term] = float(host_train_norm.mean())
            train_norm_medians[term] = float(np.median(host_train_norm))
            for variant, value in values.items():
                datapoint_score = float(weight) * np.asarray(
                    jax.device_get(value), dtype=np.float64
                )
                component_predictions[variant][term] = -(datapoint_score @ incidence.T)
                square_datapoint_score = float(weight) * np.square(
                    np.asarray(jax.device_get(value), dtype=np.float64)
                )
                square_component_predictions[variant][term] = (
                    square_datapoint_score @ incidence.T
                )
            used_terms += 1
        print(
            f"[components] checkpoint={checkpoint + 1}/50 terms={used_terms}",
            flush=True,
        )

    if (
        used_terms != num_terms
        or component_predictions is None
        or square_component_predictions is None
    ):
        raise ValueError(f"expected {num_terms} terms, found {used_terms}")
    if np.isnan(query_norms).any():
        raise ValueError("some query norms were not populated")

    totals = {variant: values.sum(axis=0) for variant, values in component_predictions.items()}
    total_joint = {
        variant: joint(values, endpoint, trajectory) for variant, values in totals.items()
    }
    square_totals = {
        variant: values.sum(axis=0)
        for variant, values in square_component_predictions.items()
    }
    square_p1_joint = {
        variant: joint(values, endpoint, trajectory)
        for variant, values in square_totals.items()
    }
    square_oracle_multiplier = {
        variant: 1.0 if value >= 0.0 else -1.0
        for variant, value in square_p1_joint.items()
    }
    rows = []
    for term in range(num_terms):
        row = {
            "term": term,
            "checkpoint_index": int(term_checkpoints[term]),
            "checkpoint": int(term_checkpoints[term]) + 1,
            "epoch": 4 * (int(term_checkpoints[term]) + 1),
            "timestep": int(term_timesteps[term]),
            "weight": float(term_weights[term]),
            "query_norm": float(query_norms[term]),
            "train_norm_mean": float(train_norm_means[term]),
            "train_norm_median": float(train_norm_medians[term]),
        }
        for variant in VARIANTS:
            component = component_predictions[variant][term]
            standalone = joint(component, endpoint, trajectory)
            without = joint(totals[variant] - component, endpoint, trajectory)
            row[f"{variant}_standalone_joint_percent"] = standalone
            row[f"{variant}_leave_one_out_delta_percent"] = total_joint[variant] - without
            square_component = square_component_predictions[variant][term]
            multiplier = square_oracle_multiplier[variant]
            square_standalone = joint(
                multiplier * square_component, endpoint, trajectory
            )
            square_without = joint(
                multiplier * (square_totals[variant] - square_component),
                endpoint,
                trajectory,
            )
            row[f"{variant}_square_oriented_standalone_joint_percent"] = (
                square_standalone
            )
            row[f"{variant}_square_oriented_leave_one_out_delta_percent"] = (
                abs(square_p1_joint[variant]) - square_without
            )
        rows.append(row)

    write_csv(args.out_dir / "per_component.csv", rows)
    np.savez_compressed(
        args.out_dir / "component_predictions.npz",
        query_id=np.asarray(args.query_id, dtype=np.int32),
        ckpt_indices=term_checkpoints,
        timesteps=term_timesteps,
        term_weights=term_weights,
        query_norms=query_norms,
        train_norm_means=train_norm_means,
        train_norm_medians=train_norm_medians,
        **component_predictions,
        **{
            f"square_{variant}": values
            for variant, values in square_component_predictions.items()
        },
    )

    print(f"Q{args.query_id} OWN-TRAJECTORY COMPONENT NORM/LDS DIAGNOSTIC")
    print("TOTAL LINEAR ORIGINAL-F")
    for variant in VARIANTS:
        print(f"{variant:<19s} {total_joint[variant]:+9.3f}%")

    print("\nTOTAL PRODUCT-SQUARE (FULL-DATA ORIENTED FOR DIAGNOSIS)")
    print("VARIANT               P1 LDS  BEST SIGN  ORIENTED LDS")
    print("-" * 57)
    for variant in VARIANTS:
        sign = "p1" if square_oracle_multiplier[variant] > 0 else "m1"
        print(
            f"{variant:<19s} {square_p1_joint[variant]:+9.3f}% "
            f"{sign:>9s} {abs(square_p1_joint[variant]):+12.3f}%"
        )

    for variant in VARIANTS:
        key = f"{variant}_leave_one_out_delta_percent"
        ranked = sorted(rows, key=lambda row: abs(float(row[key])), reverse=True)[:15]
        print(f"\nTOP 15 |LEAVE-ONE-OUT DELTA| — {variant}")
        print("CKPT EPOCH    T      QNORM    TRAINN   STANDALONE  LOO DELTA")
        print("-" * 76)
        for row in ranked:
            print(
                f"{int(row['checkpoint']):4d} {int(row['epoch']):5d} "
                f"{int(row['timestep']):4d} "
                f"{float(row['query_norm']):10.4e} "
                f"{float(row['train_norm_mean']):9.4e} "
                f"{float(row[f'{variant}_standalone_joint_percent']):+10.3f}% "
                f"{float(row[key]):+9.3f}%"
            )

    for variant in VARIANTS:
        key = f"{variant}_square_oriented_leave_one_out_delta_percent"
        ranked = sorted(rows, key=lambda row: abs(float(row[key])), reverse=True)[:15]
        sign = "p1" if square_oracle_multiplier[variant] > 0 else "m1"
        print(f"\nTOP 15 |SQUARE ORIENTED LOO DELTA| — {variant} ({sign})")
        print("CKPT EPOCH    T      QNORM    TRAINN   STANDALONE  LOO DELTA")
        print("-" * 76)
        for row in ranked:
            print(
                f"{int(row['checkpoint']):4d} {int(row['epoch']):5d} "
                f"{int(row['timestep']):4d} "
                f"{float(row['query_norm']):10.4e} "
                f"{float(row['train_norm_mean']):9.4e} "
                f"{float(row[f'{variant}_square_oriented_standalone_joint_percent']):+10.3f}% "
                f"{float(row[key]):+9.3f}%"
            )

    print("\nCORRELATION WITH |LEAVE-ONE-OUT DELTA|")
    print("VARIANT               log QNORM  log TRAINN")
    print("-" * 49)
    log_query = np.log(query_norms)
    log_train = np.log(train_norm_means)
    for variant in VARIANTS:
        delta = np.abs(
            np.asarray(
                [row[f"{variant}_leave_one_out_delta_percent"] for row in rows],
                dtype=np.float64,
            )
        )
        print(
            f"{variant:<19s} "
            f"{np.corrcoef(log_query, delta)[0, 1]:+10.4f} "
            f"{np.corrcoef(log_train, delta)[0, 1]:+11.4f}"
        )
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
