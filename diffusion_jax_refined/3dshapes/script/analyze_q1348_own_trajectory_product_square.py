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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-ids", default="1,3,4,8")
    parser.add_argument("--num-checkpoints", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260916)
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
                squared = np.asarray(jax.device_get(jnp.square(value)), dtype=np.float64)
                scores[variant] += float(weight) * squared.T
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

    write_csv(args.out_dir / "results.csv", rows)
    write_csv(args.out_dir / "crossfit_per_split.csv", crossfit_rows)
    write_csv(args.out_dir / "crossfit_summary.csv", crossfit_summary)
    np.savez_compressed(
        args.out_dir / "product_square_scores.npz",
        query_ids=np.asarray(query_ids, dtype=np.int32),
        score_indices=np.asarray(score_indices, dtype=np.int64),
        **{variant: values.astype(np.float32) for variant, values in scores.items()},
    )
    print("Q1,Q3,Q4,Q8 OWN-TRAJECTORY ORIGINAL-F — TERMWISE PRODUCT SQUARE")
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

    print("\nFOUR-QUERY MEAN")
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

    print("\nFOUR-QUERY CROSSFIT MEAN")
    print("VARIANT               CV MEAN")
    print("-" * 34)
    for variant in VARIANTS:
        selected = [row for row in crossfit_summary if row["variant"] == variant]
        print(
            f"{variant:<19s} "
            f"{np.mean([row['crossfit_cf_joint_mean_percent'] for row in selected]):+9.3f}%"
        )
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
