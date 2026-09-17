#!/usr/bin/env python3
"""Relate cached own-trajectory linear scores to next-checkpoint endpoint motion."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from analyze_linear_sign_endpoint_trajectory_geometry import geometry_artifact
from analyze_predicted_noise_probe24_output_alignment import write_csv
from analyze_q1348_own_trajectory_product_square import parse_ints
from run_expected_residual_jacobian_scores import load_query_bank, train_part_dir


VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")


def correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    output = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        output[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return output


def oracle_signs(path, variant):
    signs = {}
    strengths = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["reduction"] == "linear"
                and row["variant"] == variant
                and row["prediction_sign"] == "p1"
            ):
                value = float(row["cf_joint_percent"])
                query = int(row["query"])
                signs[query] = 1.0 if value >= 0.0 else -1.0
                strengths[query] = abs(value)
    return signs, strengths


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
        "--train-feature-semantics",
        default="raw_projected_expected_loss_gradient",
    )
    parser.add_argument("--score-results", type=Path, required=True)
    parser.add_argument("--sign-variant", default="query_l2", choices=VARIANTS)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.skip_predicted = True
    args.shard_index = 0
    args.shard_count = 1
    args.run_id = "linear_checkpoint_score_endpoint_updates"
    args.namespace = args.query_namespace
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
    lookup = {
        (int(checkpoint), int(timestep)): term
        for term, (checkpoint, timestep) in enumerate(
            zip(term_checkpoints, term_timesteps)
        )
    }
    unique_checkpoints = np.unique(term_checkpoints)
    checkpoint_position = {
        int(checkpoint): position
        for position, checkpoint in enumerate(unique_checkpoints)
    }
    scores = {
        variant: np.zeros(
            (len(query_ids), len(unique_checkpoints), 5000), dtype=np.float64
        )
        for variant in VARIANTS
    }

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
        for local, (term_checkpoint, timestep) in enumerate(
            zip(part_checkpoints, part_timesteps)
        ):
            term = lookup.get((int(term_checkpoint), int(timestep)))
            if term is None:
                continue
            position = checkpoint_position[int(term_checkpoint)]
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
                # Preserve the original-f linear orientation.
                scores[variant][:, position] -= (
                    term_weights[term]
                    * np.asarray(jax.device_get(value), dtype=np.float64).T
                )
            used_terms += 1
        print(
            f"[linear components] checkpoint={checkpoint + 1}/"
            f"{args.num_checkpoints} terms={used_terms}",
            flush=True,
        )
    if used_terms != len(term_checkpoints):
        raise ValueError(f"expected {len(term_checkpoints)} terms, found {used_terms}")

    signs, strengths = oracle_signs(args.score_results, args.sign_variant)
    required = (
        "checkpoint_own_trajectory_endpoints",
        "checkpoint_own_trajectory_reference_endpoint",
    )
    detail_rows = []
    summary_rows = []
    for query_position, query in enumerate(query_ids):
        path = geometry_artifact(args, query, required)
        with np.load(path, allow_pickle=False) as payload:
            endpoints = np.asarray(
                payload["checkpoint_own_trajectory_endpoints"], dtype=np.float64
            )
            reference = np.asarray(
                payload["checkpoint_own_trajectory_reference_endpoint"],
                dtype=np.float64,
            )
        distances = np.sqrt(
            np.mean(np.square(endpoints - reference[None, ...]), axis=tuple(range(1, endpoints.ndim)))
        )
        if len(distances) != len(unique_checkpoints):
            raise ValueError(
                f"Q{query} endpoint count {len(distances)} != "
                f"{len(unique_checkpoints)}"
            )
        improvement = distances[:-1] - distances[1:]
        sign = signs[query]
        sign_name = "p1" if sign > 0 else "m1"
        for variant in VARIANTS:
            # Sum over all training examples: the checkpoint's full-training-set
            # scalar prediction. Divide by N only for readable scale.
            prediction = scores[variant][query_position].mean(axis=1)
            current = prediction[:-1]
            centered = current - current.mean()
            oriented = sign * current
            oriented_centered = oriented - oriented.mean()
            pearson = correlation(current, improvement)
            spearman = correlation(ranks(current), ranks(improvement))
            oriented_pearson = correlation(oriented, improvement)
            oriented_spearman = correlation(ranks(oriented), ranks(improvement))
            agreement = float(np.mean(np.sign(centered) == np.sign(improvement)))
            oriented_agreement = float(
                np.mean(np.sign(oriented_centered) == np.sign(improvement))
            )
            amplitude_correlation = correlation(
                np.abs(centered), np.abs(improvement)
            )
            summary_rows.append(
                {
                    "query": query,
                    "oracle_sign": sign_name,
                    "abs_lds_percent": strengths[query],
                    "variant": variant,
                    "pearson_score_vs_next_improvement": pearson,
                    "spearman_score_vs_next_improvement": spearman,
                    "centered_sign_agreement": agreement,
                    "oracle_oriented_pearson": oriented_pearson,
                    "oracle_oriented_spearman": oriented_spearman,
                    "oracle_oriented_centered_sign_agreement": oriented_agreement,
                    "amplitude_correlation": amplitude_correlation,
                }
            )
            if variant == args.sign_variant:
                for position, value in enumerate(current):
                    detail_rows.append(
                        {
                            "query": query,
                            "oracle_sign": sign_name,
                            "checkpoint": int(unique_checkpoints[position]),
                            "epoch": 4 * (int(unique_checkpoints[position]) + 1),
                            "linear_score_mean": float(value),
                            "centered_linear_score": float(centered[position]),
                            "oracle_oriented_centered_score": float(
                                oriented_centered[position]
                            ),
                            "endpoint_rmse_current": float(distances[position]),
                            "endpoint_rmse_next": float(distances[position + 1]),
                            "next_endpoint_improvement": float(improvement[position]),
                            "centered_sign_match": int(
                                np.sign(centered[position])
                                == np.sign(improvement[position])
                            ),
                            "oriented_sign_match": int(
                                np.sign(oriented_centered[position])
                                == np.sign(improvement[position])
                            ),
                        }
                    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_query_checkpoint.csv", detail_rows)
    write_csv(args.out_dir / "per_query_summary.csv", summary_rows)
    selected = [row for row in summary_rows if row["variant"] == args.sign_variant]
    print(
        f"LINEAR {args.sign_variant.upper()} CHECKPOINT SCORE vs "
        "NEXT ENDPOINT IMPROVEMENT"
    )
    print("Q SIGN |LDS| PEARSON SPEARMAN MATCH ORIENT-P ORIENT-S ORIENT-M AMP-R")
    print("-" * 84)
    for row in selected:
        print(
            f"{int(row['query']):1d} {row['oracle_sign']:>4s} "
            f"{row['abs_lds_percent']:5.2f}% "
            f"{row['pearson_score_vs_next_improvement']:+7.3f} "
            f"{row['spearman_score_vs_next_improvement']:+8.3f} "
            f"{row['centered_sign_agreement']:5.2f} "
            f"{row['oracle_oriented_pearson']:+8.3f} "
            f"{row['oracle_oriented_spearman']:+8.3f} "
            f"{row['oracle_oriented_centered_sign_agreement']:8.2f} "
            f"{row['amplitude_correlation']:+6.3f}"
        )
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
