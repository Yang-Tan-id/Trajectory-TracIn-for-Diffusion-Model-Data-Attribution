#!/usr/bin/env python3
"""Evaluate 24-probe next-delta original-f with a parameter-space probe.

The unique next-checkpoint predicted-noise difference is reconstructed from
24 output-space probes.  For every checkpoint/timestep term, replace the
usual parameter-gradient product

    <g_train, g_query>

with

    <g_train, v> <g_query, v>.

with a rank-one parameter-space estimate.  Rademacher parameter probes are
used, so the latter is an unbiased estimator of the former.  The same
parameter probe is shared by all train examples, queries, and 24 output
probes for a term; different terms receive independent parameter probes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from analyze_original_f_timestamp_sign_crossfit import TARGETS, lds
from analyze_nearest_train_probe_predicted_noise_relation import (
    probe_alignment_matrix,
)
from analyze_predicted_noise_probe24_output_alignment import load_bank
from analyze_predicted_noise_probe24_term_winners import (
    FRESH_PATTERN,
    ORIGINAL_PATTERN,
    load_features,
    target_data,
)
from run_predicted_noise_jvp_l2_squared import train_part_dir


VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")
REDUCTIONS = ("linear", "square")


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def folds_for(n: int, repeats: int, seed: int):
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(repeats):
        permutation = rng.permutation(n)
        output.append((permutation[::2], permutation[1::2]))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--num-checkpoints", type=int, default=50)
    parser.add_argument("--num-parameter-probes", type=int, default=1)
    parser.add_argument("--probe-seed", type=int, default=20260917)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260916)
    parser.add_argument(
        "--original-namespace", default="predicted_noise_output_next_original12"
    )
    parser.add_argument(
        "--fresh-namespace", default="predicted_noise_output_next_fresh12"
    )
    parser.add_argument("--checkpoint-direction", default="next")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.num_parameter_probes <= 0:
        raise ValueError("--num-parameter-probes must be positive")

    query_ids = parse_ints(args.query_ids)
    args.query_ids = query_ids

    original, metadata = load_features(args, ORIGINAL_PATTERN, range(12))
    fresh, fresh_metadata = load_features(args, FRESH_PATTERN, range(12))
    for key, value in metadata.items():
        if not np.array_equal(value, fresh_metadata[key]):
            raise ValueError(f"original/fresh metadata mismatch for {key}")
    output_probe_bank = np.concatenate((original, fresh), axis=0)
    if output_probe_bank.shape[0] != 24:
        raise ValueError(f"expected 24 output probes, got {output_probe_bank.shape[0]}")
    alignment_banks = {
        (query_id, bank): load_bank(args, query_id, bank)
        for query_id in query_ids
        for bank in ("original", "fresh")
    }
    checkpoints = np.asarray(metadata["ckpt_indices"], dtype=np.int32)
    timesteps = np.asarray(metadata["timesteps"], dtype=np.int32)
    # Probe banks include the reference/final checkpoint, but next-delta has no
    # target there.  Keep only the 49 checkpoints that have a successor.
    valid_terms = checkpoints < int(np.max(checkpoints))
    term_weights = np.full(len(checkpoints), np.nan, dtype=np.float64)
    lookup = {
        (int(checkpoint), int(timestep)): term
        for term, (checkpoint, timestep) in enumerate(zip(checkpoints, timesteps))
        if valid_terms[term]
    }

    banks = {
        reduction: {
            variant: np.zeros((len(query_ids), 5000), dtype=np.float64)
            for variant in VARIANTS
        }
        for reduction in REDUCTIONS
    }
    score_indices = None
    used_terms = 0
    feature_dimension = int(output_probe_bank.shape[-1])

    for checkpoint in range(args.num_checkpoints):
        path = train_part_dir(args.experiment, args.train_seed) / f"ckpt_{checkpoint:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            train_terms = np.asarray(payload["train_features"], dtype=np.float32)
            part_checkpoints = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            part_timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            part_weights = np.asarray(payload["term_weights"], dtype=np.float64)
            indices = np.asarray(payload["score_indices"], dtype=np.int64)
        if train_terms.shape[-1] != feature_dimension:
            raise ValueError(
                f"{path}: feature dimension {train_terms.shape[-1]} != {feature_dimension}"
            )
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
            if np.isnan(term_weights[term]):
                term_weights[term] = float(part_weights[local])
            elif not np.isclose(
                term_weights[term], float(part_weights[local]), rtol=1e-6, atol=1e-12
            ):
                raise ValueError(f"term-weight mismatch for term {term}")
            train = np.asarray(train_terms[local], dtype=np.float64)
            probe_queries = np.asarray(
                output_probe_bank[:, :, term, :], dtype=np.float64
            )
            delta_scalars = np.stack(
                [
                    probe_alignment_matrix(
                        alignment_banks,
                        query_id,
                        int(term_checkpoint) + 1,
                        int(timestep),
                    )["projection_on_next_predicted_noise_delta"]
                    for query_id in query_ids
                ],
                axis=0,
            )
            # (Q, 24, D) weighted by the one reference delta's 24 projections.
            query = np.mean(
                np.transpose(probe_queries, (1, 0, 2))
                * delta_scalars[:, :, None],
                axis=1,
                dtype=np.float64,
            )
            train_norm = np.linalg.norm(train, axis=1) + 1e-8
            query_norm = np.linalg.norm(query, axis=1) + 1e-8

            # Seed by semantic coordinates so results do not depend on traversal order.
            seed_sequence = np.random.SeedSequence(
                [args.probe_seed, int(term_checkpoint), int(timestep)]
            )
            rng = np.random.default_rng(seed_sequence)
            probes = rng.integers(
                0,
                2,
                size=(args.num_parameter_probes, feature_dimension),
                dtype=np.int8,
            ).astype(np.float64)
            probes = 2.0 * probes - 1.0
            train_projection = train @ probes.T
            query_projection = query @ probes.T
            rank_one = np.einsum(
                "ir,qr->iq", train_projection, query_projection
            ) / float(args.num_parameter_probes)
            rank_one_square = np.einsum(
                "ir,qr->iq",
                np.square(train_projection),
                np.square(query_projection),
            ) / float(args.num_parameter_probes)
            values = {
                "raw": (rank_one, rank_one_square),
                "query_l2": (
                    rank_one / query_norm[None, :],
                    rank_one_square / np.square(query_norm[None, :]),
                ),
                "train_l2": (
                    rank_one / train_norm[:, None],
                    rank_one_square / np.square(train_norm[:, None]),
                ),
                "query_train_l2": (
                    rank_one / (train_norm[:, None] * query_norm[None, :]),
                    rank_one_square
                    / (
                        np.square(train_norm[:, None])
                        * np.square(query_norm[None, :])
                    ),
                ),
            }
            weight = float(term_weights[term])
            for variant, (linear, square) in values.items():
                # Match the historical original-f orientation for signed linear.
                banks["linear"][variant] += -weight * linear.T
                banks["square"][variant] += weight * square.T
            used_terms += 1
        print(
            f"[parameter probe] checkpoint={checkpoint + 1}/{args.num_checkpoints} "
            f"terms={used_terms}",
            flush=True,
        )

    expected_terms = int(np.sum(valid_terms))
    if score_indices is None or used_terms != expected_terms:
        raise ValueError(f"expected {expected_terms} terms, found {used_terms}")
    if np.any(np.isnan(term_weights[valid_terms])):
        raise ValueError("one or more valid next-delta terms are missing weights")

    full_rows = []
    split_rows = []
    summary_rows = []
    all_targets = target_data(args, score_indices)
    for query_slot, query_id in enumerate(query_ids):
        incidence, true = all_targets[query_id]
        endpoint = true[TARGETS[0]]
        trajectory = true[TARGETS[1]]
        folds = folds_for(len(endpoint), args.repeats, args.random_seed)
        for reduction in REDUCTIONS:
            for variant in VARIANTS:
                subset_prediction = banks[reduction][variant][query_slot] @ incidence.T
                end, traj, joint = lds(
                    subset_prediction, endpoint, trajectory
                )
                full_p1 = float(joint[0])
                for sign, multiplier in (("p1", 1.0), ("m1", -1.0)):
                    signed_end, signed_traj, signed_joint = lds(
                        multiplier * subset_prediction, endpoint, trajectory
                    )
                    full_rows.append(
                        {
                            "reduction": reduction,
                            "query": query_id,
                            "variant": variant,
                            "sign": sign,
                            "endpoint_percent": float(signed_end[0]),
                            "trajectory_percent": float(signed_traj[0]),
                            "cf_joint_percent": float(signed_joint[0]),
                        }
                    )

                repeat_values = []
                selected_p1 = []
                for repeat, pair in enumerate(folds):
                    heldout_values = []
                    for train_fold in range(2):
                        train_ids = pair[train_fold]
                        heldout_ids = pair[1 - train_fold]
                        _, _, train_joint = lds(
                            subset_prediction[train_ids],
                            endpoint[train_ids],
                            trajectory[train_ids],
                        )
                        multiplier = 1.0 if float(train_joint[0]) >= 0.0 else -1.0
                        heldout_end, heldout_traj, heldout_joint = lds(
                            multiplier * subset_prediction[heldout_ids],
                            endpoint[heldout_ids],
                            trajectory[heldout_ids],
                        )
                        value = float(heldout_joint[0])
                        heldout_values.append(value)
                        selected_p1.append(multiplier > 0.0)
                        split_rows.append(
                            {
                                "reduction": reduction,
                                "query": query_id,
                                "variant": variant,
                                "repeat": repeat,
                                "train_fold": train_fold,
                                "selected_sign": "p1" if multiplier > 0 else "m1",
                                "train_p1_cf_joint_percent": float(train_joint[0]),
                                "heldout_endpoint_percent": float(heldout_end[0]),
                                "heldout_trajectory_percent": float(heldout_traj[0]),
                                "heldout_cf_joint_percent": value,
                            }
                        )
                    repeat_values.append(float(np.mean(heldout_values)))
                repeat_values = np.asarray(repeat_values, dtype=np.float64)
                summary_rows.append(
                    {
                        "reduction": reduction,
                        "query": query_id,
                        "variant": variant,
                        "full_p1_cf_joint_percent": full_p1,
                        "full_best_sign": "p1" if full_p1 >= 0 else "m1",
                        "full_oracle_cf_joint_percent": abs(full_p1),
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

    write_csv(args.out_dir / "results.csv", full_rows)
    write_csv(args.out_dir / "crossfit_per_split.csv", split_rows)
    write_csv(args.out_dir / "crossfit_summary.csv", summary_rows)
    np.savez_compressed(
        args.out_dir / "scores.npz",
        query_ids=np.asarray(query_ids, dtype=np.int32),
        score_indices=np.asarray(score_indices, dtype=np.int64),
        checkpoints=checkpoints,
        timesteps=timesteps,
        term_weights=term_weights,
        num_output_probes=np.asarray(24, dtype=np.int32),
        num_parameter_probes=np.asarray(args.num_parameter_probes, dtype=np.int32),
        probe_seed=np.asarray(args.probe_seed, dtype=np.int64),
        **{
            f"{reduction}_{variant}": values.astype(np.float32)
            for reduction, reduction_bank in banks.items()
            for variant, values in reduction_bank.items()
        },
    )

    print(
        "NEXT-DELTA ORIGINAL-F PARAMETER-PROBE: "
        "g_query=mean_24[(delta_eps^T v_r) J^T v_r]; "
        "z=(g_train^T u)(g_query^T u)"
    )
    print(
        f"output_probes=24 parameter_probes={args.num_parameter_probes} "
        f"seed={args.probe_seed} "
        f"terms={used_terms} feature_dim={feature_dimension}"
    )
    print("REDUCTION Q VARIANT             FULL BEST   CV MEAN      STD   CV>0")
    print("-" * 76)
    for row in summary_rows:
        print(
            f"{row['reduction']:<9s} "
            f"{int(row['query']):1d} {row['variant']:<19s} "
            f"{row['full_oracle_cf_joint_percent']:+8.3f}% "
            f"{row['full_best_sign']:>4s} "
            f"{row['crossfit_cf_joint_mean_percent']:+9.3f}% "
            f"{row['crossfit_cf_joint_std_percent']:8.3f}% "
            f"{row['crossfit_positive_repeat_fraction']:6.2f}"
        )

    print("\n10-QUERY CROSSFIT MEAN")
    print("REDUCTION VARIANT               CV MEAN")
    print("-" * 46)
    for reduction in REDUCTIONS:
        for variant in VARIANTS:
            selected = [
                row
                for row in summary_rows
                if row["reduction"] == reduction and row["variant"] == variant
            ]
            print(
                f"{reduction:<9s} {variant:<19s} "
                f"{np.mean([row['crossfit_cf_joint_mean_percent'] for row in selected]):+9.3f}%"
            )

    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
