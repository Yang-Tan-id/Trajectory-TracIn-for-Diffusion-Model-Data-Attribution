#!/usr/bin/env python3
"""Summarize Q8 checkpoint/timestep score distributions over 5,000 train points."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from analyze_original_f_checkpoint_sign_crossfit import VARIANTS, write_csv
from analyze_q8_checkpoint_sign_predicted_noise_geometry import majority_signs
from run_expected_residual_jacobian_scores import load_query_bank, train_part_dir


def aggregate(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for values, group in sorted(groups.items()):
        result = dict(zip(keys, values))
        result["terms"] = len(group)
        for metric in (
            "score_mean",
            "score_median",
            "score_std",
            "positive_fraction",
        ):
            samples = np.asarray([row[metric] for row in group], dtype=np.float64)
            result[f"{metric}_mean"] = float(np.mean(samples))
            result[f"{metric}_std"] = float(np.std(samples))
        result["positive_mean_fraction"] = float(
            np.mean([row["score_mean"] > 0.0 for row in group])
        )
        output.append(result)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-id", type=int, default=8)
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
    parser.add_argument("--checkpoint-crossfit-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.skip_predicted = True
    args.shard_index = 0
    args.shard_count = 1
    args.run_id = "q8_train_population_alignment"

    signs, stability, split_count = majority_signs(
        args.checkpoint_crossfit_dir / "per_split.csv",
        args.query_id,
        "query_train_l2",
        "five_bins",
        "checkpoint",
    )
    query_bank, metadata = load_query_bank(
        args,
        args.original_query_namespace,
        "trajectory_next_checkpoint_noise_mse",
        query_ids=[args.query_id],
    )
    term_checkpoints = np.asarray(metadata["ckpt_indices"], dtype=np.int32)
    term_timesteps = np.asarray(metadata["timesteps"], dtype=np.int32)
    term_lookup = {
        (int(checkpoint), int(timestep)): term
        for term, (checkpoint, timestep) in enumerate(
            zip(term_checkpoints, term_timesteps)
        )
    }

    import jax
    import jax.numpy as jnp

    rows = []
    used_terms = 0
    for checkpoint in range(args.num_checkpoints):
        path = train_part_dir(args) / f"ckpt_{checkpoint:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            train_terms = np.asarray(payload["train_features"], dtype=np.float32)
            part_checkpoints = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            part_timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            weights = np.asarray(payload["term_weights"], dtype=np.float64)
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

        for local, (term_checkpoint, timestep, weight) in enumerate(
            zip(part_checkpoints, part_timesteps, weights)
        ):
            term = term_lookup.get((int(term_checkpoint), int(timestep)))
            if term is None:
                continue
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
            checkpoint_index = int(term_checkpoint)
            for variant, value in values.items():
                scores = float(weight) * np.asarray(
                    jax.device_get(value), dtype=np.float64
                )
                rows.append(
                    {
                        "query": args.query_id,
                        "variant": variant,
                        "checkpoint": checkpoint_index + 1,
                        "epoch": 4 * (checkpoint_index + 1),
                        "checkpoint_bin": min(checkpoint_index // 10 + 1, 5),
                        "flip_sign": int(signs[checkpoint_index]),
                        "flip_stability": float(stability[checkpoint_index]),
                        "timestep": int(timestep),
                        "score_mean": float(np.mean(scores)),
                        "score_median": float(np.median(scores)),
                        "score_std": float(np.std(scores)),
                        "positive_fraction": float(np.mean(scores > 0.0)),
                    }
                )
            used_terms += 1
        print(
            f"[score distributions] checkpoint={checkpoint + 1}/50 terms={used_terms}",
            flush=True,
        )

    if used_terms != 490:
        raise ValueError(f"expected 490 terms, found {used_terms}")
    by_bin = aggregate(rows, ("variant", "checkpoint_bin", "flip_sign"))
    by_bin_timestep = aggregate(
        rows, ("variant", "checkpoint_bin", "flip_sign", "timestep")
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_checkpoint_timestep.csv", rows)
    write_csv(args.out_dir / "by_checkpoint_bin.csv", by_bin)
    write_csv(args.out_dir / "by_checkpoint_bin_timestep.csv", by_bin_timestep)

    print(
        f"Q{args.query_id} TRAIN-GRADIENT POPULATION ALIGNMENT; "
        f"flip signs from {split_count} splits"
    )
    print("score_mean = mean_i <g_query(c,t), g_train(c,t,i)> with term weight\n")
    for variant in VARIANTS:
        print(f"{variant.upper()} — BY CHECKPOINT BIN")
        print("BIN FLIP TERMS   SCORE MEAN   MEDIAN MEAN    POS%  MEAN>0")
        for row in by_bin:
            if row["variant"] != variant:
                continue
            print(
                f"{int(row['checkpoint_bin']):3d} {int(row['flip_sign']):+4d} "
                f"{int(row['terms']):5d} {row['score_mean_mean']:+12.5e} "
                f"{row['score_median_mean']:+12.5e} "
                f"{row['positive_fraction_mean']:7.3f} "
                f"{row['positive_mean_fraction']:7.3f}"
            )
        print()

    print("QUERY_TRAIN_L2 — BY BIN AND TIMESTAMP")
    print("BIN FLIP    T   N   SCORE MEAN   MEDIAN MEAN    POS%  MEAN>0")
    for row in sorted(
        (item for item in by_bin_timestep if item["variant"] == "query_train_l2"),
        key=lambda item: (item["checkpoint_bin"], -item["timestep"]),
    ):
        print(
            f"{int(row['checkpoint_bin']):3d} {int(row['flip_sign']):+4d} "
            f"{int(row['timestep']):4d} {int(row['terms']):3d} "
            f"{row['score_mean_mean']:+12.5e} "
            f"{row['score_median_mean']:+12.5e} "
            f"{row['positive_fraction_mean']:7.3f} "
            f"{row['positive_mean_fraction']:7.3f}"
        )
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
