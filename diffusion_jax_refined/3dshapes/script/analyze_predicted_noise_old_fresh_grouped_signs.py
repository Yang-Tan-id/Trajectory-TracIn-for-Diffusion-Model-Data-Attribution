#!/usr/bin/env python3
"""Compare old/fresh linear signs after grouping by timestamp or checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT / "script") not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT / "script"))

from analyze_predicted_noise_old_fresh_term_signs import (
    BANKS,
    atomic_savez,
    load_effective_query_bank,
    parse_query_ids,
    score_path,
    write_csv,
)
from run_predicted_noise_jvp_l2_squared import train_part_dir


def shard(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp

    output = args.out_dir / f"grouped_shard_{args.shard_index}.npz"
    if output.is_file():
        print(f"[skip] grouped shard exists: {output}", flush=True)
        return

    old_query, meta = load_effective_query_bank(args, BANKS["old12"]["query_pattern"])
    fresh_query, fresh_meta = load_effective_query_bank(
        args, BANKS["fresh12"]["query_pattern"]
    )
    for key in meta:
        if not np.array_equal(meta[key], fresh_meta[key]):
            raise ValueError(f"old/fresh query metadata mismatch for {key}")
    lookup = {
        (int(ckpt), int(timestep)): term
        for term, (ckpt, timestep) in enumerate(
            zip(meta["ckpt_indices"], meta["timesteps"])
        )
    }
    unique_timesteps = np.asarray(
        list(dict.fromkeys(int(value) for value in meta["timesteps"])),
        dtype=np.int32,
    )
    if len(unique_timesteps) != 10:
        raise ValueError(f"expected 10 timestamps, got {unique_timesteps}")
    timestep_slots = {int(value): slot for slot, value in enumerate(unique_timesteps)}

    shape = (len(args.query_ids), 5000)
    old_timestamp = np.zeros((10, *shape), dtype=np.float64)
    fresh_timestamp = np.zeros((10, *shape), dtype=np.float64)
    old_checkpoint = np.zeros((50, *shape), dtype=np.float64)
    fresh_checkpoint = np.zeros((50, *shape), dtype=np.float64)
    score_indices = None

    for ckpt_i in range(args.shard_index, 50, args.shard_count):
        path = train_part_dir(args.experiment, args.train_seed) / f"ckpt_{ckpt_i:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            train = np.asarray(payload["train_features"], dtype=np.float32)
            indices = np.asarray(payload["score_indices"], dtype=np.int64)
            ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            weights = np.asarray(payload["term_weights"], dtype=np.float64)
        if score_indices is None:
            score_indices = indices
        elif not np.array_equal(score_indices, indices):
            raise ValueError(f"score indices differ in {path}")

        for local_term, (ckpt, timestep, weight) in enumerate(
            zip(ckpts, timesteps, weights)
        ):
            term = lookup[(int(ckpt), int(timestep))]
            train_device = jax.device_put(jnp.asarray(train[local_term]))
            train_unit = train_device / jnp.maximum(
                jnp.linalg.norm(train_device, axis=1, keepdims=True), 1e-8
            )
            combined_query = np.concatenate(
                (old_query[:, term], fresh_query[:, term]), axis=0
            )
            values = np.asarray(
                jax.device_get(train_unit @ jnp.asarray(combined_query).T),
                dtype=np.float32,
            ).T
            old = values[: len(args.query_ids)]
            fresh = values[len(args.query_ids) :]
            slot = timestep_slots[int(timestep)]
            old_timestamp[slot] += float(weight) * old
            fresh_timestamp[slot] += float(weight) * fresh
            old_checkpoint[int(ckpt)] += float(weight) * old
            fresh_checkpoint[int(ckpt)] += float(weight) * fresh
        print(f"[grouped scores] checkpoint={ckpt_i + 1}/50", flush=True)

    assert score_indices is not None
    atomic_savez(
        output,
        old_timestamp=old_timestamp,
        fresh_timestamp=fresh_timestamp,
        old_checkpoint=old_checkpoint,
        fresh_checkpoint=fresh_checkpoint,
        query_ids=np.asarray(args.query_ids, dtype=np.int32),
        timesteps=unique_timesteps,
        checkpoint_indices=np.arange(50, dtype=np.int32),
        score_indices=score_indices,
    )
    print(f"[saved] {output}", flush=True)


def sign_summary(old: np.ndarray, fresh: np.ndarray) -> dict[str, float]:
    old_pos = np.asarray(old) > 0.0
    fresh_pos = np.asarray(fresh) > 0.0
    return {
        "old_positive_fraction": float(old_pos.mean()),
        "fresh_positive_fraction": float(fresh_pos.mean()),
        "both_positive_fraction": float(np.mean(old_pos & fresh_pos)),
        "old_positive_fresh_negative_fraction": float(np.mean(old_pos & ~fresh_pos)),
        "old_negative_fresh_positive_fraction": float(np.mean(~old_pos & fresh_pos)),
        "both_negative_fraction": float(np.mean(~old_pos & ~fresh_pos)),
        "same_sign_fraction": float(np.mean(old_pos == fresh_pos)),
    }


def paired_pearson(old: np.ndarray, fresh: np.ndarray) -> float:
    old = np.asarray(old, dtype=np.float64).reshape(-1)
    fresh = np.asarray(fresh, dtype=np.float64).reshape(-1)
    return float(np.corrcoef(old, fresh)[0, 1])


def merge(args: argparse.Namespace) -> None:
    payloads = []
    for shard_index in range(args.shard_count):
        path = args.out_dir / f"grouped_shard_{shard_index}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        payloads.append(np.load(path, allow_pickle=False))

    query_ids = np.asarray(payloads[0]["query_ids"], dtype=np.int32)
    timesteps = np.asarray(payloads[0]["timesteps"], dtype=np.int32)
    checkpoints = np.asarray(payloads[0]["checkpoint_indices"], dtype=np.int32)
    score_indices = np.asarray(payloads[0]["score_indices"], dtype=np.int64)
    for payload in payloads[1:]:
        for key, reference in (
            ("query_ids", query_ids),
            ("timesteps", timesteps),
            ("checkpoint_indices", checkpoints),
            ("score_indices", score_indices),
        ):
            if not np.array_equal(payload[key], reference):
                raise ValueError(f"{key} differs across shards")

    arrays = {
        key: sum(np.asarray(payload[key], dtype=np.float64) for payload in payloads)
        for key in (
            "old_timestamp",
            "fresh_timestamp",
            "old_checkpoint",
            "fresh_checkpoint",
        )
    }
    # Convert from (group, query, datapoint) to the user-facing axis order.
    for key in arrays:
        arrays[key] = np.swapaxes(arrays[key], 0, 1)

    timestamp_rows = []
    checkpoint_rows = []
    datapoint_rows = []
    query_rows = []

    print("OLD12 vs FRESH12 LINEAR BOTH-L2 — GROUPED SIGN DISTRIBUTIONS")
    print(
        f"{'Q':>2s} {'GROUP':10s} {'N':>3s} {'OLD+':>8s} {'FRESH+':>8s} "
        f"{'++':>8s} {'+-':>8s} {'-+':>8s} {'--':>8s} "
        f"{'SAME':>8s} {'PEARSON':>9s}"
    )
    print("-" * 108)

    for qslot, query_id in enumerate(query_ids):
        for grouping, labels, old_key, fresh_key, output_rows in (
            (
                "timestamp",
                timesteps,
                "old_timestamp",
                "fresh_timestamp",
                timestamp_rows,
            ),
            (
                "checkpoint",
                checkpoints + 1,
                "old_checkpoint",
                "fresh_checkpoint",
                checkpoint_rows,
            ),
        ):
            old = arrays[old_key][qslot]
            fresh = arrays[fresh_key][qslot]
            summary = sign_summary(old, fresh)
            pearson = paired_pearson(old, fresh)
            query_row = {
                "query": int(query_id),
                "grouping": grouping,
                "num_groups": len(labels),
                **summary,
                "old_fresh_pearson": pearson,
            }
            query_rows.append(query_row)
            print(
                f"{int(query_id):2d} {grouping:10s} {len(labels):3d} "
                f"{summary['old_positive_fraction']:8.3f} "
                f"{summary['fresh_positive_fraction']:8.3f} "
                f"{summary['both_positive_fraction']:8.3f} "
                f"{summary['old_positive_fresh_negative_fraction']:8.3f} "
                f"{summary['old_negative_fresh_positive_fraction']:8.3f} "
                f"{summary['both_negative_fraction']:8.3f} "
                f"{summary['same_sign_fraction']:8.3f} {pearson:+9.4f}"
            )

            for group_slot, label in enumerate(labels):
                group_summary = sign_summary(old[group_slot], fresh[group_slot])
                output_rows.append(
                    {
                        "query": int(query_id),
                        "grouping": grouping,
                        "group": int(label),
                        **group_summary,
                        "old_mean": float(old[group_slot].mean()),
                        "fresh_mean": float(fresh[group_slot].mean()),
                        "old_std": float(old[group_slot].std()),
                        "fresh_std": float(fresh[group_slot].std()),
                        "old_fresh_pearson": paired_pearson(
                            old[group_slot], fresh[group_slot]
                        ),
                    }
                )

            for datapoint_slot, score_index in enumerate(score_indices):
                datapoint_summary = sign_summary(
                    old[:, datapoint_slot], fresh[:, datapoint_slot]
                )
                datapoint_rows.append(
                    {
                        "query": int(query_id),
                        "score_index": int(score_index),
                        "grouping": grouping,
                        "num_groups": len(labels),
                        **datapoint_summary,
                        "old_fresh_pearson_across_groups": paired_pearson(
                            old[:, datapoint_slot], fresh[:, datapoint_slot]
                        ),
                    }
                )

        expected_old = np.asarray(
            np.load(score_path(args, int(query_id), BANKS["old12"]["score_namespace"])),
            dtype=np.float64,
        )
        expected_fresh = np.asarray(
            np.load(
                score_path(args, int(query_id), BANKS["fresh12"]["score_namespace"])
            ),
            dtype=np.float64,
        )
        for bank, expected, timestamp_values, checkpoint_values in (
            (
                "old12",
                expected_old,
                arrays["old_timestamp"][qslot],
                arrays["old_checkpoint"][qslot],
            ),
            (
                "fresh12",
                expected_fresh,
                arrays["fresh_timestamp"][qslot],
                arrays["fresh_checkpoint"][qslot],
            ),
        ):
            timestamp_error = float(
                np.max(np.abs(timestamp_values.sum(axis=0) - expected))
            )
            checkpoint_error = float(
                np.max(np.abs(checkpoint_values.sum(axis=0) - expected))
            )
            print(
                f"    validate Q{int(query_id)} {bank}: "
                f"timestamp_error={timestamp_error:.3e} "
                f"checkpoint_error={checkpoint_error:.3e}"
            )
            if max(timestamp_error, checkpoint_error) > 2e-7:
                raise ValueError(f"grouped reconstruction failed for Q{query_id} {bank}")

    write_csv(args.out_dir / "per_timestamp.csv", timestamp_rows)
    write_csv(args.out_dir / "per_checkpoint.csv", checkpoint_rows)
    write_csv(args.out_dir / "per_datapoint_grouped.csv", datapoint_rows)
    write_csv(args.out_dir / "per_query_grouped_summary.csv", query_rows)
    atomic_savez(
        args.out_dir / "grouped_scores.npz",
        **{key: values.astype(np.float32) for key, values in arrays.items()},
        query_ids=query_ids,
        timesteps=timesteps,
        checkpoint_indices=checkpoints,
        score_indices=score_indices,
    )
    (args.out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "timestamp_group": "sum_checkpoint(term_weight * signed_component)",
                "checkpoint_group": "sum_timestamp(term_weight * signed_component)",
                "timestamp_shape": [len(query_ids), 10, 5000],
                "checkpoint_shape": [len(query_ids), 50, 5000],
                "normalization": "query_train_l2",
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"[saved] {args.out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("shard", "merge"):
        child = subparsers.add_parser(command)
        child.add_argument("--experiment", default="experiment1")
        child.add_argument("--train-seed", type=int, default=42)
        child.add_argument("--epochs", type=int, default=200)
        child.add_argument("--num-probes", type=int, default=12)
        child.add_argument(
            "--query-ids",
            type=parse_query_ids,
            default=parse_query_ids("0,1,2,3,4,5,6,7,8,9"),
        )
        child.add_argument("--shard-count", type=int, default=2)
        child.add_argument("--out-dir", type=Path, required=True)
        if command == "shard":
            child.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args()
    if args.num_probes != 12:
        raise ValueError("this analysis expects old12 and fresh12")
    if args.command == "shard":
        shard(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
