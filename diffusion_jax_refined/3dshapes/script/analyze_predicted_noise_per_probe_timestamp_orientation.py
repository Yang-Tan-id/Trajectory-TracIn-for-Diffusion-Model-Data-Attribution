#!/usr/bin/env python3
"""Orient every probe/timestamp score independently before reducing probes."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT / "script") not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT / "script"))

from analyze_predicted_noise_old_fresh_term_signs import (
    BANKS,
    atomic_savez,
    parse_query_ids,
    score_path,
    write_csv,
)
from analyze_predicted_noise_probe8_choose4 import (
    TARGETS,
    cache_group,
    load_target_data,
    spearman,
)
from run_predicted_noise_jvp_l2_squared import (
    query_artifact_path,
    train_part_dir,
)


METHODS = (
    "per_probe_timestamp_mean_oriented_negative",
    "per_probe_datapoint_timestamp_forced_negative",
    "full_linear_reconstructed",
)


def load_query_bank(
    args: argparse.Namespace,
    pattern: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return unit query features with shape (probe, query, term, dimension)."""
    probes = []
    reference = None
    for probe_index in range(args.num_probes):
        per_query = []
        for query_id in args.query_ids:
            path = query_artifact_path(
                args.experiment,
                args.train_seed,
                args.epochs,
                query_id,
                num_probes=args.num_probes,
                probe_index=probe_index,
                query_namespace_pattern=pattern,
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as payload:
                feature = np.asarray(payload["query_features"], dtype=np.float32)
                metadata = {
                    "ckpt_indices": np.asarray(
                        payload["ckpt_indices"], dtype=np.int32
                    ),
                    "timesteps": np.asarray(payload["timesteps"], dtype=np.int32),
                }
            norm = np.linalg.norm(feature, axis=-1, keepdims=True)
            per_query.append(feature / np.maximum(norm, 1e-8))
            if reference is None:
                reference = metadata
            else:
                for key, value in metadata.items():
                    if not np.array_equal(value, reference[key]):
                        raise ValueError(f"query metadata mismatch for {path}:{key}")
        probes.append(np.stack(per_query, axis=0))
        print(f"[query bank] {pattern} probe={probe_index + 1}/{args.num_probes}")
    assert reference is not None
    return np.stack(probes, axis=0).astype(np.float32), reference


def shard(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp

    output = args.out_dir / f"per_probe_shard_{args.shard_index}.npz"
    if output.is_file():
        print(f"[skip] shard exists: {output}", flush=True)
        return

    banks = {}
    metadata = None
    for bank, config in BANKS.items():
        values, bank_metadata = load_query_bank(args, config["query_pattern"])
        banks[bank] = values
        if metadata is None:
            metadata = bank_metadata
        else:
            for key, reference in metadata.items():
                if not np.array_equal(bank_metadata[key], reference):
                    raise ValueError(f"{bank} metadata differs for {key}")
    assert metadata is not None

    lookup = {
        (int(checkpoint), int(timestep)): term
        for term, (checkpoint, timestep) in enumerate(
            zip(metadata["ckpt_indices"], metadata["timesteps"])
        )
    }
    unique_timesteps = np.asarray(
        list(dict.fromkeys(int(value) for value in metadata["timesteps"])),
        dtype=np.int32,
    )
    if len(unique_timesteps) != 10:
        raise ValueError(f"expected 10 timestamps, got {unique_timesteps}")
    timestep_slots = {int(value): slot for slot, value in enumerate(unique_timesteps)}
    shape = (args.num_probes, 10, len(args.query_ids), 5000)
    grouped = {
        bank: np.zeros(shape, dtype=np.float64)
        for bank in BANKS
    }
    score_indices = None

    for checkpoint_slot in range(args.shard_index, 50, args.shard_count):
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
            timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            weights = np.asarray(payload["term_weights"], dtype=np.float64)
        if score_indices is None:
            score_indices = indices
        elif not np.array_equal(score_indices, indices):
            raise ValueError(f"score indices differ in {path}")

        for local_term, (checkpoint, timestep, weight) in enumerate(
            zip(checkpoints, timesteps, weights)
        ):
            term = lookup[(int(checkpoint), int(timestep))]
            train_device = jax.device_put(jnp.asarray(train[local_term]))
            train_unit = train_device / jnp.maximum(
                jnp.linalg.norm(train_device, axis=1, keepdims=True), 1e-8
            )
            combined = np.concatenate(
                [banks[bank][:, :, term, :].reshape(-1, train.shape[-1]) for bank in BANKS],
                axis=0,
            )
            values = np.asarray(
                jax.device_get(train_unit @ jnp.asarray(combined).T),
                dtype=np.float32,
            ).T
            offset = 0
            width = args.num_probes * len(args.query_ids)
            timestamp_slot = timestep_slots[int(timestep)]
            for bank in BANKS:
                bank_values = values[offset : offset + width].reshape(
                    args.num_probes, len(args.query_ids), len(indices)
                )
                grouped[bank][:, timestamp_slot] += float(weight) * bank_values
                offset += width
        print(f"[per-probe grouped] checkpoint={checkpoint_slot + 1}/50", flush=True)

    assert score_indices is not None
    atomic_savez(
        output,
        **{bank: values for bank, values in grouped.items()},
        query_ids=np.asarray(args.query_ids, dtype=np.int32),
        timesteps=unique_timesteps,
        score_indices=score_indices,
    )
    print(f"[saved] {output}", flush=True)


def merge(args: argparse.Namespace) -> None:
    payloads = []
    for shard_index in range(args.shard_count):
        path = args.out_dir / f"per_probe_shard_{shard_index}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        payloads.append(np.load(path, allow_pickle=False))
    query_ids = np.asarray(payloads[0]["query_ids"], dtype=np.int32)
    timesteps = np.asarray(payloads[0]["timesteps"], dtype=np.int32)
    score_indices = np.asarray(payloads[0]["score_indices"], dtype=np.int64)
    for payload in payloads[1:]:
        for key, reference in (
            ("query_ids", query_ids),
            ("timesteps", timesteps),
            ("score_indices", score_indices),
        ):
            if not np.array_equal(payload[key], reference):
                raise ValueError(f"{key} differs across shards")
    grouped = {
        bank: sum(np.asarray(payload[bank], dtype=np.float64) for payload in payloads)
        for bank in BANKS
    }

    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    rows: list[dict[str, object]] = []
    orientation_rows: list[dict[str, object]] = []
    print("PER-PROBE/PER-TIMESTAMP ORIENTATION — BOTH-L2, prediction sign +1")
    print(
        f"{'BANK':8s} {'Q':>2s} {'METHOD':47s} {'ENDPOINT':>10s} "
        f"{'TRAJ':>10s} {'CF JOINT':>10s} {'NOISE':>10s} {'SIMPLE':>10s}"
    )
    print("-" * 132)
    for bank, values in grouped.items():
        for qslot, query_id_value in enumerate(query_ids):
            query_id = int(query_id_value)
            record = records[query_id]
            prompt = str(record["prompt"])
            prompt_tag = prompt.replace(",", "_")
            eval_root = (
                SHAPES_ROOT
                / "result"
                / args.experiment
                / "eval"
                / "prompted_solo"
                / f"query_{prompt_tag}"
                / f"initial_seed_{int(record['initial_seed'])}"
            )
            incidence, true_values = load_target_data(
                cache_group(eval_root), score_indices
            )
            per_probe_timestamp = values[:, :, qslot, :]
            component_means = per_probe_timestamp.mean(axis=-1)
            signs = np.where(component_means >= 0.0, 1.0, -1.0)
            scores = {
                "per_probe_timestamp_mean_oriented_negative": -np.sum(
                    signs[:, :, None] * per_probe_timestamp, axis=(0, 1)
                ) / args.num_probes,
                "per_probe_datapoint_timestamp_forced_negative": -np.sum(
                    np.abs(per_probe_timestamp), axis=(0, 1)
                ) / args.num_probes,
                "full_linear_reconstructed": np.sum(
                    per_probe_timestamp, axis=(0, 1)
                ) / args.num_probes,
            }
            for probe in range(args.num_probes):
                for timestamp_slot, timestep in enumerate(timesteps):
                    orientation_rows.append(
                        {
                            "bank": bank,
                            "query": query_id,
                            "probe": probe + 1,
                            "timestep": int(timestep),
                            "component_mean": float(
                                component_means[probe, timestamp_slot]
                            ),
                            "original_mean_sign": int(
                                signs[probe, timestamp_slot]
                            ),
                        }
                    )

            lookup = {}
            for method in METHODS:
                prediction = scores[method] @ incidence.T
                for target in TARGETS:
                    value = 100.0 * spearman(prediction, true_values[target])
                    lookup[(method, target)] = value
                    rows.append(
                        {
                            "bank": bank,
                            "query": query_id,
                            "method": method,
                            "target": target,
                            "lds_percent": value,
                            "prompt": prompt_tag,
                        }
                    )
                endpoint = lookup[(method, "endpoint_contarfactual")]
                trajectory = lookup[(method, "traj_contarfactual")]
                print(
                    f"{bank:8s} {query_id:2d} {method:47s} "
                    f"{endpoint:9.3f}% {trajectory:9.3f}% "
                    f"{0.5 * (endpoint + trajectory):9.3f}% "
                    f"{lookup[(method, 'noise_trajectory')]:9.3f}% "
                    f"{lookup[(method, 'simple_loss')]:9.3f}%"
                )

            expected = np.asarray(
                np.load(score_path(args, query_id, BANKS[bank]["score_namespace"])),
                dtype=np.float64,
            )
            error = float(
                np.max(np.abs(scores["full_linear_reconstructed"] - expected))
            )
            print(f"    validate {bank} Q{query_id}: max_abs_error={error:.3e}")
            if error > 2e-7:
                raise ValueError(f"linear reconstruction failed for {bank} Q{query_id}")

    summary_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        summary_groups[(str(row["bank"]), str(row["method"]), str(row["target"]))].append(
            float(row["lds_percent"])
        )
    summary_rows = []
    print("\n10-query mean")
    print(
        f"{'BANK':8s} {'METHOD':47s} {'ENDPOINT':>10s} {'TRAJ':>10s} "
        f"{'CF JOINT':>10s} {'NOISE':>10s} {'SIMPLE':>10s}"
    )
    print("-" * 132)
    for bank in BANKS:
        for method in METHODS:
            means = {
                target: statistics.mean(summary_groups[(bank, method, target)])
                for target in TARGETS
            }
            joint = 0.5 * (
                means["endpoint_contarfactual"] + means["traj_contarfactual"]
            )
            print(
                f"{bank:8s} {method:47s} "
                f"{means['endpoint_contarfactual']:9.3f}% "
                f"{means['traj_contarfactual']:9.3f}% {joint:9.3f}% "
                f"{means['noise_trajectory']:9.3f}% "
                f"{means['simple_loss']:9.3f}%"
            )
            for target in TARGETS:
                summary_rows.append(
                    {
                        "bank": bank,
                        "method": method,
                        "target": target,
                        "mean_lds_percent": means[target],
                        "std_lds_percent": statistics.stdev(
                            summary_groups[(bank, method, target)]
                        ),
                    }
                )

    write_csv(args.out_dir / "per_query_lds.csv", rows)
    write_csv(args.out_dir / "probe_timestamp_orientations.csv", orientation_rows)
    write_csv(args.out_dir / "ten_query_summary.csv", summary_rows)
    atomic_savez(
        args.out_dir / "per_probe_timestamp_scores.npz",
        **{bank: values.astype(np.float32) for bank, values in grouped.items()},
        query_ids=query_ids,
        timesteps=timesteps,
        score_indices=score_indices,
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
        raise ValueError("this analysis expects 12 probes per bank")
    if args.command == "shard":
        shard(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
