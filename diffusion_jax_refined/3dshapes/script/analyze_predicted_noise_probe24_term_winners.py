#!/usr/bin/env python3
"""Select the best of original+fresh probes independently for every trajectory term."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from analyze_predicted_noise_probe8_choose4 import (
    cache_group,
    load_target_data,
    spearman,
)
from run_predicted_noise_jvp_l2_squared import (
    query_artifact_path,
    train_part_dir,
)


ORIGINAL_PATTERN = "loss_direction_residual_rms_predicted_noise_probe4_r{probe_index}"
FRESH_PATTERN = (
    "loss_direction_residual_rms_predicted_noise_fresh_seed20260915_r{probe_index}"
)
NEXT_PATTERN = "loss_direction_residual_rms_original_f"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def load_features(
    args: argparse.Namespace,
    pattern: str,
    probe_indices: range,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    probes = []
    reference = None
    for probe_index in probe_indices:
        queries = []
        for query_id in args.query_ids:
            path = query_artifact_path(
                args.experiment,
                args.train_seed,
                args.epochs,
                query_id,
                num_probes=12,
                probe_index=probe_index,
                query_namespace_pattern=pattern,
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as payload:
                queries.append(np.asarray(payload["query_features"], dtype=np.float32))
                meta = {
                    "ckpt_indices": np.asarray(payload["ckpt_indices"], dtype=np.int32),
                    "timesteps": np.asarray(payload["timesteps"], dtype=np.int32),
                    "snapshot_positions": np.asarray(
                        payload["snapshot_positions"], dtype=np.int32
                    ),
                }
            if reference is None:
                reference = meta
            else:
                for key, value in meta.items():
                    if not np.array_equal(value, reference[key]):
                        raise ValueError(f"metadata mismatch for {path}:{key}")
        probes.append(np.stack(queries, axis=0))
    assert reference is not None
    return np.stack(probes, axis=0), reference


def load_next_gradients(
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    queries = []
    reference = None
    for query_id in args.query_ids:
        path = query_artifact_path(
            args.experiment,
            args.train_seed,
            args.epochs,
            query_id,
            num_probes=1,
            probe_index=0,
            query_namespace_pattern=NEXT_PATTERN,
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            queries.append(np.asarray(payload["query_features"], dtype=np.float32))
            meta = {
                "ckpt_indices": np.asarray(payload["ckpt_indices"], dtype=np.int32),
                "timesteps": np.asarray(payload["timesteps"], dtype=np.int32),
            }
        if reference is None:
            reference = meta
        else:
            for key, value in meta.items():
                if not np.array_equal(value, reference[key]):
                    raise ValueError(f"next-gradient metadata mismatch for {path}:{key}")
    assert reference is not None
    lookup = {
        (int(ckpt), int(timestep)): term
        for term, (ckpt, timestep) in enumerate(
            zip(reference["ckpt_indices"], reference["timesteps"])
        )
    }
    return np.stack(queries, axis=0), lookup


def target_data(
    args: argparse.Namespace,
    score_indices: np.ndarray,
) -> dict[int, tuple[np.ndarray, dict[str, np.ndarray]]]:
    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    root = SHAPES_ROOT / "result" / args.experiment
    result = {}
    for query_id in args.query_ids:
        record = records[query_id]
        eval_root = (
            root
            / "eval"
            / "prompted_solo"
            / f"query_{str(record['prompt']).replace(',', '_')}"
            / f"initial_seed_{int(record['initial_seed'])}"
        )
        result[query_id] = load_target_data(cache_group(eval_root), score_indices)
    return result


def unit_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def analyze_shard(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp

    original, meta = load_features(args, ORIGINAL_PATTERN, range(12))
    fresh, fresh_meta = load_features(args, FRESH_PATTERN, range(12))
    for key in meta:
        if not np.array_equal(meta[key], fresh_meta[key]):
            raise ValueError(f"original/fresh metadata mismatch for {key}")
    probes = np.concatenate((original, fresh), axis=0)
    next_gradients, next_lookup = load_next_gradients(args)
    probe_lookup = {
        (int(ckpt), int(timestep)): term
        for term, (ckpt, timestep) in enumerate(
            zip(meta["ckpt_indices"], meta["timesteps"])
        )
    }

    output_dir = args.out_dir
    all_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    targets = None
    incidence_devices = None
    score_indices_ref = None

    for ckpt_i in range(args.shard_index, 49, args.shard_count):
        path = train_part_dir(args.experiment, args.train_seed) / f"ckpt_{ckpt_i:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            train = np.asarray(payload["train_features"], dtype=np.float32)
            indices = np.asarray(payload["score_indices"], dtype=np.int64)
            ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        if score_indices_ref is None:
            score_indices_ref = indices
            targets = target_data(args, indices)
            incidence_devices = {
                query_id: jax.device_put(jnp.asarray(targets[query_id][0]))
                for query_id in args.query_ids
            }
        elif not np.array_equal(indices, score_indices_ref):
            raise ValueError(f"score indices differ in {path}")
        assert targets is not None and incidence_devices is not None

        for local_term, (ckpt, timestep) in enumerate(zip(ckpts, timesteps)):
            key = (int(ckpt), int(timestep))
            probe_term = probe_lookup[key]
            next_term = next_lookup.get(key)
            if next_term is None:
                continue
            train_unit = unit_rows(train[local_term])
            query_matrix = np.concatenate(
                [probes[:, qslot, probe_term, :] for qslot in range(len(args.query_ids))],
                axis=0,
            )
            query_unit = unit_rows(query_matrix)
            directional_device = jnp.asarray(train_unit) @ jnp.asarray(query_unit).T

            for qslot, query_id in enumerate(args.query_ids):
                columns = slice(qslot * 24, (qslot + 1) * 24)
                scores_device = directional_device[:, columns]
                scores = np.asarray(jax.device_get(scores_device), dtype=np.float64)
                incidence, true_values = targets[query_id]
                del incidence
                predictions = np.asarray(
                    jax.device_get(
                        scores_device.T @ incidence_devices[query_id].T
                    ),
                    dtype=np.float64,
                )
                endpoint = np.asarray(
                    [spearman(row, true_values["endpoint_contarfactual"]) for row in predictions]
                ) * 100.0
                trajectory = np.asarray(
                    [spearman(row, true_values["traj_contarfactual"]) for row in predictions]
                ) * 100.0
                joint = 0.5 * (endpoint + trajectory)

                q_unit = query_unit[columns]
                next_vector = next_gradients[qslot, next_term]
                next_unit = next_vector / max(float(np.linalg.norm(next_vector)), 1e-8)
                next_mse_cosine = q_unit @ next_unit
                next_update_cosine = -next_mse_cosine
                train_mean_cosine = scores.mean(axis=0)
                train_abs_cosine = np.abs(scores).mean(axis=0)
                train_positive_fraction = np.mean(scores > 0.0, axis=0)
                lds_next_relation = 100.0 * spearman(joint, next_update_cosine)
                winner = int(np.argmax(joint))

                for probe_index in range(24):
                    row = {
                        "query": query_id,
                        "checkpoint": int(ckpt) + 1,
                        "epoch": (int(ckpt) + 1) * 4,
                        "timestamp": int(timestep),
                        "probe": probe_index + 1,
                        "bank": "original" if probe_index < 12 else "fresh",
                        "bank_probe": probe_index + 1 if probe_index < 12 else probe_index - 11,
                        "endpoint_lds_percent": float(endpoint[probe_index]),
                        "traj_lds_percent": float(trajectory[probe_index]),
                        "cf_joint_lds_percent": float(joint[probe_index]),
                        "cosine_to_next_mse_gradient": float(next_mse_cosine[probe_index]),
                        "cosine_to_next_update": float(next_update_cosine[probe_index]),
                        "mean_train_gradient_cosine": float(train_mean_cosine[probe_index]),
                        "mean_abs_train_gradient_cosine": float(train_abs_cosine[probe_index]),
                        "train_gradient_positive_fraction": float(
                            train_positive_fraction[probe_index]
                        ),
                        "probe_lds_vs_next_update_spearman_percent": float(
                            lds_next_relation
                        ),
                        "selected": probe_index == winner,
                    }
                    all_rows.append(row)
                    if probe_index == winner:
                        selected_rows.append(row.copy())
            print(
                f"[term winner shard {args.shard_index}/{args.shard_count}] "
                f"checkpoint={int(ckpt) + 1}/49 timestamp={int(timestep)}",
                flush=True,
            )

    write_csv(output_dir / f"all_probes_shard_{args.shard_index:02d}.csv", all_rows)
    write_csv(output_dir / f"selected_shard_{args.shard_index:02d}.csv", selected_rows)


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def merge(args: argparse.Namespace) -> None:
    output_dir = args.out_dir
    all_rows = read_rows(
        [output_dir / f"all_probes_shard_{i:02d}.csv" for i in range(args.shard_count)]
    )
    selected = read_rows(
        [output_dir / f"selected_shard_{i:02d}.csv" for i in range(args.shard_count)]
    )
    order = lambda row: (
        int(row["query"]),
        int(row["checkpoint"]),
        int(row["timestamp"]),
        int(row.get("probe", 0)),
    )
    all_rows.sort(key=order)
    selected.sort(key=order)
    write_csv(output_dir / "all_probe_term_lds.csv", all_rows)
    write_csv(output_dir / "selected_term_winners.csv", selected)

    summaries = []
    for query_id in args.query_ids:
        rows = [row for row in selected if int(row["query"]) == query_id]
        counts = Counter(row["bank"] for row in rows)
        values = lambda key: [float(row[key]) for row in rows]
        summaries.append(
            {
                "query": query_id,
                "num_terms": len(rows),
                "selected_original": counts["original"],
                "selected_fresh": counts["fresh"],
                "mean_best_cf_joint_lds_percent": statistics.mean(
                    values("cf_joint_lds_percent")
                ),
                "mean_cosine_to_next_update": statistics.mean(
                    values("cosine_to_next_update")
                ),
                "mean_abs_cosine_to_next_update": statistics.mean(
                    [abs(value) for value in values("cosine_to_next_update")]
                ),
                "next_update_positive_fraction": statistics.mean(
                    [value > 0.0 for value in values("cosine_to_next_update")]
                ),
                "mean_train_gradient_cosine": statistics.mean(
                    values("mean_train_gradient_cosine")
                ),
                "mean_abs_train_gradient_cosine": statistics.mean(
                    values("mean_abs_train_gradient_cosine")
                ),
                "mean_probe_lds_vs_next_update_spearman_percent": statistics.mean(
                    values("probe_lds_vs_next_update_spearman_percent")
                ),
            }
        )
    write_csv(output_dir / "query_summary.csv", summaries)

    print("24-PROBE PER-TERM WINNER ANALYSIS — BOTH-L2, CF JOINT")
    print(
        f"{'Q':>2s} {'TERMS':>5s} {'OLD':>5s} {'FRESH':>5s} {'BEST LDS':>10s} "
        f"{'NEXT COS':>10s} {'|NEXT|':>9s} {'NEXT+':>7s} {'TRAIN COS':>10s} "
        f"{'LDS~NEXT':>10s}"
    )
    print("-" * 104)
    for row in summaries:
        print(
            f"{int(row['query']):2d} {int(row['num_terms']):5d} "
            f"{int(row['selected_original']):5d} {int(row['selected_fresh']):5d} "
            f"{float(row['mean_best_cf_joint_lds_percent']):9.3f}% "
            f"{float(row['mean_cosine_to_next_update']):+10.5f} "
            f"{float(row['mean_abs_cosine_to_next_update']):9.5f} "
            f"{float(row['next_update_positive_fraction']):7.3f} "
            f"{float(row['mean_train_gradient_cosine']):+10.5f} "
            f"{float(row['mean_probe_lds_vs_next_update_spearman_percent']):9.3f}%"
        )
    print(f"[saved] {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("analyze-shard", "merge"))
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-ids", default="2,3")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    args.query_ids = tuple(int(value) for value in args.query_ids.split(","))
    if len(args.query_ids) != 2:
        raise ValueError("this diagnostic expects exactly two query IDs")
    args.out_dir = args.out_dir or (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "eval"
        / "probe24_term_winners"
        / f"run_{args.run_id}"
    )
    if args.command == "analyze-shard":
        analyze_shard(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
