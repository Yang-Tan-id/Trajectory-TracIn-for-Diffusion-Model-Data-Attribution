"""Merge four probe12 timestamp shards and retain every timestamp score."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from traj_probe12_config import *


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    args = parser.parse_args()
    if args.timestamp_shard_count <= 0:
        raise ValueError("--timestamp-shard-count must be positive")

    timestamp_paths = {}
    covered = []
    for shard_index in range(args.timestamp_shard_count):
        root = traj_probe12_shard_root(shard_index, args.timestamp_shard_count)
        with open(root / "done.json") as handle:
            metadata = json.load(handle)
        if metadata["query_ids"] != list(TRAJ_PROBE12_QUERY_IDS):
            raise ValueError(f"query IDs mismatch in {root}")
        if metadata["variant_order"] != [list(item) for item in TRAJ_PROBE12_VARIANTS]:
            raise ValueError(f"variant order mismatch in {root}")
        if int(metadata["num_probes"]) != TRAJ_PROBE12_NUM_PROBES:
            raise ValueError(f"probe count mismatch in {root}")
        for timestamp_index in metadata["timestamp_indices"]:
            timestamp_index = int(timestamp_index)
            if timestamp_index in timestamp_paths:
                raise ValueError(f"duplicate timestamp {timestamp_index}")
            timestamp_paths[timestamp_index] = root / f"timestamp_{timestamp_index:03d}.npy"
            covered.append(timestamp_index)
    if sorted(covered) != list(range(TRAJ_SNAPSHOTS)):
        raise ValueError(f"shards do not cover timestamps 0..99 exactly: {covered}")

    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    records = [by_id[qid] for qid in TRAJ_PROBE12_QUERY_IDS]
    t_seq = np.load(Path(records[0]["dir"]) / "trajectory_t.npy")
    q_count = len(records)
    variant_count = len(TRAJ_PROBE12_VARIANTS)
    totals = np.zeros((variant_count, q_count, N_TRAIN), dtype=np.float64)
    memmaps = {}

    for variant_index, variant in enumerate(TRAJ_PROBE12_VARIANTS):
        method = TRAJ_PROBE12_METHODS[variant]
        for query_index, record in enumerate(records):
            output = ATTR_DIR / method / f"q{int(record['query_id']):02d}"
            output.mkdir(parents=True, exist_ok=True)
            memmaps[(variant_index, query_index)] = np.lib.format.open_memmap(
                output / "per_timestamp_scores.npy",
                mode="w+",
                dtype=np.float32,
                shape=(TRAJ_SNAPSHOTS, N_TRAIN),
            )

    for timestamp_index in range(TRAJ_SNAPSHOTS):
        values = np.load(timestamp_paths[timestamp_index], mmap_mode="r")
        expected = (variant_count, q_count, N_TRAIN)
        if values.shape != expected:
            raise ValueError(
                f"{timestamp_paths[timestamp_index]} shape={values.shape}, "
                f"expected={expected}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite scores in {timestamp_paths[timestamp_index]}")
        for variant_index in range(variant_count):
            for query_index in range(q_count):
                row = np.asarray(values[variant_index, query_index], dtype=np.float32)
                memmaps[(variant_index, query_index)][timestamp_index] = row
                totals[variant_index, query_index] += row.astype(np.float64)
        if timestamp_index == 0 or (timestamp_index + 1) % 10 == 0:
            print(f"[merge] timestamp {timestamp_index + 1}/100", flush=True)

    for value in memmaps.values():
        value.flush()
    del memmaps

    for variant_index, variant in enumerate(TRAJ_PROBE12_VARIANTS):
        contraction, query_normalization = variant
        method = TRAJ_PROBE12_METHODS[variant]
        for query_index, record in enumerate(records):
            query_id = int(record["query_id"])
            output = ATTR_DIR / method / f"q{query_id:02d}"
            np.save(output / "scores.npy", totals[variant_index, query_index])
            atomic_json(
                output / "info.json",
                {
                    "query": record,
                    "method": method,
                    "order": "first",
                    "query_objective": "predicted_noise_output_probe",
                    "param_source": "raw",
                    "trajectory_source": "cached_final_ema_ddim_trajectory",
                    "projection": "countsketch",
                    "proj_dim": TRACIN_PROJ_DIM,
                    "num_output_probes": TRAJ_PROBE12_NUM_PROBES,
                    "output_probe_distribution": "timestamp-shared standard Gaussian / sqrt(27)",
                    "probe_sharing": "shared across checkpoints and queries within timestamp",
                    "query_normalization": query_normalization,
                    "query_l2_domain": "exact full parameter gradient before CountSketch",
                    "contraction": contraction,
                    "num_checkpoints": TRAJ_PROBE12_CHECKPOINT_COUNT,
                    "checkpoint_indices": list(range(TRAJ_PROBE12_CHECKPOINT_COUNT)),
                    "num_timestamps": TRAJ_SNAPSHOTS,
                    "trajectory_timesteps": [int(value) for value in t_seq],
                    "train_mc": TRACIN_TRAIN_MC,
                    "lr_weighted": TRACIN_USE_LR_WEIGHTS,
                    "timestamp_weight": 1.0 / TRAJ_SNAPSHOTS,
                    "probe_reduction": "mean",
                    "per_timestamp_scores": "per_timestamp_scores.npy",
                    "per_timestamp_shape": [TRAJ_SNAPSHOTS, N_TRAIN],
                    "total_score": "sum over per_timestamp_scores rows",
                    "timestamp_shards": args.timestamp_shard_count,
                },
            )
    print("[done] merged probe12 scores and retained all timestamp rows", flush=True)


if __name__ == "__main__":
    main()
