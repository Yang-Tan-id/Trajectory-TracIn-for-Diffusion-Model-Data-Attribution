"""Merge q00-q98 interval-mean-LR timestamp-count sweep shards."""

import argparse
import json
import os

import numpy as np

from tracin_das_config import *


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--timestamp-shard-count", type=int, default=2)
    args = parser.parse_args()
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    query_ids = [
        query_id
        for query_id in TRACIN_DAS_FIRST99_QUERY_IDS
        if by_id[query_id]["family"] == args.family
    ]
    contractions = ("linear", "termwise_squared", "timestamp_sum_squared")
    totals = {
        str(count): {
            contraction: np.zeros((len(query_ids), N_TRAIN), dtype=np.float64)
            for contraction in contractions
        }
        for count in TRACIN_DAS_TIMESTAMP_COUNTS
    }
    covered = []
    for shard_index in range(args.timestamp_shard_count):
        root = tracin_das_timestamp_sweep_shard_root(
            args.family, shard_index, args.timestamp_shard_count
        )
        with open(root / "done.json") as handle:
            info = json.load(handle)
        if info["query_ids"] != query_ids:
            raise ValueError(f"query mismatch in {root}")
        if info["family"] != args.family or info["query_scope"] != "first99":
            raise ValueError(f"family/query-scope mismatch in {root}")
        if not info.get("timestamp_count_multi", False):
            raise ValueError(f"{root} is not a timestamp-count sweep shard")
        expected_timestamp_groups = {
            str(count): list(tracin_das_timestamp_indices(count))
            for count in TRACIN_DAS_TIMESTAMP_COUNTS
        }
        if info["selected_timestamp_indices_by_group"] != expected_timestamp_groups:
            raise ValueError(f"timestamp selections mismatch in {root}")
        covered.extend(int(value) for value in info["timestamp_indices"])
        for count in TRACIN_DAS_TIMESTAMP_COUNTS:
            group = str(count)
            for contraction in contractions:
                path = root / f"{group}_{contraction}.npy"
                values = np.load(path)
                if values.shape != totals[group][contraction].shape:
                    raise ValueError(f"{path} shape={values.shape}")
                totals[group][contraction] += values
    if sorted(covered) != list(range(len(DAS_TIMESTEPS))):
        raise ValueError("timestamp shards do not cover all 100 timestamps exactly")

    for count in TRACIN_DAS_TIMESTAMP_COUNTS:
        group = str(count)
        methods = tracin_das_interval_mean_lr_timestamp_methods(count)
        selected_indices = list(tracin_das_timestamp_indices(count))
        for contraction, method in methods.items():
            for query_position, query_id in enumerate(query_ids):
                output = ATTR_DIR / method / f"q{query_id:02d}"
                output.mkdir(parents=True, exist_ok=True)
                np.save(output / "scores.npy", totals[group][contraction][query_position])
                atomic_json(
                    output / "info.json",
                    {
                        "query": by_id[query_id],
                        "query_scope": "q00-q98",
                        "method": method,
                        "contraction": contraction,
                        "checkpoint_bank_count": 50,
                        "actual_transition_count": 49,
                        "timestamp_count": count,
                        "selected_timestamp_indices": selected_indices,
                        "selected_diffusion_timesteps": [
                            int(DAS_TIMESTEPS[index]) for index in selected_indices
                        ],
                        "checkpoint_target": "next",
                        "parameter_source": "raw",
                        "endpoint_source": "cached final-EMA query endpoint",
                        "noise_mode": "checkpoint",
                        "parameter_projection": "projected4096",
                        "parameter_projection_dim": TRACIN_PROJ_DIM,
                        "train_noise_mode": "aligned",
                        "train_mc": 1,
                        "learning_rate_source": (
                            "exact mean scheduled LR over "
                            "[current global_step, next global_step)"
                        ),
                        "timestamp_weight": 1.0 / count,
                        "timestamp_shards": args.timestamp_shard_count,
                    },
                )
            print(
                f"[saved] timestamps={count} contraction={contraction} "
                f"family={args.family} q{query_ids[0]:02d}-q{query_ids[-1]:02d}",
                flush=True,
            )


if __name__ == "__main__":
    main()
