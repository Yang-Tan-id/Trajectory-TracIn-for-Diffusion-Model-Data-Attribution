"""Merge reference-trajectory MC4 timestamp shards."""

import argparse
import json

import numpy as np

from reference_traj_mc4_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=REF_MC4_DEFAULT_EPSILON)
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    parser.add_argument("--train-mc", type=int, default=REF_MC4_DEFAULT_TRAIN_MC)
    args = parser.parse_args()
    methods = ref_mc4_methods(args.epsilon, args.train_mc)
    totals = {
        name: np.zeros((len(REF_MC4_QUERY_IDS), N_TRAIN), dtype=np.float64)
        for name in methods
    }
    covered = []
    metadata = []
    for shard_index in range(args.timestamp_shard_count):
        root = ref_mc4_shard_root(
            shard_index, args.timestamp_shard_count, args.epsilon, args.train_mc
        )
        with open(root / "done.json") as handle:
            info = json.load(handle)
        metadata.append(info)
        if info["query_ids"] != list(REF_MC4_QUERY_IDS):
            raise ValueError(f"query IDs differ in {root}")
        if float(info["epsilon"]) != float(args.epsilon):
            raise ValueError(f"epsilon differs in {root}")
        if int(info["train_mc"]) != int(args.train_mc):
            raise ValueError(f"train MC differs in {root}")
        covered.extend(int(value) for value in info["timestamp_indices"])
        for name in methods:
            values = np.load(root / f"{name}.npy")
            if values.shape != totals[name].shape:
                raise ValueError(f"shape mismatch: {root}/{name}.npy")
            totals[name] += values
    if sorted(covered) != list(range(TRAJ_SNAPSHOTS)):
        raise ValueError("timestamp shards do not cover 0..99 exactly")
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    for contraction, method in methods.items():
        for position, query_id in enumerate(REF_MC4_QUERY_IDS):
            output = ATTR_DIR / method / f"q{query_id:02d}"
            output.mkdir(parents=True, exist_ok=True)
            np.save(output / "scores.npy", totals[contraction][position])
            with open(output / "info.json", "w") as handle:
                json.dump(
                    {
                        "query": by_id[query_id],
                        "method": method,
                        "contraction": contraction,
                        "epsilon": args.epsilon,
                        "query_mc": REF_MC4_COUNT,
                        "query_state_source": "cached final-EMA reference trajectory",
                        "query_perturbation": "four unit-L2 Gaussian directions",
                        "train_mc": args.train_mc,
                        "train_noise": "independent from query perturbations",
                        "parameter_source": "raw",
                        "checkpoint_target": "next",
                        "projection": "countsketch",
                        "projection_dim": TRACIN_PROJ_DIM,
                        "lr_weighted": TRACIN_USE_LR_WEIGHTS,
                    },
                    handle, indent=2,
                )
        print(f"[saved] {method} q00-q09", flush=True)


if __name__ == "__main__":
    main()
