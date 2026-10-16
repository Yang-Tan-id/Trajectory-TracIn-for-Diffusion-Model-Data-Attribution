"""Merge direction20 full-AdamW shards and evaluate all X3 LDS targets."""

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

from direction20_adamw_dual_config import *


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction-shard-count", type=int, default=2)
    parser.add_argument("--direction-count", type=int, default=D20_DIRECTION_COUNT)
    parser.add_argument("--train-shard-count", type=int, default=1)
    parser.add_argument(
        "--noise-pairing",
        choices=(
            "aligned-shared-direction",
            "aligned-per-t",
            "independent-per-t",
        ),
        default="aligned-shared-direction",
    )
    parser.add_argument(
        "--query-mode",
        choices=("dual", "trajectory", "endpoint"),
        default="dual",
    )
    args = parser.parse_args()
    query_modes = (
        D20_QUERY_MODES if args.query_mode == "dual" else (args.query_mode,)
    )
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    merged = {
        mode: {
            variant: np.zeros((100, N_TRAIN), dtype=np.float64)
            for variant in D20_VARIANTS
        }
        for mode in query_modes
    }

    for family in FAMILIES:
        query_ids = sorted(
            query_id
            for query_id, record in by_id.items()
            if record["family"] == family
        )
        covered = []
        family_values = {
            mode: {
                variant: np.zeros((len(query_ids), N_TRAIN), dtype=np.float64)
                for variant in D20_VARIANTS
            }
            for mode in query_modes
        }
        train_ranges = []
        for shard_index in range(args.direction_shard_count):
            for train_shard_index in range(args.train_shard_count):
                root = d20_shard_root(
                    family,
                    shard_index,
                    args.direction_shard_count,
                    args.noise_pairing,
                    args.direction_count,
                    query_modes,
                    train_shard_index,
                    args.train_shard_count,
                )
                with open(root / "done.json") as handle:
                    info = json.load(handle)
                if info["query_ids"] != query_ids:
                    raise ValueError(f"query mismatch in {root}")
                covered.extend(int(value) for value in info["direction_indices"])
                train_ranges.append(tuple(int(value) for value in info["train_range"]))
                for mode in query_modes:
                    for variant in D20_VARIANTS:
                        family_values[mode][variant] += np.load(
                            root / f"{mode}_{variant}.npy"
                        ).astype(np.float64)
        expected_directions = sorted(
            list(range(args.direction_count)) * args.train_shard_count
        )
        if sorted(covered) != expected_directions:
            raise ValueError(
                f"{family} shards do not cover all {args.direction_count} banks"
            )
        expected_ranges = [
            (
                N_TRAIN * index // args.train_shard_count,
                N_TRAIN * (index + 1) // args.train_shard_count,
            )
            for index in range(args.train_shard_count)
        ] * args.direction_shard_count
        if sorted(train_ranges) != sorted(expected_ranges):
            raise ValueError(f"{family} train shards do not cover the dataset")
        for mode in query_modes:
            for variant in D20_VARIANTS:
                values = family_values[mode][variant]
                merged[mode][variant][query_ids] = values
                method = d20_method(
                    mode, variant, args.noise_pairing, args.direction_count
                )
                for position, query_id in enumerate(query_ids):
                    output = ATTR_DIR / method / f"q{query_id:02d}"
                    output.mkdir(parents=True, exist_ok=True)
                    np.save(output / "scores.npy", values[position])
                    atomic_json(
                        output / "info.json",
                        {
                            "query": by_id[query_id],
                            "method": method,
                            "query_mode": mode,
                            "normalization": variant,
                            "direction_count": args.direction_count,
                            "train_timestamps": 100,
                            "query_timestamps": 100,
                            "checkpoint_transitions": 49,
                            "parameter_transform": "full saved-state AdamW",
                            "parameter_projection": "CountSketch4096",
                            "contraction": "timestamp_sum_squared",
                        },
                    )

    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    result = {"query_ids": list(range(100)), "modes": {}}
    for mode in query_modes:
        result["modes"][mode] = {}
        for variant in D20_VARIANTS:
            method = d20_method(
                mode, variant, args.noise_pairing, args.direction_count
            )
            scores = merged[mode][variant]
            entry = {"method": method, "targets": {}}
            print(f"\nMETHOD: {method}", flush=True)
            for metric in LDS_METRICS:
                observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
                signs = {}
                for sign_name, sign in (("negative", -1.0), ("positive", 1.0)):
                    per_query = [
                        float(
                            spearmanr(
                                sign * (membership @ scores[query_id]),
                                observed[query_id],
                            ).statistic
                        )
                        for query_id in range(100)
                    ]
                    signs[sign_name] = {
                        "mean": float(np.nanmean(per_query)),
                        "std": float(np.nanstd(per_query)),
                        "per_query": per_query,
                    }
                entry["targets"][metric] = signs
                print(
                    f"{metric:30s} "
                    f"sign=-1 {signs['negative']['mean']:+.6f}±{signs['negative']['std']:.6f} | "
                    f"sign=+1 {signs['positive']['mean']:+.6f}±{signs['positive']['std']:.6f}",
                    flush=True,
                )
            result["modes"][mode][variant] = entry

    output = LDS_DIR / (
        f"bank{args.direction_count}_100x1_adamw_full_{args.query_mode}_"
        f"{args.noise_pairing}_q00_q99.json"
        if args.noise_pairing in ("independent-per-t", "aligned-per-t")
        else "direction20_mean100t_adamw_full_dual_q00_q99.json"
    )
    atomic_json(output, result)
    print(f"[saved] {output}", flush=True)


if __name__ == "__main__":
    main()
