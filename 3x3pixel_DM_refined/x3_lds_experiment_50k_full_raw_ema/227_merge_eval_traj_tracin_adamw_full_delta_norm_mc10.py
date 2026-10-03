"""Merge and evaluate the 100-query non-aligned MC10 trajectory run."""

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

from exp_config import *
from tracin_das_config import TRACIN_DAS_ALL_QUERY_IDS
from importlib.machinery import SourceFileLoader
from pathlib import Path


worker = SourceFileLoader(
    "traj_adamw_mc10_worker",
    str(Path(__file__).with_name(
        "226_run_traj_tracin_adamw_full_delta_norm_mc10_shard.py"
    )),
).load_module()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompted-shard-count", type=int, default=3)
    parser.add_argument("--unprompted-shard-count", type=int, default=1)
    args = parser.parse_args()
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}

    for family in FAMILIES:
        shard_count = (
            args.prompted_shard_count
            if family == "prompted"
            else args.unprompted_shard_count
        )
        query_ids = [
            query_id
            for query_id in TRACIN_DAS_ALL_QUERY_IDS
            if by_id[query_id]["family"] == family
        ]
        covered = []
        values = {
            variant: np.zeros((len(query_ids), N_TRAIN), dtype=np.float64)
            for variant in worker.VARIANTS
        }
        for shard_index in range(shard_count):
            root = worker.shard_root(family, shard_index, shard_count)
            with open(root / "done.json") as handle:
                info = json.load(handle)
            if info["query_ids"] != query_ids:
                raise ValueError(f"query mismatch in {root}")
            covered.extend(int(value) for value in info["timestamp_indices"])
            for variant in worker.VARIANTS:
                values[variant] += np.load(
                    root / f"partial_{variant}.npy"
                ).astype(np.float64)
        if sorted(covered) != list(range(100)):
            raise ValueError(f"{family} shards do not cover all timestamps")

        for variant in worker.VARIANTS:
            method = worker.method_name(variant)
            for position, query_id in enumerate(query_ids):
                output = ATTR_DIR / method / f"q{query_id:02d}"
                output.mkdir(parents=True, exist_ok=True)
                np.save(output / "scores.npy", values[variant][position])
                atomic_json(
                    output / "info.json",
                    {
                        "query": by_id[query_id],
                        "method": method,
                        "normalization": variant,
                        "query_input": "cached reference trajectory x_t",
                        "query_delta": (
                            "L2-normalized next-minus-current predicted noise"
                        ),
                        "train_mc": worker.TRAIN_MC,
                        "train_noise_alignment": "independent",
                        "parameter_transform": "full AdamW",
                        "parameter_projection": "CountSketch4096",
                        "contraction": "timestamp_sum_squared",
                    },
                )

    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    result = {
        "query_ids": list(TRACIN_DAS_ALL_QUERY_IDS),
        "variants": {},
    }
    for variant in worker.VARIANTS:
        method = worker.method_name(variant)
        scores = np.stack(
            [
                np.load(
                    ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy"
                ).astype(np.float64)
                for query_id in TRACIN_DAS_ALL_QUERY_IDS
            ]
        )
        method_result = {"method": method, "targets": {}}
        print(f"\nMETHOD: {method}", flush=True)
        for metric in LDS_METRICS:
            observed = np.load(
                LDS_DIR / f"observed_{metric}.npy"
            ).astype(np.float64)
            signs = {}
            for sign_name, sign in (("negative", -1.0), ("positive", 1.0)):
                per_query = [
                    float(
                        spearmanr(
                            sign * (membership @ scores[query_id]),
                            observed[query_id],
                        ).statistic
                    )
                    for query_id in TRACIN_DAS_ALL_QUERY_IDS
                ]
                signs[sign_name] = {
                    "mean": float(np.nanmean(per_query)),
                    "std": float(np.nanstd(per_query)),
                    "per_query": per_query,
                }
            method_result["targets"][metric] = signs
            print(
                f"{metric:30s} "
                f"sign=-1 {signs['negative']['mean']:+.6f}±{signs['negative']['std']:.6f} | "
                f"sign=+1 {signs['positive']['mean']:+.6f}±{signs['positive']['std']:.6f}",
                flush=True,
            )
        result["variants"][variant] = method_result

    output = LDS_DIR / (
        "traj_tracin_reference_delta_norm_adamw_full_independent_mc10_"
        "norm4_timestamp_sum_squared_q00_q99.json"
    )
    atomic_json(output, result)
    print(f"[saved] {output}", flush=True)


if __name__ == "__main__":
    main()
