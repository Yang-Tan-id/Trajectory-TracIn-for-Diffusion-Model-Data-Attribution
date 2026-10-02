"""Merge four direction shards and evaluate q00-q09 LDS."""

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

from direction_integrated_tracin_das_config import *


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction-shard-count", type=int, default=4)
    parser.add_argument("--direction-count", type=int, default=DITD_DIRECTION_COUNT)
    args = parser.parse_args()
    totals = {
        contraction: np.zeros((len(DITD_QUERY_IDS), N_TRAIN), dtype=np.float64)
        for contraction in DITD_METHODS
    }
    covered = []
    metadata = []
    for shard_index in range(args.direction_shard_count):
        root = ditd_shard_root(shard_index, args.direction_shard_count)
        with open(root / "done.json") as handle:
            info = json.load(handle)
        metadata.append(info)
        if info["query_ids"] != list(DITD_QUERY_IDS):
            raise ValueError(f"query mismatch in {root}")
        if int(info["direction_count"]) != args.direction_count:
            raise ValueError(f"direction count mismatch in {root}")
        covered.extend(int(value) for value in info["direction_indices"])
        for contraction in DITD_METHODS:
            values = np.load(root / f"{contraction}.npy")
            if values.shape != totals[contraction].shape:
                raise ValueError(f"shape mismatch in {root}/{contraction}.npy")
            totals[contraction] += values.astype(np.float64)
    if sorted(covered) != list(range(args.direction_count)):
        raise ValueError("direction shards do not cover every direction exactly once")

    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    for contraction, method in DITD_METHODS.items():
        for query_position, query_id in enumerate(DITD_QUERY_IDS):
            output = ATTR_DIR / method / f"q{query_id:02d}"
            output.mkdir(parents=True, exist_ok=True)
            np.save(output / "scores.npy", totals[contraction][query_position])
            atomic_json(
                output / "info.json",
                {
                    "query": by_id[query_id],
                    "method": method,
                    "contraction": contraction,
                    "checkpoint_target": "next",
                    "checkpoint_transitions": 49,
                    "parameter_source": "raw",
                    "endpoint_source": "cached final-EMA query endpoint",
                    "parameter_projection": "CountSketch4096",
                    "direction_count": args.direction_count,
                    "train_timesteps": list(DITD_TRAIN_TIMESTEPS),
                    "query_timesteps": list(DITD_QUERY_TIMESTEPS),
                    "train_loss": (
                        "for each checkpoint/direction, exact mean simple loss "
                        "over t=0..999 using the same Gaussian noise direction"
                    ),
                    "direction_alignment": metadata[0]["direction_alignment"],
                    "checkpoint_noise_policy": metadata[0]["checkpoint_noise_policy"],
                    "query_scalar": (
                        "dot(epsilon_current, normalize(epsilon_next-epsilon_current))"
                    ),
                    "lr_weighted": True,
                    "direction_shards": args.direction_shard_count,
                },
            )
        print(f"[saved] {method} q00-q09", flush=True)

    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    result = {
        "direction_count": args.direction_count,
        "train_t_count": len(DITD_TRAIN_TIMESTEPS),
        "query_t_count": len(DITD_QUERY_TIMESTEPS),
        "methods": {},
    }
    for contraction, method in DITD_METHODS.items():
        method_result = {"contraction": contraction, "targets": {}}
        scores = totals[contraction]
        print(f"\nMETHOD: {method}", flush=True)
        for metric in LDS_METRICS:
            observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
            signs = {}
            for sign_name, sign in (("negative", -1.0), ("positive", 1.0)):
                per_query = []
                for query_position, query_id in enumerate(DITD_QUERY_IDS):
                    prediction = sign * (membership @ scores[query_position])
                    per_query.append(
                        float(spearmanr(prediction, observed[query_id]).statistic)
                    )
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
        result["methods"][method] = method_result
    output_path = LDS_DIR / f"{DITD_METHOD_STEM}_q00_q09.json"
    LDS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(output_path, result)
    print(f"[saved] {output_path}", flush=True)


if __name__ == "__main__":
    main()

