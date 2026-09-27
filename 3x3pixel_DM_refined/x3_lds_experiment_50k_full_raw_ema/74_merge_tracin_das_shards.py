"""Merge four TracIn-DAS timestamp shards into q00-q09 score banks."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from tracin_das_config import *


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    args = parser.parse_args()
    totals = {
        contraction: np.zeros((len(TRACIN_DAS_QUERY_IDS), N_TRAIN), dtype=np.float64)
        for contraction in TRACIN_DAS_METHODS
    }
    covered = []
    metadata = []
    for shard_index in range(args.timestamp_shard_count):
        root = tracin_das_shard_root(shard_index, args.timestamp_shard_count)
        with open(root / "done.json") as handle:
            info = json.load(handle)
        metadata.append(info)
        if info["query_ids"] != list(TRACIN_DAS_QUERY_IDS):
            raise ValueError(f"query mismatch in {root}")
        covered.extend(int(value) for value in info["timestamp_indices"])
        for contraction in TRACIN_DAS_METHODS:
            values = np.load(root / f"{contraction}.npy")
            if values.shape != totals[contraction].shape:
                raise ValueError(f"{root}/{contraction}.npy shape={values.shape}")
            totals[contraction] += values
    if sorted(covered) != list(range(len(DAS_TIMESTEPS))):
        raise ValueError("timestamp shards do not cover all 100 timestamps exactly")

    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    for contraction, method in TRACIN_DAS_METHODS.items():
        for query_position, query_id in enumerate(TRACIN_DAS_QUERY_IDS):
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
                    "timestamps": [int(value) for value in DAS_TIMESTEPS],
                    "noise_alignment": "one shared noise per timestamp for query and every train loss",
                    "query_scalar": "dot(epsilon_current, normalize(epsilon_next-epsilon_current))",
                    "parameter_projection": None,
                    "lr_weighted": TRACIN_USE_LR_WEIGHTS,
                    "timestamp_weight": 1.0 / len(DAS_TIMESTEPS),
                    "timestamp_shards": args.timestamp_shard_count,
                },
            )
        print(f"[saved] {method} q00-q09", flush=True)


if __name__ == "__main__":
    main()
