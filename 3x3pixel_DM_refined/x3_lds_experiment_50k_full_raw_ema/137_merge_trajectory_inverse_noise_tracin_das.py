"""Merge trajectory inverse-noise TracIn-DAS timestamp shards."""

import argparse
import json

import numpy as np

from trajectory_inverse_noise_tracin_das_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    args = parser.parse_args()
    totals = {
        contraction: np.zeros(
            (len(INVERSE_NOISE_QUERY_IDS), N_TRAIN), dtype=np.float64
        )
        for contraction in INVERSE_NOISE_CONTRACTIONS
    }
    covered = []
    metadata = []
    for shard_index in range(args.timestamp_shard_count):
        root = inverse_noise_shard_root(shard_index, args.timestamp_shard_count)
        with open(root / "done.json") as handle:
            info = json.load(handle)
        metadata.append(info)
        if info["query_ids"] != list(INVERSE_NOISE_QUERY_IDS):
            raise ValueError(f"query IDs differ in {root}")
        if not info["endpoint_excluded"]:
            raise ValueError(f"endpoint was not excluded in {root}")
        covered.extend(int(value) for value in info["timestamp_indices"])
        for contraction in INVERSE_NOISE_CONTRACTIONS:
            values = np.load(root / f"{contraction}.npy")
            if values.shape != totals[contraction].shape:
                raise ValueError(f"score shape differs in {root}: {values.shape}")
            totals[contraction] += values
    if sorted(covered) != list(range(99)):
        raise ValueError(f"timestamp shards do not cover indices 0..98: {covered}")

    with open(QUERY_DIR / "manifest.json") as handle:
        by_id = {int(record["query_id"]): record for record in json.load(handle)}
    for query_position, query_id in enumerate(INVERSE_NOISE_QUERY_IDS):
        for contraction, method in INVERSE_NOISE_METHODS.items():
            output = ATTR_DIR / method / f"q{query_id:02d}"
            output.mkdir(parents=True, exist_ok=True)
            np.save(output / "scores.npy", totals[contraction][query_position])
            with open(output / "info.json", "w") as handle:
                json.dump(
                    {
                        "query": by_id[query_id],
                        "method": method,
                        "contraction": contraction,
                        "query_ids": list(INVERSE_NOISE_QUERY_IDS),
                        "checkpoint_direction": "forward/next",
                        "checkpoint_count": 50,
                        "checkpoint_pairs": metadata[0]["checkpoint_pairs"],
                        "parameter_source": "raw",
                        "projection_dim": INVERSE_NOISE_PROJ_DIM,
                        "included_timestamp_indices": list(range(99)),
                        "endpoint_excluded": True,
                        "timestamp_weight": 1.0 / 99.0,
                        "train_loss_is_query_dependent": True,
                        "loss_definition": metadata[0]["loss_definition"],
                    },
                    handle,
                    indent=2,
                )
    print("[done] merged trajectory inverse-noise TracIn-DAS q00-q09")


if __name__ == "__main__":
    main()
