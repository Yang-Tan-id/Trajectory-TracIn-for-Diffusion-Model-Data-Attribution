"""Merge four unprojected first-order next-checkpoint Traj timestamp shards."""

import argparse
import json

import numpy as np

from exp_config import ATTR_DIR, N_TRAIN, QUERY_DIR
from run_exact_traj_next_bank import (
    METHOD,
    SCORE_CONTRACT_VERSION,
    SHARD_NAMESPACE,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("prompted", "unprompted"), required=True)
    parser.add_argument("--timestamp-shard-count", type=int, required=True)
    args = parser.parse_args()

    with open(QUERY_DIR / "manifest.json") as handle:
        records = [item for item in json.load(handle) if item["family"] == args.family]
    expected_ids = [int(item["query_id"]) for item in records]
    root = ATTR_DIR / SHARD_NAMESPACE / args.family
    merged = None
    covered_timestamps = []
    shard_metadata = []
    for shard_index in range(args.timestamp_shard_count):
        shard = root / f"shard_{shard_index:02d}_of_{args.timestamp_shard_count:02d}"
        with open(shard / "done.json") as handle:
            metadata = json.load(handle)
        if metadata["query_ids"] != expected_ids:
            raise ValueError(f"query IDs mismatch in {shard}")
        if metadata.get("projection") is not None:
            raise ValueError(f"{shard} is not an exact/no-projection shard")
        if metadata.get("score_contract_version") != SCORE_CONTRACT_VERSION:
            raise ValueError(f"score contract mismatch in {shard}")
        values = np.load(shard / "linear.npy")
        expected_shape = (len(records), N_TRAIN)
        if values.shape != expected_shape:
            raise ValueError(f"{shard} has shape {values.shape}, expected {expected_shape}")
        merged = values if merged is None else merged + values
        covered_timestamps.extend(int(value) for value in metadata["timestamp_indices"])
        shard_metadata.append(metadata)
    if sorted(covered_timestamps) != list(range(100)):
        raise ValueError("timestamp shards do not cover 0..99 exactly")

    for query_index, record in enumerate(records):
        out = ATTR_DIR / METHOD / f"q{int(record['query_id']):02d}"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "scores.npy", merged[query_index])
        with open(out / "info.json", "w") as handle:
            json.dump(
                {
                    "query": record,
                    "order": "first",
                    "target": "next",
                    "param_source": "raw",
                    "projection": None,
                    "contraction": "linear",
                    "backend": "shared_family_full_gradient_matrix",
                    "timestamp_shards": args.timestamp_shard_count,
                    "merged_timestamp_indices": sorted(covered_timestamps),
                    "train_mc": shard_metadata[0]["train_mc"],
                    "learning_rate_source": "current_checkpoint",
                    "score_contract_version": SCORE_CONTRACT_VERSION,
                    "train_noise_seed_contract": shard_metadata[0][
                        "train_noise_seed_contract"
                    ],
                    "score_scale": shard_metadata[0]["score_scale"],
                },
                handle,
                indent=2,
            )
    print(
        f"[done] merged exact/no-projection Traj next {args.family} "
        f"from {args.timestamp_shard_count} shards",
        flush=True,
    )


if __name__ == "__main__":
    main()
