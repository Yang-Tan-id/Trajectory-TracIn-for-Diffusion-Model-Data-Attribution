"""Merge four unprojected first-order next-checkpoint Traj timestamp shards."""

import argparse
import json

import numpy as np

from exp_config import ATTR_DIR, N_TRAIN, QUERY_DIR
from run_exact_traj_next_bank import (
    EXACT_METHODS,
    EXACT_SCORE_CONTRACT_VERSION,
    EXACT_SHARD_NAMESPACE,
    METHOD,
    SCORE_CONTRACT_VERSION,
    SHARD_NAMESPACE,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("prompted", "unprompted"), required=True)
    parser.add_argument("--timestamp-shard-count", type=int, required=True)
    parser.add_argument("--all-contractions", action="store_true")
    args = parser.parse_args()

    with open(QUERY_DIR / "manifest.json") as handle:
        records = [item for item in json.load(handle) if item["family"] == args.family]
    expected_ids = [int(item["query_id"]) for item in records]
    shard_namespace = EXACT_SHARD_NAMESPACE if args.all_contractions else SHARD_NAMESPACE
    score_contract_version = (
        EXACT_SCORE_CONTRACT_VERSION if args.all_contractions else SCORE_CONTRACT_VERSION
    )
    contractions = tuple(EXACT_METHODS) if args.all_contractions else ("linear",)
    root = ATTR_DIR / shard_namespace / args.family
    merged = {contraction: None for contraction in contractions}
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
        if metadata.get("score_contract_version") != score_contract_version:
            raise ValueError(f"score contract mismatch in {shard}")
        if bool(metadata.get("true_full_dot", False)) != args.all_contractions:
            raise ValueError(f"full-dot mode mismatch in {shard}")
        expected_shape = (len(records), N_TRAIN)
        for contraction in contractions:
            values = np.load(shard / f"{contraction}.npy")
            if values.shape != expected_shape:
                raise ValueError(
                    f"{shard}/{contraction} has shape {values.shape}, "
                    f"expected {expected_shape}"
                )
            merged[contraction] = (
                values
                if merged[contraction] is None
                else merged[contraction] + values
            )
        covered_timestamps.extend(int(value) for value in metadata["timestamp_indices"])
        shard_metadata.append(metadata)
    if sorted(covered_timestamps) != list(range(100)):
        raise ValueError("timestamp shards do not cover 0..99 exactly")

    for query_index, record in enumerate(records):
        for contraction in contractions:
            method = EXACT_METHODS[contraction] if args.all_contractions else METHOD
            out = ATTR_DIR / method / f"q{int(record['query_id']):02d}"
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "scores.npy", merged[contraction][query_index])
            with open(out / "info.json", "w") as handle:
                json.dump(
                    {
                        "query": record,
                        "order": "first",
                        "target": "next",
                        "param_source": "raw",
                        "projection": None,
                        "contraction": contraction,
                        "backend": "shared_family_full_gradient_matrix",
                        "checkpoint_count": 50,
                        "checkpoint_transitions": 49,
                        "timestamp_shards": args.timestamp_shard_count,
                        "merged_timestamp_indices": sorted(covered_timestamps),
                        "train_mc": shard_metadata[0]["train_mc"],
                        "learning_rate_source": "current_checkpoint",
                        "score_contract_version": score_contract_version,
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
