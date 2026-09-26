"""Merge timestamp-aligned SOURCE-DAS shards into the 100-query score bank."""

import argparse
import json

import numpy as np

from exp_config import *
from run_exact_traj_next_bank import atomic_json_save, atomic_numpy_save
from source_das_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--timestamp-shard-count", type=int, required=True)
    args = parser.parse_args()
    merged = None
    query_ids = None
    covered = []
    shard_metadata = []
    for shard_index in range(args.timestamp_shard_count):
        root = (
            SOURCE_DAS_SHARD_ROOT / args.family
            / f"shard_{shard_index:02d}_of_{args.timestamp_shard_count:02d}"
        )
        with open(root / "done.json") as handle:
            metadata = json.load(handle)
        values = np.load(root / "scores.npy").astype(np.float64)
        if values.shape[1] != N_TRAIN:
            raise ValueError(f"unexpected shard shape {values.shape} in {root}")
        if query_ids is None:
            query_ids = metadata["query_ids"]
            merged = values
        else:
            if metadata["query_ids"] != query_ids:
                raise ValueError("query IDs differ between timestamp shards")
            merged += values
        covered.extend(metadata["timestamp_indices"])
        shard_metadata.append(metadata)
    if sorted(covered) != list(range(TRAJ_SNAPSHOTS)):
        raise ValueError(f"timestamp coverage is not 0..{TRAJ_SNAPSHOTS-1}")
    for row, query_id in enumerate(query_ids):
        output = ATTR_DIR / SOURCE_DAS_METHOD / f"q{int(query_id):02d}"
        output.mkdir(parents=True, exist_ok=True)
        atomic_numpy_save(output / "scores.npy", merged[row])
        atomic_json_save(
            output / "meta.json",
            {
                "method": SOURCE_DAS_METHOD,
                "query_id": int(query_id),
                "family": args.family,
                "score_sign": "nonnegative_exact_jvp_norm_squared",
                "checkpoints": list(SOURCE_DAS_CHECKPOINT_EPOCHS),
                "train_mc": SOURCE_DAS_TRAIN_MC,
                "timestamp_aligned": True,
                "shards": shard_metadata,
            },
        )
    print(
        f"[done] SOURCE-DAS merge family={args.family} "
        f"queries={len(query_ids)} shape={merged.shape}",
        flush=True,
    )


if __name__ == "__main__":
    main()
