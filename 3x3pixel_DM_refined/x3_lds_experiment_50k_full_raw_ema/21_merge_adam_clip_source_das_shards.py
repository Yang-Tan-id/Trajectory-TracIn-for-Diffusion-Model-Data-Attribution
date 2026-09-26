"""Merge Adam/clipping-aware SOURCE-DAS timestamp shards."""

import argparse
import json

import numpy as np

from adam_clip_source_das_config import *
from run_exact_traj_next_bank import atomic_json_save, atomic_numpy_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--timestamp-shard-count", type=int, required=True)
    args = parser.parse_args()
    merged = {variant: None for variant in ADAM_CLIP_SOURCE_METHODS}
    query_ids = None
    covered = []
    shard_metadata = []
    for shard_index in range(args.timestamp_shard_count):
        root = (
            ADAM_CLIP_SOURCE_SHARD_ROOT
            / args.family
            / f"shard_{shard_index:02d}_of_{args.timestamp_shard_count:02d}"
        )
        with open(root / "done.json") as handle:
            metadata = json.load(handle)
        if query_ids is None:
            query_ids = metadata["query_ids"]
        elif metadata["query_ids"] != query_ids:
            raise ValueError("query IDs differ between timestamp shards")
        for variant in ADAM_CLIP_SOURCE_METHODS:
            values = np.load(root / f"scores_{variant}.npy").astype(np.float64)
            if values.shape != (len(query_ids), N_TRAIN):
                raise ValueError(f"unexpected {variant} shape {values.shape} in {root}")
            if merged[variant] is None:
                merged[variant] = values
            else:
                merged[variant] += values
        covered.extend(metadata["timestamp_indices"])
        shard_metadata.append(metadata)
    if sorted(covered) != list(range(TRAJ_SNAPSHOTS)):
        raise ValueError(f"timestamp coverage is not 0..{TRAJ_SNAPSHOTS-1}")

    for variant, method in ADAM_CLIP_SOURCE_METHODS.items():
        for row, query_id in enumerate(query_ids):
            output = ATTR_DIR / method / f"q{int(query_id):02d}"
            output.mkdir(parents=True, exist_ok=True)
            atomic_numpy_save(output / "scores.npy", merged[variant][row])
            atomic_json_save(
                output / "meta.json",
                {
                    "method": method,
                    "variant": variant,
                    "query_id": int(query_id),
                    "family": args.family,
                    "score_sign": "nonnegative_jvp_norm_squared",
                    "final_parameter_source": "raw",
                    "query_trajectory_source": "cached_final_ema_ddim",
                    "curvature_checkpoints_per_segment": [
                        list(values)
                        for values in ADAM_CLIP_SOURCE_CURVATURE_EPOCHS_PER_SEGMENT
                    ],
                    "preconditioner_checkpoints_per_segment": [
                        list(values)
                        for values in ADAM_CLIP_SOURCE_P_CHECKPOINT_EPOCHS
                    ],
                    "train_mc": ADAM_CLIP_SOURCE_TRAIN_MC,
                    "timestamp_aligned": True,
                    "adam_second_moment": True,
                    "clip_jacobian": "frozen_batch_scale",
                    "shards": shard_metadata,
                },
            )
        print(
            f"[done] Adam/clipping SOURCE merge family={args.family} "
            f"variant={variant} queries={len(query_ids)} shape={merged[variant].shape}",
            flush=True,
        )


if __name__ == "__main__":
    main()
