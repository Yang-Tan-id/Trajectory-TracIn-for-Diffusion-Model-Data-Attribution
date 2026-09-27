"""Merge four timestamp shards into ten staged Traj score vectors."""

import json

import numpy as np

from staged_lds_config import *


def main():
    values = np.zeros((len(STAGED_QUERY_IDS), N_TRAIN), dtype=np.float64)
    covered = []
    for shard in range(4):
        root = STAGED_ATTR_DIR / "_traj_shards" / f"shard_{shard:02d}_of_04"
        with open(root / "done.json") as handle:
            covered.extend(json.load(handle)["timestamp_indices"])
        shard_values = np.load(root / "scores.npy")
        if shard_values.shape != values.shape:
            raise ValueError(f"{root}: {shard_values.shape}")
        values += shard_values
    if sorted(covered) != list(range(TRAJ_SNAPSHOTS)):
        raise ValueError("timestamp shards are incomplete or overlapping")
    for position, qid in enumerate(STAGED_QUERY_IDS):
        out = STAGED_ATTR_DIR / STAGED_TRAJ_METHOD / f"q{qid:02d}"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "scores.npy", values[position])
        with open(out / "info.json", "w") as handle:
            json.dump(
                {
                    "query_id": qid, "parameter_source": "raw", "target": "next",
                    "projection": "countsketch", "projection_dim": TRACIN_PROJ_DIM,
                    "contraction": "timestamp_sum_squared", "train_mc": TRACIN_TRAIN_MC,
                    "stage_aligned_train_pool": True, "stage_size": STAGE_SIZE,
                    "checkpoint_count": STAGED_CHECKPOINT_COUNT, "transition_count": 49,
                },
                handle, indent=2,
            )
    print(f"[done] {STAGED_TRAJ_METHOD} q00-q09", flush=True)


if __name__ == "__main__":
    main()
