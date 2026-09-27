"""Create five disjoint 10k stages and 192 balanced 50% LDS masks."""

import json

import numpy as np

from staged_lds_config import *


def main():
    if N_TRAIN != STAGE_COUNT * STAGE_SIZE:
        raise ValueError("50k must split exactly into five 10k stages")
    STAGED_PARTITION_DIR.mkdir(parents=True, exist_ok=True)
    STAGED_MASK_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(STAGED_PARTITION_SEED)
    stages = rng.permutation(N_TRAIN).reshape(STAGE_COUNT, STAGE_SIZE)
    membership = np.zeros(
        (STAGED_LDS_MASK_COUNT, N_TRAIN), dtype=np.uint8
    )
    mask_rng = np.random.default_rng(STAGED_MASK_SEED)
    for mask_id in range(STAGED_LDS_MASK_COUNT):
        for stage_indices in stages:
            chosen = mask_rng.choice(
                stage_indices, size=STAGED_LDS_PER_STAGE, replace=False
            )
            membership[mask_id, chosen] = 1
    if not np.all(membership.sum(axis=1) == STAGED_LDS_SUBSET_SIZE):
        raise RuntimeError("unbalanced total LDS mask size")
    for stage_indices in stages:
        if not np.all(membership[:, stage_indices].sum(axis=1) == STAGED_LDS_PER_STAGE):
            raise RuntimeError("unbalanced per-stage LDS mask size")
    np.save(STAGED_PARTITION_DIR / "stages.npy", stages)
    np.save(STAGED_MASK_DIR / "membership.npy", membership)
    manifest = [
        {
            "mask_id": mask_id,
            "subset_size": STAGED_LDS_SUBSET_SIZE,
            "per_stage_size": STAGED_LDS_PER_STAGE,
        }
        for mask_id in range(STAGED_LDS_MASK_COUNT)
    ]
    with open(STAGED_MASK_DIR / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    with open(STAGED_PARTITION_DIR / "metadata.json", "w") as handle:
        json.dump(
            {
                "partition_seed": STAGED_PARTITION_SEED,
                "mask_seed": STAGED_MASK_SEED,
                "stage_count": STAGE_COUNT,
                "stage_size": STAGE_SIZE,
                "stage_epochs": STAGE_EPOCHS,
                "mask_count": STAGED_LDS_MASK_COUNT,
                "per_stage_subset_size": STAGED_LDS_PER_STAGE,
                "total_subset_size": STAGED_LDS_SUBSET_SIZE,
            },
            handle,
            indent=2,
        )
    print(f"[saved] stages {stages.shape}; membership {membership.shape}")


if __name__ == "__main__":
    main()
