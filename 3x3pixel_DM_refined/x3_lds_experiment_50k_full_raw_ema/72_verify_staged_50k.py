"""Verify staged experiment configuration and report available artifacts."""

import json

import numpy as np

from staged_lds_config import *


def main():
    if not BASE_CSV.is_file():
        raise FileNotFoundError(BASE_CSV)
    stages = np.load(STAGED_PARTITION_DIR / "stages.npy")
    membership = np.load(STAGED_MASK_DIR / "membership.npy")
    assert stages.shape == (5, 10000)
    assert np.array_equal(np.sort(stages.reshape(-1)), np.arange(N_TRAIN))
    assert membership.shape == (192, 50000)
    assert np.all(membership.sum(axis=1) == 25000)
    assert all(np.all(membership[:, stage].sum(axis=1) == 5000) for stage in stages)
    print(f"root = {STAGED_ROOT}")
    print("training = five disjoint 10k stages x 40 epochs, continuous AdamW/EMA/LR")
    print(f"base checkpoints = {STAGED_CHECKPOINT_COUNT} (every {STAGED_SAVE_EVERY} epochs)")
    print("LDS = 192 masks x (5 stages * 5000 points) = 25000 points/model")
    print("queries = prompted q00-q09, regenerated from staged final EMA")
    print("Traj = raw next, CountSketch 4096, MC10, timestamp-sum-square")
    print("Traj train pool = interval's active 10k stage")
    print("DAS = final EMA only, all 50000, 100 timestamps x MC10")
    print(f"base final exists = {staged_base_checkpoint(STAGED_EPOCHS).is_file()}")
    print(f"subset finals = {sum(staged_subset_checkpoint(i).is_file() for i in range(STAGED_LDS_MASK_COUNT))}/192")
    print("[OK] staged 50k configuration verified")


if __name__ == "__main__":
    main()
