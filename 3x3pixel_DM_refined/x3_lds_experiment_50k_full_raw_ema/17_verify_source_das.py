"""Preflight checks for timestamp-aligned SOURCE-DAS."""

import json
from pathlib import Path

import numpy as np

from source_das_config import *


def main():
    source_file = SOURCE_DAS_SIMPLE_INFLUENCE_ROOT / "src" / "source.py"
    if not source_file.is_file():
        raise FileNotFoundError(source_file)
    if not BASE_CSV.is_file():
        raise FileNotFoundError(BASE_CSV)
    with open(QUERY_DIR / "manifest.json") as handle:
        queries = json.load(handle)
    if len(queries) != 100:
        raise ValueError(f"expected 100 queries, found {len(queries)}")
    for family in FAMILIES:
        for epoch in SOURCE_DAS_CHECKPOINT_EPOCHS:
            path = MODEL_DIR / "base" / family / f"epoch_{epoch:04d}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
    timestamp_reference = None
    for query in queries:
        query_dir = Path(query["dir"])
        trajectory = np.load(query_dir / "trajectory_xt.npy", mmap_mode="r")
        timestamps = np.load(query_dir / "trajectory_t.npy")
        if trajectory.shape != (TRAJ_SNAPSHOTS, 1, 3, 3, 3):
            raise ValueError(f"unexpected trajectory shape {trajectory.shape}")
        if timestamp_reference is None:
            timestamp_reference = timestamps
        elif not np.array_equal(timestamp_reference, timestamps):
            raise ValueError("query timestamp arrays differ")
    print(f"simple-influence = {SOURCE_DAS_SIMPLE_INFLUENCE_ROOT}")
    print(f"method = {SOURCE_DAS_METHOD}")
    print(f"checkpoints = {SOURCE_DAS_CHECKPOINT_EPOCHS}")
    print(f"segments = {SOURCE_DAS_NUM_SEGMENTS}")
    print(f"iters/segment = {source_das_iters_per_segment()}")
    print(f"mean LR/segment = {source_das_lrs_per_segment()}")
    print(f"train/curvature MC = {SOURCE_DAS_TRAIN_MC}")
    print(f"trajectory timestamps = {len(timestamp_reference)}")
    print(f"influence modules = {SOURCE_DAS_INFLUENCE_MODULES}")
    print("[OK] timestamp-aligned SOURCE-DAS prerequisites verified")


if __name__ == "__main__":
    main()
