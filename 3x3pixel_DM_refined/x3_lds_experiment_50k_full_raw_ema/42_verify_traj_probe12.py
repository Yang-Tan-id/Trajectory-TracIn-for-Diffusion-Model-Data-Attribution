"""Verify prerequisites and print exact twelve-probe Traj semantics."""

import json
from pathlib import Path

import numpy as np

from attribution_one_query import model_paths
from traj_probe12_config import *


def main():
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    missing_ids = [qid for qid in TRAJ_PROBE12_QUERY_IDS if qid not in by_id]
    if missing_ids:
        raise ValueError(f"missing query ids: {missing_ids}")
    records = [by_id[qid] for qid in TRAJ_PROBE12_QUERY_IDS]
    if any(record["family"] != TRAJ_PROBE12_FAMILY for record in records):
        raise ValueError("q00-q19 must all be prompted")

    paths = model_paths(TRAJ_PROBE12_FAMILY)
    if len(paths) < TRAJ_PROBE12_CHECKPOINT_COUNT:
        raise ValueError(
            f"need at least {TRAJ_PROBE12_CHECKPOINT_COUNT} checkpoints, "
            f"found {len(paths)}"
        )
    for record in records:
        query_dir = Path(record["dir"])
        trajectory_path = query_dir / "trajectory_xt.npy"
        timestep_path = query_dir / "trajectory_t.npy"
        if not trajectory_path.is_file() or not timestep_path.is_file():
            raise FileNotFoundError(f"missing trajectory for q{record['query_id']:02d}")
        trajectory = np.load(trajectory_path, mmap_mode="r")
        timesteps = np.load(timestep_path, mmap_mode="r")
        if trajectory.shape[0] != TRAJ_SNAPSHOTS or timesteps.shape != (TRAJ_SNAPSHOTS,):
            raise ValueError(
                f"q{record['query_id']:02d}: trajectory={trajectory.shape}, "
                f"timesteps={timesteps.shape}"
            )

    print(f"queries = q00-q{TRAJ_PROBE12_QUERY_IDS[-1]:02d}")
    print(f"family = {TRAJ_PROBE12_FAMILY}")
    print(f"raw checkpoints used = first {TRAJ_PROBE12_CHECKPOINT_COUNT} of {len(paths)}")
    print(f"trajectory timestamps = {TRAJ_SNAPSHOTS}")
    print(f"output probes/timestamp = {TRAJ_PROBE12_NUM_PROBES}")
    print("probe sharing = same Gaussian probe bank across checkpoints and queries at a timestamp")
    print(f"parameter projection = CountSketch dim {TRACIN_PROJ_DIM}")
    print(f"train gradient MC = {TRACIN_TRAIN_MC}")
    print(f"train batch = {TRAJ_PROBE12_BATCH_SIZE}")
    print("query variants = raw; exact full-parameter query-gradient L2 normalized")
    print("train gradient normalization = disabled in both variants")
    print("termwise = sum_(t,c) eta_c/100 * mean_r(z[c,t,r]^2)")
    print("timestamp-wise = sum_t mean_r((sum_c eta_c/100 * z[c,t,r])^2)")
    print("per-timestamp 50k scores = saved for all four variants")
    timestamp_bytes = (
        len(TRAJ_PROBE12_VARIANTS)
        * len(TRAJ_PROBE12_QUERY_IDS)
        * TRAJ_SNAPSHOTS
        * N_TRAIN
        * 4
    )
    print(
        f"estimated timestamp-score storage = {timestamp_bytes / 1e9:.2f} GB merged "
        f"+ {timestamp_bytes / 1e9:.2f} GB restart shards"
    )
    print("[OK] twelve-probe Traj prerequisites verified")


if __name__ == "__main__":
    main()
