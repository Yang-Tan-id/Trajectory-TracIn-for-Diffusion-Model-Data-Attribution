"""Cache nine non-initial reference-trajectory states for relative noising."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from multiclean_das_config import *


def atomic_numpy(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    timestep_arrays = [
        np.load(Path(by_id[query_id]["dir"]) / "trajectory_t.npy")
        for query_id in MULTICLEAN_QUERY_IDS
    ]
    t_seq = timestep_arrays[0]
    if len(t_seq) != TRAJ_SNAPSHOTS or any(
        not np.array_equal(t_seq, values) for values in timestep_arrays
    ):
        raise ValueError("reference trajectory timestamp banks differ")
    anchor_timesteps = [int(t_seq[index]) for index in MULTICLEAN_ANCHOR_INDICES]
    if anchor_timesteps != sorted(anchor_timesteps, reverse=True):
        raise ValueError(f"anchor timesteps must decrease: {anchor_timesteps}")

    sample_trajectory = np.load(Path(by_id[0]["dir"]) / "trajectory_xt.npy")
    states = np.empty(
        (
            MULTICLEAN_ANCHOR_COUNT,
            len(MULTICLEAN_QUERY_IDS),
            *sample_trajectory.shape[2:],
        ),
        dtype=np.float32,
    )
    for family in MULTICLEAN_FAMILIES:
        query_ids = multiclean_query_ids(family)
        records = [by_id[query_id] for query_id in query_ids]
        if any(record["family"] != family for record in records):
            raise ValueError(f"query family mismatch for {family}")
        trajectories = [
            np.load(Path(record["dir"]) / "trajectory_xt.npy") for record in records
        ]
        for anchor_position, snapshot_index in enumerate(
            MULTICLEAN_ANCHOR_INDICES, start=1
        ):
            anchor_batch = np.concatenate(
                [trajectory[snapshot_index] for trajectory in trajectories], axis=0
            ).astype(np.float32, copy=False)
            states[anchor_position - 1, list(query_ids)] = anchor_batch
            timestep = int(t_seq[snapshot_index])
            targets = multiclean_anchor_targets(
                timestep, MULTICLEAN_ANCHOR_DAS_COUNTS[anchor_position - 1]
            )
            print(
                f"[multiclean] family={family} "
                f"anchor={anchor_position}/{MULTICLEAN_ANCHOR_COUNT} "
                f"snapshot={snapshot_index} timestep={timestep} "
                f"targets={len(targets)} range={targets[0]}..{targets[-1]}",
                flush=True,
            )
    MULTICLEAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_numpy(MULTICLEAN_CACHE_DIR / "trajectory_states.npy", states)
    with open(MULTICLEAN_CACHE_DIR / "info.json", "w") as handle:
        json.dump(
            {
                "query_ids": list(MULTICLEAN_QUERY_IDS),
                "families": list(MULTICLEAN_FAMILIES),
                "anchor_snapshot_indices": list(MULTICLEAN_ANCHOR_INDICES),
                "anchor_timesteps": anchor_timesteps,
                "anchor_das_timestamp_counts": list(MULTICLEAN_ANCHOR_DAS_COUNTS),
                "anchor_target_timesteps": [
                    list(multiclean_anchor_targets(timestep, count))
                    for timestep, count in zip(
                        anchor_timesteps, MULTICLEAN_ANCHOR_DAS_COUNTS
                    )
                ],
                "definition": "cached reference trajectory state x_t",
                "initial_t999_anchor": "skipped",
                "shape": list(states.shape),
            },
            handle,
            indent=2,
        )
    print(f"[saved] {MULTICLEAN_CACHE_DIR}", flush=True)


if __name__ == "__main__":
    main()
