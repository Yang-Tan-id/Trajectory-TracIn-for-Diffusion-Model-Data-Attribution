"""Verify prerequisites for the checkpoint-transition diagnostic."""

import json

import numpy as np
import torch

from attribution_one_query import model_paths
from checkpoint_transition_diagnostic_config import *
from forward_loss_alignment_config import replay_noise_path, replay_t_path


def main():
    print("pairs = epoch 4->8 through epoch 196->200 (49 transitions)")
    print(f"queries = q{CTD_QUERY_IDS[0]:02d}-q{CTD_QUERY_IDS[-1]:02d}")
    print(f"trajectory timestamps/query = {TRAJ_SNAPSHOTS}")
    print("AdamW control = restored optimizer + exact batch order/t/noise/LR replay")
    print("current control = fixed target-checkpoint gradient sum, no projection")

    expected_t = (50, N_TRAIN, CTD_EVENTS_PER_INTERVAL)
    expected_noise = expected_t + (3, 3, 3)
    for path, expected in (
        (replay_t_path(), expected_t),
        (replay_noise_path(), expected_noise),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path}; run 24_prepare_forward_loss_alignment.py "
                "--skip-queries --skip-baseline"
            )
        value = np.load(path, mmap_mode="r")
        if value.shape != expected:
            raise ValueError(f"{path}: shape={value.shape}, expected={expected}")
        print(f"replay cache = {path} {value.shape}")

    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    if len(manifest) < max(CTD_QUERY_IDS) + 1:
        raise ValueError("query manifest does not contain q00-q09")

    for family in FAMILIES:
        paths = model_paths(family)
        if len(paths) != 50:
            raise ValueError(f"{family}: found {len(paths)} checkpoints, expected 50")
        for path in (paths[0], paths[-1]):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            required = {"model_state", "optimizer_state", "global_step"}
            missing = required - set(payload)
            if missing:
                raise ValueError(f"{path}: missing {sorted(missing)}")
        print(f"{family} checkpoints = 50 with optimizer state")
    print("[OK] checkpoint-transition diagnostic prerequisites verified")


if __name__ == "__main__":
    main()
