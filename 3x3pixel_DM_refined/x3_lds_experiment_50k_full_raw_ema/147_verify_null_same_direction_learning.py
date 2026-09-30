"""Verify prerequisites and print the exact ten-point experiment contract."""

import json

import torch

from null_same_direction_learning_config import *


def main():
    checkpoint_path = nsdl_checkpoint_path()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if T % NSDL_UPDATE_COUNT:
        raise ValueError("T must split exactly across four updates")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    required = {"model_state", "optimizer_state"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise KeyError(f"null checkpoint missing keys: {missing}")
    indices = nsdl_datapoint_indices()
    if len(indices) != NSDL_DATAPOINT_COUNT or len(set(indices)) != len(indices):
        raise ValueError("invalid selected datapoint indices")
    payload = {
        "null_checkpoint": str(checkpoint_path),
        "null_epoch": NSDL_NULL_EPOCH,
        "datapoint_indices": list(indices),
        "timestamps": len(NSDL_TIMESTAMPS),
        "updates": NSDL_UPDATE_COUNT,
        "timestamps_per_update": NSDL_UPDATE_BATCH_SIZE,
        "training_noise": "one fixed direction and radius across all timestamps",
        "evaluation_directions": {
            "same": 1,
            "opposite": 1,
            "random_same_norm": NSDL_RANDOM_DIRECTION_COUNT,
        },
        "evaluation": (
            "direct predicted-noise after-minus-before; MSE/RMSE/mean-L2/"
            "max-abs/signed-direction-projection; no loss comparison"
        ),
        "optimizer": "restored checkpoint AdamW state and checkpoint LR",
    }
    print(json.dumps(payload, indent=2))
    print("[OK] null-model same-direction experiment verified")


if __name__ == "__main__":
    main()
