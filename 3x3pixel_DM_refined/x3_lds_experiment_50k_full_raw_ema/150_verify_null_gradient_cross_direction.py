"""Verify null-to-next checkpoint cross-direction JVP prerequisites."""

import json
import argparse

import torch

from null_gradient_cross_direction_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-prompt", choices=("original", "random"), default="original"
    )
    args = parser.parse_args()
    paths = {
        "null": ngcd_checkpoint_path(NGCD_NULL_EPOCH),
        "next": ngcd_checkpoint_path(NGCD_NEXT_EPOCH),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "model_state" not in checkpoint:
            raise KeyError(f"{name} checkpoint has no model_state")
    payload = {
        "null_checkpoint": str(paths["null"]),
        "next_checkpoint": str(paths["next"]),
        "datapoint_indices": list(nsdl_datapoint_indices()),
        "training_gradient": (
            "mean null-checkpoint diffusion-loss gradient over all 1000 "
            "timestamps using one fixed positive noise direction"
        ),
        "primary_prediction": "J_null(x_minus) @ (-loss_gradient_positive)",
        "primary_target": "epsilon_next(x_minus)-epsilon_null(x_minus)",
        "loss_prompt": "original datapoint prompt",
        "evaluation_prompt": (
            "a deterministic random valid training prompt different from the loss prompt"
            if args.evaluation_prompt == "random"
            else "original datapoint prompt"
        ),
        "controls": [
            "same-direction input",
            "J_null @ (theta_next-theta_null) checkpoint-delta linearization",
        ],
        "timestamp_batches": T // NSDL_UPDATE_BATCH_SIZE,
        "timestamps_per_batch": NSDL_UPDATE_BATCH_SIZE,
    }
    print(json.dumps(payload, indent=2))
    print("[OK] null-gradient cross-direction experiment verified")


if __name__ == "__main__":
    main()
