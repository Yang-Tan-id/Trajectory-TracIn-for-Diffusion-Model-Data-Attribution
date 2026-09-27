"""Verify prerequisites for continuous checkpoint-0 reference learning."""

import argparse

import torch

from checkpoint0_continuous_reference_learning_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    path = carl_checkpoint_path(C0RL_INITIAL_EPOCH)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in (
        "model_state",
        "optimizer_state",
        "global_step",
        "learning_rate_at_checkpoint",
        "config",
    ):
        if key not in checkpoint:
            raise KeyError(f"{path} has no {key}")
    for query_id in CARL_QUERY_IDS:
        for filename in (
            "trajectory_xt_1000.npy",
            "target_eps_ema_1000.npy",
            "trajectory_t_1000.npy",
        ):
            query_path = FLA_QUERY_DIR / f"q{query_id:02d}" / filename
            if not query_path.is_file():
                raise FileNotFoundError(query_path)
    print(f"initial checkpoint = epoch {C0RL_INITIAL_EPOCH} (saved index 0)")
    print(f"initial global step = {checkpoint['global_step']}")
    print("optimizer = restored checkpoint AdamW m/v/step/param-groups")
    print("LR = continuation of original warmup/cosine schedule")
    print("reference batches = 4 consecutive chunks x 250 states, cycled")
    print(f"stop = full reference loss <= {C0RL_TARGET_FRACTION:.0%} of start")
    print(f"maximum updates = {C0RL_MAX_STEPS}")
    print(f"queries = q00-q09; paired training MC = {CARL_TRAIN_MC}")
    print("[OK] checkpoint-0 continuous reference-learning prerequisites verified")


if __name__ == "__main__":
    main()
