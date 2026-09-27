"""Verify checkpoint-AdamW reference-learning prerequisites."""

import argparse
import json

import numpy as np
import torch

from checkpoint_adamw_reference_learning_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    if DDIM_STEPS != 1000 or CARL_REFERENCE_BATCH_SIZE * CARL_REFERENCE_STEPS != 1000:
        raise ValueError("reference trajectory must split exactly into four batches of 250")
    with open(QUERY_DIR / "manifest.json") as handle:
        records = {int(item["query_id"]): item for item in json.load(handle)}
    for query_id in CARL_QUERY_IDS:
        if records[query_id]["family"] != CARL_FAMILY:
            raise ValueError(f"q{query_id:02d} is not prompted")
        for filename in (
            "trajectory_xt_1000.npy",
            "target_eps_ema_1000.npy",
            "trajectory_t_1000.npy",
        ):
            path = FLA_QUERY_DIR / f"q{query_id:02d}" / filename
            if not path.is_file():
                raise FileNotFoundError(path)
        shapes = {
            filename: np.load(
                FLA_QUERY_DIR / f"q{query_id:02d}" / filename,
                mmap_mode="r",
            ).shape
            for filename in (
                "trajectory_xt_1000.npy",
                "target_eps_ema_1000.npy",
                "trajectory_t_1000.npy",
            )
        }
        if shapes["trajectory_xt_1000.npy"][0] != DDIM_STEPS:
            raise ValueError(f"q{query_id:02d} trajectory shape={shapes}")
        if shapes["target_eps_ema_1000.npy"] != shapes["trajectory_xt_1000.npy"]:
            raise ValueError(f"q{query_id:02d} target shape={shapes}")
        if shapes["trajectory_t_1000.npy"] != (DDIM_STEPS,):
            raise ValueError(f"q{query_id:02d} timestamp shape={shapes}")

    learning_rates = []
    for epoch in CARL_CHECKPOINT_EPOCHS:
        path = carl_checkpoint_path(epoch)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for key in (
            "model_state",
            "optimizer_state",
            "learning_rate_at_checkpoint",
            "config",
        ):
            if key not in payload:
                raise KeyError(f"{path} has no {key}")
        optimizer_states = payload["optimizer_state"].get("state", {})
        if not optimizer_states:
            raise ValueError(f"empty optimizer state in {path}")
        for state in optimizer_states.values():
            if "exp_avg" not in state or "exp_avg_sq" not in state:
                raise ValueError(f"incomplete AdamW moments in {path}")
        learning_rates.append(float(payload["learning_rate_at_checkpoint"]))

    print(f"queries = q00-q49 ({len(CARL_QUERY_IDS)} prompted)")
    print(f"checkpoints = {len(CARL_CHECKPOINT_EPOCHS)}")
    print("reference updates/checkpoint = 4 AdamW steps over consecutive 250-state batches")
    print("optimizer = checkpoint m/v/step/param-groups; checkpoint LR; original clipping")
    print(f"training loss MC = {CARL_TRAIN_MC} paired before/after")
    print("scores = raw decrease, difference-over-sum")
    print("checkpoint aggregation = uniform, normalized LR-weighted")
    print(
        f"checkpoint LR range/sum = [{min(learning_rates):.6g}, "
        f"{max(learning_rates):.6g}] / {sum(learning_rates):.6g}"
    )
    print("[OK] checkpoint-AdamW reference-learning prerequisites verified")


if __name__ == "__main__":
    main()
