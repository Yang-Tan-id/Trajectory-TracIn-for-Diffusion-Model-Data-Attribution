"""Build ten x0 predictions from evenly spaced reference-trajectory states."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import x3pixel_DM_training as base
from attribution_one_query import build_model, cond_for, model_paths
from dataset_loader import ColorGridDataset
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
    records = [by_id[query_id] for query_id in MULTICLEAN_QUERY_IDS]
    if any(record["family"] != MULTICLEAN_FAMILY for record in records):
        raise ValueError("q00-q09 must be prompted")
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    model, _, checkpoint = build_model(
        model_paths(MULTICLEAN_FAMILY)[-1], "ema", device
    )
    model.eval()
    schedule = base.make_linear_schedule(T, device=device)
    conditions = torch.cat(
        [cond_for(record, dataset, device) for record in records], dim=0
    )
    trajectories = [
        np.load(Path(record["dir"]) / "trajectory_xt.npy") for record in records
    ]
    timestep_arrays = [
        np.load(Path(record["dir"]) / "trajectory_t.npy") for record in records
    ]
    t_seq = timestep_arrays[0]
    if len(t_seq) != TRAJ_SNAPSHOTS or any(
        not np.array_equal(t_seq, values) for values in timestep_arrays
    ):
        raise ValueError("reference trajectory timestamp banks differ")

    clean_banks = []
    with torch.no_grad():
        for anchor_position, snapshot_index in enumerate(
            MULTICLEAN_ANCHOR_INDICES, start=1
        ):
            timestep = int(t_seq[snapshot_index])
            states = torch.cat(
                [
                    torch.from_numpy(trajectory[snapshot_index]).to(
                        device=device, dtype=torch.float32
                    )
                    for trajectory in trajectories
                ],
                dim=0,
            )
            t_batch = torch.full(
                (len(records),), timestep, device=device, dtype=torch.long
            )
            predicted_noise = model(states, t_batch, conditions)
            alpha_bar = schedule.alpha_bars[timestep]
            predicted_clean = (
                states - torch.sqrt(1.0 - alpha_bar) * predicted_noise
            ) / torch.sqrt(alpha_bar)
            clean_banks.append(predicted_clean.cpu().numpy())
            print(
                f"[multiclean] anchor={anchor_position}/{MULTICLEAN_ANCHOR_COUNT} "
                f"snapshot={snapshot_index} timestep={timestep} "
                f"range=[{predicted_clean.min().item():.4g},"
                f"{predicted_clean.max().item():.4g}]",
                flush=True,
            )

    clean = np.stack(clean_banks, axis=0)
    MULTICLEAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_numpy(MULTICLEAN_CACHE_DIR / "predicted_clean.npy", clean)
    with open(MULTICLEAN_CACHE_DIR / "info.json", "w") as handle:
        json.dump(
            {
                "query_ids": list(MULTICLEAN_QUERY_IDS),
                "family": MULTICLEAN_FAMILY,
                "anchor_snapshot_indices": list(MULTICLEAN_ANCHOR_INDICES),
                "anchor_timesteps": [
                    int(t_seq[index]) for index in MULTICLEAN_ANCHOR_INDICES
                ],
                "definition": (
                    "x0_hat=(x_k-sqrt(1-alpha_bar_k)*eps_ema(x_k,k))"
                    "/sqrt(alpha_bar_k)"
                ),
                "parameter_source": "final EMA",
                "checkpoint_epoch": int(checkpoint.get("epoch", EPOCHS)),
                "shape": list(clean.shape),
            },
            handle,
            indent=2,
        )
    print(f"[saved] {MULTICLEAN_CACHE_DIR}", flush=True)


if __name__ == "__main__":
    main()
