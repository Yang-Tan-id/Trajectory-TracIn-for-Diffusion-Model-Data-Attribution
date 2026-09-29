"""Predict one clean endpoint from every cached reference-trajectory state."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import x3pixel_DM_training as base
from attribution_one_query import build_model, cond_for, model_paths
from dataset_loader import ColorGridDataset
from diagonal_clean_das_config import *


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
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    records = [by_id[query_id] for query_id in DIAGONAL_CLEAN_QUERY_IDS]
    if any(record["family"] != DIAGONAL_CLEAN_FAMILY for record in records):
        raise ValueError("q00-q09 must all be prompted")
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    model, _, checkpoint = build_model(
        model_paths(DIAGONAL_CLEAN_FAMILY)[-1], "ema", device
    )
    schedule = base.make_linear_schedule(T, device=device)
    conditions = torch.cat(
        [cond_for(record, dataset, device) for record in records], dim=0
    )
    trajectories = [
        np.load(Path(record["dir"]) / "trajectory_xt.npy") for record in records
    ]
    timestamp_arrays = [
        np.load(Path(record["dir"]) / "trajectory_t.npy") for record in records
    ]
    timestamps = timestamp_arrays[0]
    if len(timestamps) != 100 or any(
        not np.array_equal(timestamps, values) for values in timestamp_arrays
    ):
        raise ValueError("expected one shared 100-timestamp trajectory bank")

    clean_banks = []
    with torch.no_grad():
        for snapshot_index, timestep_raw in enumerate(timestamps):
            timestep = int(timestep_raw)
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
            if snapshot_index == 0 or (snapshot_index + 1) % 10 == 0:
                print(
                    f"[diagonal-clean] snapshot={snapshot_index+1}/100 t={timestep} "
                    f"range=[{predicted_clean.min().item():.4g},"
                    f"{predicted_clean.max().item():.4g}]",
                    flush=True,
                )
    clean = np.stack(clean_banks, axis=0)
    DIAGONAL_CLEAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_numpy(DIAGONAL_CLEAN_CACHE_DIR / "predicted_clean.npy", clean)
    atomic_numpy(
        DIAGONAL_CLEAN_CACHE_DIR / "trajectory_t.npy",
        np.asarray(timestamps, dtype=np.int64),
    )
    with open(DIAGONAL_CLEAN_CACHE_DIR / "info.json", "w") as handle:
        json.dump(
            {
                "query_ids": list(DIAGONAL_CLEAN_QUERY_IDS),
                "family": DIAGONAL_CLEAN_FAMILY,
                "snapshot_indices": list(range(100)),
                "trajectory_timesteps": [int(value) for value in timestamps],
                "definition": (
                    "x0_hat_k=(x_k-sqrt(1-alpha_bar_k)*eps_ema(x_k,k))"
                    "/sqrt(alpha_bar_k)"
                ),
                "pairing": "predicted clean from snapshot k is scored only at its own noise level k",
                "parameter_source": "final EMA",
                "checkpoint_epoch": int(checkpoint.get("epoch", EPOCHS)),
                "shape": list(clean.shape),
            },
            handle,
            indent=2,
        )
    print(f"[saved] {DIAGONAL_CLEAN_CACHE_DIR}", flush=True)


if __name__ == "__main__":
    main()
