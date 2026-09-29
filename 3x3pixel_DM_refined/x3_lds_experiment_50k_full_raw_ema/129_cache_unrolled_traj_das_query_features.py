"""Cache projected query features from a differentiable full DDIM unroll."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, grad

import x3pixel_DM_training as base
from attribution_one_query import _project_batched_grads, build_model, cond_for, model_paths
from dataset_loader import ColorGridDataset
from unrolled_traj_das_config import *
from x3_endpoint_das_jax_logic_pytorch import build_countsketch_specs, make_torch_generator


def atomic_numpy(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    feature_path = UNROLLED_TRAJ_DAS_CACHE_DIR / "query_features.npy"
    info_path = UNROLLED_TRAJ_DAS_CACHE_DIR / "info.json"
    if feature_path.is_file() and info_path.is_file() and not args.force:
        print(f"[skip] existing query-feature cache {UNROLLED_TRAJ_DAS_CACHE_DIR}")
        return
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    records = [by_id[query_id] for query_id in UNROLLED_TRAJ_DAS_QUERY_IDS]
    if any(record["family"] != UNROLLED_TRAJ_DAS_FAMILY for record in records):
        raise ValueError("unrolled trajectory DAS q00-q09 must be prompted")
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    model, _, checkpoint = build_model(
        model_paths(UNROLLED_TRAJ_DAS_FAMILY)[-1], "ema", device
    )
    named = dict(model.named_parameters())
    names = tuple(named)
    active = tuple(named.values())
    specs = build_countsketch_specs(
        list(active),
        UNROLLED_TRAJ_DAS_PROJECTION_DIM,
        device=device,
        seed_parts=UNROLLED_TRAJ_DAS_PROJECTION_SEED,
    )
    schedule = base.make_linear_schedule(T, device=device)
    ddim_timesteps = torch.linspace(
        T - 1, 0, DDIM_STEPS, device=device
    ).long()
    save_steps = np.linspace(
        0, DDIM_STEPS - 1, TRAJ_SNAPSHOTS, dtype=np.int64
    ).tolist()
    save_position = {int(step): position for position, step in enumerate(save_steps)}
    trajectory_weight_sqrt = 1.0 / math.sqrt(float(TRAJ_SNAPSHOTS))

    def trajectory_probe_scalar(parameter_dict, x, condition, probes):
        value = x.new_zeros(())
        if 0 in save_position:
            value = value + trajectory_weight_sqrt * (
                x * probes[save_position[0]]
            ).sum()
        for step_index in range(len(ddim_timesteps) - 1):
            timestep = ddim_timesteps[step_index].repeat(x.shape[0])
            previous_timestep = int(ddim_timesteps[step_index + 1].item())
            predicted_noise = functional_call(
                model, parameter_dict, (x, timestep, condition)
            )
            alpha_bar = schedule.alpha_bars[timestep].view(-1, 1, 1, 1)
            previous_alpha_bar = schedule.alpha_bars[previous_timestep].view(
                1, 1, 1, 1
            )
            predicted_clean = (
                x - torch.sqrt(1.0 - alpha_bar) * predicted_noise
            ) / torch.sqrt(alpha_bar)
            x = (
                torch.sqrt(previous_alpha_bar) * predicted_clean
                + torch.sqrt(1.0 - previous_alpha_bar) * predicted_noise
            )
            saved_step = step_index + 1
            if saved_step in save_position:
                value = value + trajectory_weight_sqrt * (
                    x * probes[save_position[saved_step]]
                ).sum()
        return value

    gradient_function = grad(trajectory_probe_scalar, argnums=0)
    features = np.empty(
        (
            len(UNROLLED_TRAJ_DAS_QUERY_IDS),
            UNROLLED_TRAJ_DAS_PROBES,
            UNROLLED_TRAJ_DAS_PROJECTION_DIM,
        ),
        dtype=np.float32,
    )
    for query_position, record in enumerate(records):
        trajectory = np.load(Path(record["dir"]) / "trajectory_xt.npy")
        x_initial = torch.from_numpy(trajectory[0]).to(
            device=device, dtype=torch.float32
        )
        condition = cond_for(record, dataset, device)
        for probe_index in range(UNROLLED_TRAJ_DAS_PROBES):
            generator = make_torch_generator(
                device,
                811,
                "unrolled_trajectory_probe",
                int(record["query_id"]),
                probe_index,
            )
            probes = (
                torch.randint(
                    0,
                    2,
                    (TRAJ_SNAPSHOTS, *x_initial.shape),
                    generator=generator,
                    device=device,
                    dtype=torch.int64,
                ).to(torch.float32)
                * 2.0
                - 1.0
            )
            gradients = gradient_function(named, x_initial, condition, probes)
            batched_gradients = {
                name: value.unsqueeze(0) for name, value in gradients.items()
            }
            projected = _project_batched_grads(
                batched_gradients,
                names,
                specs,
                UNROLLED_TRAJ_DAS_PROJECTION_DIM,
                False,
                1e-8,
            )[0]
            features[query_position, probe_index] = projected.detach().cpu().numpy()
            print(
                f"[unrolled-query] q{record['query_id']:02d} "
                f"probe={probe_index + 1}/{UNROLLED_TRAJ_DAS_PROBES} "
                f"norm={projected.norm().item():.6e}",
                flush=True,
            )
            del probes, gradients, batched_gradients, projected
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    UNROLLED_TRAJ_DAS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_numpy(feature_path, features)
    with open(info_path, "w") as handle:
        json.dump(
            {
                "method": UNROLLED_TRAJ_DAS_METHOD,
                "query_ids": list(UNROLLED_TRAJ_DAS_QUERY_IDS),
                "family": UNROLLED_TRAJ_DAS_FAMILY,
                "parameter_source": "final EMA",
                "checkpoint_epoch": int(checkpoint.get("epoch", EPOCHS)),
                "ddim_steps": int(DDIM_STEPS),
                "trajectory_snapshots": int(TRAJ_SNAPSHOTS),
                "trajectory_weight": 1.0 / float(TRAJ_SNAPSHOTS),
                "probe_count": int(UNROLLED_TRAJ_DAS_PROBES),
                "probe_distribution": "independent Rademacher per saved state/pixel",
                "projection_dim": int(UNROLLED_TRAJ_DAS_PROJECTION_DIM),
                "projection_seed": list(UNROLLED_TRAJ_DAS_PROJECTION_SEED),
                "normalize_query_features": False,
                "shape": list(features.shape),
            },
            handle,
            indent=2,
        )
    print(f"[saved] {UNROLLED_TRAJ_DAS_CACHE_DIR}", flush=True)


if __name__ == "__main__":
    main()
