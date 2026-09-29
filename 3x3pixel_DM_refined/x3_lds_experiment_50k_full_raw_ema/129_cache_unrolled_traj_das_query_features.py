"""Cache one fully-unrolled projected Jacobian feature per trajectory state."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

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


def differentiable_ddim_trajectory(model, schedule, condition, initial_state):
    ddim_timesteps = torch.linspace(
        T - 1, 0, DDIM_STEPS, device=initial_state.device
    ).long()
    save_steps = np.linspace(
        0, DDIM_STEPS - 1, TRAJ_SNAPSHOTS, dtype=np.int64
    ).tolist()
    save_position = {int(step): position for position, step in enumerate(save_steps)}
    saved = [None] * TRAJ_SNAPSHOTS
    x = initial_state
    if 0 in save_position:
        saved[save_position[0]] = x
    for step_index in range(len(ddim_timesteps) - 1):
        timestep = ddim_timesteps[step_index].repeat(x.shape[0])
        previous_timestep = int(ddim_timesteps[step_index + 1].item())
        predicted_noise = model(x, timestep, condition)
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
            saved[save_position[saved_step]] = x
    if any(value is None for value in saved):
        raise RuntimeError("differentiable DDIM did not produce every saved state")
    trajectory_timesteps = [
        int(ddim_timesteps[step].item()) for step in save_steps
    ]
    return saved, trajectory_timesteps


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
    features = np.zeros(
        (
            len(UNROLLED_TRAJ_DAS_QUERY_IDS),
            TRAJ_SNAPSHOTS,
            UNROLLED_TRAJ_DAS_PROBES,
            UNROLLED_TRAJ_DAS_PROJECTION_DIM,
        ),
        dtype=np.float32,
    )
    common_timesteps = None
    for query_position, record in enumerate(records):
        reference_trajectory = np.load(Path(record["dir"]) / "trajectory_xt.npy")
        initial_state = torch.from_numpy(reference_trajectory[0]).to(
            device=device, dtype=torch.float32
        )
        condition = cond_for(record, dataset, device)
        states, trajectory_timesteps = differentiable_ddim_trajectory(
            model, schedule, condition, initial_state
        )
        if common_timesteps is None:
            common_timesteps = trajectory_timesteps
        elif trajectory_timesteps != common_timesteps:
            raise ValueError("query trajectory timestep banks differ")
        with torch.no_grad():
            endpoint_error = (
                states[-1] - torch.from_numpy(reference_trajectory[-1]).to(device)
            ).abs().max().item()
        if endpoint_error > 1e-5:
            raise ValueError(
                f"q{record['query_id']:02d} differentiable DDIM endpoint mismatch "
                f"{endpoint_error:.6e}"
            )

        for state_index, state in enumerate(states):
            if not state.requires_grad:
                print(
                    f"[unrolled-query] q{record['query_id']:02d} "
                    f"state={state_index + 1}/{TRAJ_SNAPSHOTS} "
                    f"t={trajectory_timesteps[state_index]} fixed-initial-state",
                    flush=True,
                )
                continue
            for probe_index in range(UNROLLED_TRAJ_DAS_PROBES):
                generator = make_torch_generator(
                    device,
                    811,
                    "unrolled_trajectory_state_probe",
                    int(record["query_id"]),
                    state_index,
                    probe_index,
                )
                probe = (
                    torch.randint(
                        0,
                        2,
                        state.shape,
                        generator=generator,
                        device=device,
                        dtype=torch.int64,
                    ).to(torch.float32)
                    * 2.0
                    - 1.0
                )
                scalar = (state * probe).sum()
                is_last = (
                    state_index == TRAJ_SNAPSHOTS - 1
                    and probe_index == UNROLLED_TRAJ_DAS_PROBES - 1
                )
                gradient_values = torch.autograd.grad(
                    scalar, active, retain_graph=not is_last
                )
                gradients = dict(zip(names, gradient_values))
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
                features[
                    query_position, state_index, probe_index
                ] = projected.detach().cpu().numpy()
                print(
                    f"[unrolled-query] q{record['query_id']:02d} "
                    f"state={state_index + 1}/{TRAJ_SNAPSHOTS} "
                    f"t={trajectory_timesteps[state_index]} "
                    f"probe={probe_index + 1}/{UNROLLED_TRAJ_DAS_PROBES} "
                    f"norm={projected.norm().item():.6e}",
                    flush=True,
                )
                del probe, scalar, gradient_values, gradients
                del batched_gradients, projected
        del states
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
                "trajectory_timesteps": common_timesteps,
                "probe_count_per_state": int(UNROLLED_TRAJ_DAS_PROBES),
                "probe_distribution": "independent Rademacher per state/pixel",
                "projection_dim": int(UNROLLED_TRAJ_DAS_PROJECTION_DIM),
                "projection_seed": list(UNROLLED_TRAJ_DAS_PROJECTION_SEED),
                "normalize_query_features": False,
                "initial_state_feature": "exact zero because initial noise is fixed",
                "shape": list(features.shape),
            },
            handle,
            indent=2,
        )
    print(f"[saved] {UNROLLED_TRAJ_DAS_CACHE_DIR}", flush=True)


if __name__ == "__main__":
    main()
