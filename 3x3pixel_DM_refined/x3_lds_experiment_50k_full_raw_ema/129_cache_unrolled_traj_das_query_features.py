"""Cache 27 projected RGB output-basis VJPs per unrolled trajectory state."""

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
from x3_endpoint_das_jax_logic_pytorch import build_countsketch_specs


def atomic_numpy(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def query_shard_root(shard_index, shard_count):
    return UNROLLED_TRAJ_DAS_CACHE_DIR / "_query_shards" / (
        f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"
    )


def merge_query_shards(shard_count):
    merged = np.empty(
        (
            len(UNROLLED_TRAJ_DAS_QUERY_IDS),
            TRAJ_SNAPSHOTS,
            UNROLLED_TRAJ_DAS_OUTPUT_DIM,
            UNROLLED_TRAJ_DAS_PROJECTION_DIM,
        ),
        dtype=np.float32,
    )
    query_position = {
        query_id: position
        for position, query_id in enumerate(UNROLLED_TRAJ_DAS_QUERY_IDS)
    }
    covered = []
    common_info = None
    for shard_index in range(shard_count):
        root = query_shard_root(shard_index, shard_count)
        with open(root / "info.json") as handle:
            info = json.load(handle)
        values = np.load(root / "query_features.npy")
        shard_query_ids = [int(value) for value in info["query_ids"]]
        expected_shape = (
            len(shard_query_ids),
            TRAJ_SNAPSHOTS,
            UNROLLED_TRAJ_DAS_OUTPUT_DIM,
            UNROLLED_TRAJ_DAS_PROJECTION_DIM,
        )
        if values.shape != expected_shape:
            raise ValueError(f"invalid query shard shape in {root}: {values.shape}")
        if info["method"] != UNROLLED_TRAJ_DAS_METHOD:
            raise ValueError(f"query shard method mismatch in {root}")
        if common_info is None:
            common_info = info
        else:
            for key in (
                "family",
                "parameter_source",
                "checkpoint_epoch",
                "ddim_steps",
                "trajectory_snapshots",
                "trajectory_timesteps",
                "query_output_dimension",
                "query_feature_basis",
                "projection_dim",
                "projection_seed",
                "normalize_query_features",
                "initial_state_feature",
            ):
                if info[key] != common_info[key]:
                    raise ValueError(f"query shard {key} mismatch in {root}")
        for local_position, query_id in enumerate(shard_query_ids):
            if query_id not in query_position:
                raise ValueError(f"unexpected q{query_id:02d} in {root}")
            merged[query_position[query_id]] = values[local_position]
            covered.append(query_id)
    if sorted(covered) != sorted(UNROLLED_TRAJ_DAS_QUERY_IDS):
        raise ValueError(f"query shards do not cover q00-q09 exactly: {covered}")
    output_info = dict(common_info)
    output_info.pop("query_shard_index", None)
    output_info["query_ids"] = list(UNROLLED_TRAJ_DAS_QUERY_IDS)
    output_info["query_shard_count"] = int(shard_count)
    output_info["shape"] = list(merged.shape)
    UNROLLED_TRAJ_DAS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_numpy(UNROLLED_TRAJ_DAS_CACHE_DIR / "query_features.npy", merged)
    with open(UNROLLED_TRAJ_DAS_CACHE_DIR / "info.json", "w") as handle:
        json.dump(output_info, handle, indent=2)
    print(
        f"[merged] {shard_count} query shards -> {UNROLLED_TRAJ_DAS_CACHE_DIR}",
        flush=True,
    )


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
    parser.add_argument("--query-shard-index", type=int, default=0)
    parser.add_argument("--query-shard-count", type=int, default=1)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.query_shard_count <= 0:
        raise ValueError("query shard count must be positive")
    if args.merge_shards:
        merge_query_shards(args.query_shard_count)
        return
    if not 0 <= args.query_shard_index < args.query_shard_count:
        raise ValueError("invalid query shard index")
    if args.query_shard_count == 1:
        output_root = UNROLLED_TRAJ_DAS_CACHE_DIR
    else:
        output_root = query_shard_root(
            args.query_shard_index, args.query_shard_count
        )
    feature_path = output_root / "query_features.npy"
    info_path = output_root / "info.json"
    if feature_path.is_file() and info_path.is_file() and not args.force:
        print(f"[skip] existing query-feature cache {output_root}")
        return
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    selected_query_ids = UNROLLED_TRAJ_DAS_QUERY_IDS[
        args.query_shard_index :: args.query_shard_count
    ]
    records = [by_id[query_id] for query_id in selected_query_ids]
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
            len(selected_query_ids),
            TRAJ_SNAPSHOTS,
            UNROLLED_TRAJ_DAS_OUTPUT_DIM,
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
        if states[-1].numel() != UNROLLED_TRAJ_DAS_OUTPUT_DIM:
            raise ValueError(
                f"query state output dimension={states[-1].numel()}, "
                f"expected={UNROLLED_TRAJ_DAS_OUTPUT_DIM}"
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
            for output_index in range(UNROLLED_TRAJ_DAS_OUTPUT_DIM):
                scalar = state.reshape(-1)[output_index]
                is_last = (
                    state_index == TRAJ_SNAPSHOTS - 1
                    and output_index == UNROLLED_TRAJ_DAS_OUTPUT_DIM - 1
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
                    query_position, state_index, output_index
                ] = projected.detach().cpu().numpy()
                print(
                    f"[unrolled-query] q{record['query_id']:02d} "
                    f"state={state_index + 1}/{TRAJ_SNAPSHOTS} "
                    f"t={trajectory_timesteps[state_index]} "
                    f"jacobian_row={output_index + 1}/"
                    f"{UNROLLED_TRAJ_DAS_OUTPUT_DIM} "
                    f"norm={projected.norm().item():.6e}",
                    flush=True,
                )
                del scalar, gradient_values, gradients
                del batched_gradients, projected
        del states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_root.mkdir(parents=True, exist_ok=True)
    atomic_numpy(feature_path, features)
    with open(info_path, "w") as handle:
        json.dump(
            {
                "method": UNROLLED_TRAJ_DAS_METHOD,
                "query_ids": list(selected_query_ids),
                "query_shard_index": int(args.query_shard_index),
                "query_shard_count": int(args.query_shard_count),
                "family": UNROLLED_TRAJ_DAS_FAMILY,
                "parameter_source": "final EMA",
                "checkpoint_epoch": int(checkpoint.get("epoch", EPOCHS)),
                "ddim_steps": int(DDIM_STEPS),
                "trajectory_snapshots": int(TRAJ_SNAPSHOTS),
                "trajectory_timesteps": common_timesteps,
                "query_output_dimension": int(UNROLLED_TRAJ_DAS_OUTPUT_DIM),
                "query_feature_basis": (
                    "27 projected RGB output-basis gradients; each loss probe "
                    "is combined linearly to equal one direct scalar VJP"
                ),
                "projection_dim": int(UNROLLED_TRAJ_DAS_PROJECTION_DIM),
                "projection_seed": list(UNROLLED_TRAJ_DAS_PROJECTION_SEED),
                "normalize_query_features": False,
                "initial_state_feature": "exact zero because initial noise is fixed",
                "shape": list(features.shape),
            },
            handle,
            indent=2,
        )
    print(f"[saved] {output_root}", flush=True)


if __name__ == "__main__":
    main()
