"""One timestamp shard of reference-state forward-10 aligned-loss TracIn."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

import x3pixel_DM_training as base
from attribution_one_query import (
    _project_batched_grads,
    _project_gradient_tuple,
    build_model,
    cond_for,
    model_paths,
    preload_dataset,
    tracin_lr_weight,
)
from dataset_loader import ColorGridDataset
from reference_forward10_config import *
from x3_endpoint_das_jax_logic_pytorch import (
    build_countsketch_specs,
    make_torch_generator,
)


CONTRACT_VERSION = 1


def atomic_numpy(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def extended_schedule(device):
    """Preserve the original 0..999 schedule and extend its beta line by 10."""
    original = base.make_linear_schedule(T, device=device)
    extra_indices = torch.arange(
        T, T + REF_FORWARD10_DELTA_T, device=device, dtype=torch.float32
    )
    extra_betas = 1e-4 + (0.02 - 1e-4) * extra_indices / float(T - 1)
    betas = torch.cat((original.betas, extra_betas))
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return base.DiffusionSchedule(
        T=len(betas),
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=torch.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - alpha_bars),
    )


def forward_from_reference(state, current_t, target_t, noise, schedule):
    ratio = schedule.alpha_bars[target_t] / schedule.alpha_bars[current_t]
    return torch.sqrt(ratio) * state + torch.sqrt(1.0 - ratio) * noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--timestamp-shard-index", type=int, required=True)
    parser.add_argument("--timestamp-shard-count", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=REF_FORWARD10_BATCH_SIZE)
    args = parser.parse_args()
    if not 0 <= args.timestamp_shard_index < args.timestamp_shard_count:
        raise ValueError("invalid timestamp shard")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    records = [record for record in manifest if record["family"] == args.family]
    query_ids = [int(record["query_id"]) for record in records]
    expected_ids = list(range(75)) if args.family == "prompted" else list(range(75, 100))
    if query_ids != expected_ids:
        raise ValueError(f"unexpected {args.family} query IDs: {query_ids}")
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    checkpoint_paths = model_paths(args.family)
    if len(checkpoint_paths) != 50:
        raise ValueError(f"expected 50 checkpoints, found {len(checkpoint_paths)}")
    trajectories = [
        np.load(Path(record["dir"]) / "trajectory_xt.npy") for record in records
    ]
    timestep_arrays = [
        np.load(Path(record["dir"]) / "trajectory_t.npy") for record in records
    ]
    t_seq = timestep_arrays[0]
    if len(t_seq) != 100 or any(not np.array_equal(t_seq, ts) for ts in timestep_arrays):
        raise ValueError("reference trajectory timestamp banks differ")
    if int(t_seq[0]) != T - 1 or int(t_seq[-1]) != 0:
        raise ValueError(f"unexpected reference timestamps: {t_seq[0]}..{t_seq[-1]}")
    conditions = [cond_for(record, dataset, device) for record in records]
    x_all, cond_all = preload_dataset(dataset, args.family, device)
    schedule = extended_schedule(device)
    selected = list(
        range(args.timestamp_shard_index, len(t_seq), args.timestamp_shard_count)
    )
    root = ref_forward10_shard_root(
        args.family, args.timestamp_shard_index, args.timestamp_shard_count
    )
    done_path = root / "done.json"
    if done_path.is_file():
        print(f"[skip] {done_path}", flush=True)
        return
    partial_paths = {
        contraction: root / f"partial_{contraction}.npy"
        for contraction in REF_FORWARD10_METHODS
    }
    progress_path = root / "progress.json"
    shape = (len(records), N_TRAIN)
    completed = []
    if all(path.is_file() for path in partial_paths.values()) and progress_path.is_file():
        with open(progress_path) as handle:
            progress = json.load(handle)
        if int(progress["contract_version"]) != CONTRACT_VERSION:
            raise ValueError("partial contract differs")
        if int(progress["batch_size"]) != args.batch_size:
            raise ValueError("partial batch size differs")
        completed = [int(value) for value in progress["completed_timestamps"]]
        scores = {
            name: torch.from_numpy(np.load(path)).to(device=device, dtype=torch.float64)
            for name, path in partial_paths.items()
        }
        print(f"[resume] timestamps={len(completed)}/{len(selected)}", flush=True)
    else:
        scores = {
            name: torch.zeros(shape, device=device, dtype=torch.float64)
            for name in REF_FORWARD10_METHODS
        }

    remaining = [index for index in selected if index not in set(completed)]
    timestamp_weight = 1.0 / len(t_seq)
    total_terms = len(remaining) * len(checkpoint_paths)
    finished_terms = 0
    started = time.perf_counter()
    print(
        f"[ref-forward10 gpu={args.gpu}] family={args.family} "
        f"queries={query_ids[0]}..{query_ids[-1]} timestamps={len(selected)}/100 "
        f"checkpoints=50 delta_t=10 aligned_noise=true projection=4096 "
        f"batch={args.batch_size}",
        flush=True,
    )

    for timestamp_position, timestamp_index in enumerate(remaining, start=1):
        current_t = int(t_seq[timestamp_index])
        target_t = current_t + REF_FORWARD10_DELTA_T
        t_target = torch.tensor([target_t], device=device, dtype=torch.long)
        noise_generator = make_torch_generator(
            device,
            REF_FORWARD10_NOISE_SEED,
            "reference_forward10_aligned_noise",
            timestamp_index,
            current_t,
            target_t,
        )
        aligned_noise = torch.randn(
            trajectories[0][timestamp_index].shape,
            generator=noise_generator,
            device=device,
            dtype=torch.float32,
        )
        output_direction = aligned_noise / aligned_noise.norm().clamp_min(1e-12)
        query_states = [
            forward_from_reference(
                torch.from_numpy(trajectory[timestamp_index]).to(
                    device=device, dtype=torch.float32
                ),
                current_t,
                target_t,
                aligned_noise,
                schedule,
            )
            for trajectory in trajectories
        ]
        timestamp_accumulator = torch.zeros(shape, device=device, dtype=torch.float64)

        for checkpoint_position, checkpoint_path in enumerate(checkpoint_paths, start=1):
            term_started = time.perf_counter()
            model, _, checkpoint = build_model(checkpoint_path, "raw", device)
            named = dict(model.named_parameters())
            names = tuple(named)
            parameters = tuple(named.values())
            specs = build_countsketch_specs(
                list(parameters),
                TRACIN_PROJ_DIM,
                device=device,
                seed_parts=(
                    TRAIN_SEED,
                    "reference_forward10_parameter_projection",
                    checkpoint_position - 1,
                ),
            )
            query_vectors = []
            for state, condition in zip(query_states, conditions):
                prediction = model(state, t_target, condition)
                projected_prediction = (prediction * output_direction).sum()
                query_gradient = torch.autograd.grad(
                    projected_prediction, parameters
                )
                query_vectors.append(
                    _project_gradient_tuple(query_gradient, specs, TRACIN_PROJ_DIM)
                )
            query_matrix = torch.stack(query_vectors).detach().to(torch.float32)

            def train_loss(parameter_dict, x0, condition):
                xt = base.q_sample(
                    x0.unsqueeze(0), t_target, aligned_noise, schedule
                )
                prediction = functional_call(
                    model,
                    parameter_dict,
                    (xt, t_target, condition.unsqueeze(0)),
                )
                return (prediction - aligned_noise).square().mean()

            batched_gradient = vmap(grad(train_loss), in_dims=(None, 0, 0))
            weight = float(tracin_lr_weight(checkpoint)) * timestamp_weight
            num_batches = math.ceil(N_TRAIN / args.batch_size)
            progress_every = max(1, num_batches // 5)
            for batch_position, start in enumerate(
                range(0, N_TRAIN, args.batch_size), start=1
            ):
                end = min(start + args.batch_size, N_TRAIN)
                gradients = batched_gradient(
                    named, x_all[start:end], cond_all[start:end]
                )
                train_matrix = _project_batched_grads(
                    gradients,
                    names,
                    specs,
                    TRACIN_PROJ_DIM,
                    False,
                    1e-8,
                ).detach()
                dots = (train_matrix @ query_matrix.T).T.to(torch.float64)
                scores["linear"][:, start:end] += weight * dots
                scores["termwise_squared"][:, start:end] += weight * dots.square()
                timestamp_accumulator[:, start:end] += weight * dots
                if (
                    batch_position == 1
                    or batch_position % progress_every == 0
                    or batch_position == num_batches
                ):
                    print(
                        f"[ref-forward10 gpu={args.gpu}] "
                        f"timestamp={timestamp_position}/{len(remaining)} "
                        f"global_t={timestamp_index + 1}/100 current={current_t} "
                        f"target={target_t} checkpoint={checkpoint_position}/50 "
                        f"batch={batch_position}/{num_batches}",
                        flush=True,
                    )
            finished_terms += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / finished_terms * (total_terms - finished_terms)
            print(
                f"[ref-forward10 gpu={args.gpu}] term={finished_terms}/{total_terms} "
                f"term_elapsed={(time.perf_counter() - term_started) / 60:.1f}m "
                f"eta={eta / 3600:.2f}h",
                flush=True,
            )
            del model, checkpoint, named, parameters, specs
            del query_vectors, query_matrix, prediction, projected_prediction
            del query_gradient
            del batched_gradient, gradients, train_matrix, dots, train_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        scores["timestamp_sum_squared"] += timestamp_accumulator.square()
        completed.append(timestamp_index)
        completed.sort()
        for name, path in partial_paths.items():
            atomic_numpy(path, scores[name].cpu().numpy())
        atomic_json(
            progress_path,
            {
                "contract_version": CONTRACT_VERSION,
                "family": args.family,
                "query_ids": query_ids,
                "batch_size": args.batch_size,
                "completed_timestamps": completed,
            },
        )

    for name, values in scores.items():
        atomic_numpy(root / f"{name}.npy", values.cpu().numpy())
    atomic_json(
        done_path,
        {
            "methods": REF_FORWARD10_METHODS,
            "family": args.family,
            "query_ids": query_ids,
            "timestamp_indices": selected,
            "reference_timesteps": [int(t_seq[index]) for index in selected],
            "loss_timesteps": [
                int(t_seq[index]) + REF_FORWARD10_DELTA_T for index in selected
            ],
            "delta_t": REF_FORWARD10_DELTA_T,
            "checkpoint_count": len(checkpoint_paths),
            "parameter_source": "raw",
            "query_state_source": "cached final-EMA reference trajectory",
            "query_forward_noise": "conditional q(x_{t+10}|x_t)",
            "query_scalar": (
                "dot(predicted_noise_at_t_plus_10, unit(aligned_noise))"
            ),
            "query_uses_loss": False,
            "train_uses_diffusion_loss": True,
            "query_train_noise_aligned": True,
            "projection": "countsketch",
            "projection_dim": TRACIN_PROJ_DIM,
            "batch_size": args.batch_size,
            "contract_version": CONTRACT_VERSION,
        },
    )
    print(f"[done] {root}", flush=True)


if __name__ == "__main__":
    main()
