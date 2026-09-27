"""Exact TracIn alignment to next-checkpoint predicted-noise delta directions."""

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
from run_exact_traj_next_bank import flatten_batched_gradients, flatten_gradient_tuple
from tracin_das_config import *
from x3_endpoint_das_jax_logic_pytorch import (
    build_countsketch_specs,
    make_torch_generator,
)


CONTRACT_VERSION = 2


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--timestamp-shard-index", type=int, required=True)
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=TRACIN_DAS_BATCH_SIZE)
    parser.add_argument("--noise-mode", choices=TRACIN_DAS_NOISE_MODES, default="checkpoint")
    parser.add_argument(
        "--parameter-projection",
        choices=TRACIN_DAS_PARAMETER_PROJECTIONS,
        default="exact",
    )
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
    by_id = {int(record["query_id"]): record for record in manifest}
    records = [by_id[qid] for qid in TRACIN_DAS_QUERY_IDS]
    if any(record["family"] != TRACIN_DAS_FAMILY for record in records):
        raise ValueError("q00-q09 must be prompted queries")
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    paths = model_paths(TRACIN_DAS_FAMILY)
    if len(paths) != 50:
        raise ValueError(f"expected 50 old-experiment checkpoints, found {len(paths)}")
    schedule = base.make_linear_schedule(T, device=device)
    x_all, cond_all = preload_dataset(dataset, TRACIN_DAS_FAMILY, device)
    endpoints = [
        torch.from_numpy(np.load(Path(record["dir"]) / "final_state.npy")).to(
            device=device, dtype=torch.float32
        )
        for record in records
    ]
    conditions = [cond_for(record, dataset, device) for record in records]
    timestamps = tuple(int(value) for value in DAS_TIMESTEPS)
    selected = list(range(args.timestamp_shard_index, len(timestamps), args.timestamp_shard_count))
    methods = tracin_das_methods(args.noise_mode, args.parameter_projection)
    shard_root = tracin_das_shard_root(
        args.timestamp_shard_index,
        args.timestamp_shard_count,
        args.noise_mode,
        args.parameter_projection,
    )
    done_path = shard_root / "done.json"
    if done_path.is_file():
        print(f"[skip] {done_path}", flush=True)
        return

    partial_paths = {
        contraction: shard_root / f"partial_{contraction}.npy"
        for contraction in methods
    }
    progress_path = shard_root / "progress.json"
    completed_timestamps = []
    expected_shape = (len(records), N_TRAIN)
    if all(path.is_file() for path in partial_paths.values()) and progress_path.is_file():
        with open(progress_path) as handle:
            progress = json.load(handle)
        if int(progress["contract_version"]) != CONTRACT_VERSION:
            raise ValueError("partial score contract changed")
        if int(progress["batch_size"]) != args.batch_size:
            raise ValueError("partial shard batch size differs; move shard aside before restarting")
        if progress.get("noise_mode", "checkpoint") != args.noise_mode:
            raise ValueError("partial shard noise mode differs")
        if progress.get("parameter_projection", "exact") != args.parameter_projection:
            raise ValueError("partial shard parameter projection differs")
        completed_timestamps = [int(value) for value in progress["completed_timestamps"]]
        scores = {
            contraction: torch.from_numpy(np.load(path)).to(device=device, dtype=torch.float64)
            for contraction, path in partial_paths.items()
        }
        if any(value.shape != expected_shape for value in scores.values()):
            raise ValueError("partial score shape mismatch")
        print(f"[resume] timestamps={len(completed_timestamps)}/{len(selected)}", flush=True)
    else:
        scores = {
            contraction: torch.zeros(expected_shape, device=device, dtype=torch.float64)
            for contraction in methods
        }

    transitions = [(index, index + 1) for index in range(len(paths) - 1)]
    remaining = [index for index in selected if index not in set(completed_timestamps)]
    snapshot_weight = 1.0 / len(timestamps)
    total_terms = len(remaining) * len(transitions)
    completed_terms = 0
    started = time.perf_counter()
    print(
        f"[tracin-das gpu={args.gpu}] q00-q09 "
        f"parameter_projection={args.parameter_projection} "
        f"timestamps={len(selected)}/100 transitions=49 batch={args.batch_size} "
        f"noise_mode={args.noise_mode} "
        "term_internal_alignment=true output_delta_normalized=true",
        flush=True,
    )

    for shard_timestamp_position, timestamp_index in enumerate(remaining, start=1):
        timestep = timestamps[timestamp_index]
        t_query = torch.tensor([timestep], device=device, dtype=torch.long)
        timestamp_accumulator = torch.zeros(expected_shape, device=device, dtype=torch.float64)

        for transition_position, (checkpoint_index, target_index) in enumerate(transitions, start=1):
            term_started = time.perf_counter()
            noise_seed_parts = (
                (
                    TRACIN_DAS_NOISE_SEED,
                    "tracin_das_checkpoint_timestamp_shared_noise",
                    checkpoint_index,
                    timestamp_index,
                    timestep,
                )
                if args.noise_mode == "checkpoint"
                else (
                    TRACIN_DAS_NOISE_SEED,
                    "tracin_das_timestamp_shared_noise",
                    timestamp_index,
                    timestep,
                )
            )
            noise_generator = make_torch_generator(device, *noise_seed_parts)
            shared_noise = torch.randn(
                endpoints[0].shape,
                generator=noise_generator,
                device=device,
                dtype=endpoints[0].dtype,
            )
            query_xt = [
                base.q_sample(endpoint, t_query, shared_noise, schedule)
                for endpoint in endpoints
            ]
            model, _, checkpoint = build_model(paths[checkpoint_index], "raw", device)
            target, _, _ = build_model(paths[target_index], "raw", device)
            named = dict(model.named_parameters())
            names = tuple(named)
            parameters = tuple(named.values())
            projection_specs = (
                build_countsketch_specs(
                    list(parameters),
                    TRACIN_PROJ_DIM,
                    device=device,
                    seed_parts=(
                        TRAIN_SEED,
                        "tracin_das_parameter_projection",
                        checkpoint_index,
                    ),
                )
                if args.parameter_projection == "projected4096"
                else None
            )
            query_vectors = []
            delta_norms = []
            for xt, condition in zip(query_xt, conditions):
                current_prediction = model(xt, t_query, condition)
                with torch.no_grad():
                    next_prediction = target(xt, t_query, condition)
                    direction = next_prediction - current_prediction.detach()
                    delta_norm = direction.norm()
                    direction = direction / delta_norm.clamp_min(TRACIN_DAS_DIRECTION_EPS)
                projected_prediction = (current_prediction * direction).sum()
                query_gradient = torch.autograd.grad(projected_prediction, parameters)
                query_vectors.append(
                    _project_gradient_tuple(
                        query_gradient,
                        projection_specs,
                        TRACIN_PROJ_DIM,
                    )
                    if projection_specs is not None
                    else flatten_gradient_tuple(query_gradient)
                )
                delta_norms.append(float(delta_norm))
            query_matrix = torch.stack(query_vectors).detach().to(torch.float32)

            def aligned_loss(parameter_dict, x0, condition):
                xt = base.q_sample(x0.unsqueeze(0), t_query, shared_noise, schedule)
                prediction = functional_call(
                    model, parameter_dict, (xt, t_query, condition.unsqueeze(0))
                )
                return (prediction - shared_noise).square().mean()

            batched_gradient = vmap(grad(aligned_loss), in_dims=(None, 0, 0))
            weight = float(tracin_lr_weight(checkpoint)) * snapshot_weight
            num_batches = math.ceil(N_TRAIN / args.batch_size)
            progress_every = max(1, num_batches // 5)
            for batch_position, start in enumerate(range(0, N_TRAIN, args.batch_size), start=1):
                end = min(start + args.batch_size, N_TRAIN)
                gradients = batched_gradient(named, x_all[start:end], cond_all[start:end])
                train_matrix = (
                    _project_batched_grads(
                        gradients,
                        names,
                        projection_specs,
                        TRACIN_PROJ_DIM,
                        False,
                        1e-8,
                    )
                    if projection_specs is not None
                    else flatten_batched_gradients(gradients, names)
                ).detach()
                dots = (train_matrix @ query_matrix.T).T.to(torch.float64)
                scores["linear"][:, start:end] += weight * dots
                scores["termwise_squared"][:, start:end] += weight * dots.square()
                timestamp_accumulator[:, start:end] += weight * dots
                if batch_position == 1 or batch_position % progress_every == 0 or batch_position == num_batches:
                    print(
                        f"[tracin-das gpu={args.gpu}] timestamp={shard_timestamp_position}/{len(remaining)} "
                        f"global_t={timestamp_index+1}/100 pair={transition_position}/49 "
                        f"batch={batch_position}/{num_batches}",
                        flush=True,
                    )
            completed_terms += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / completed_terms * (total_terms - completed_terms)
            print(
                f"[tracin-das gpu={args.gpu}] term={completed_terms}/{total_terms} "
                f"delta_norm=[{min(delta_norms):.3e},{max(delta_norms):.3e}] "
                f"term_elapsed={(time.perf_counter()-term_started)/60:.1f}m "
                f"eta={eta/3600:.2f}h",
                flush=True,
            )
            del model, target, named, parameters, query_vectors, query_matrix
            del projection_specs
            del current_prediction, next_prediction, direction, projected_prediction, query_gradient
            del batched_gradient, gradients, train_matrix, dots, shared_noise, query_xt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        scores["timestamp_sum_squared"] += timestamp_accumulator.square()
        completed_timestamps.append(timestamp_index)
        completed_timestamps.sort()
        for contraction, path in partial_paths.items():
            atomic_numpy(path, scores[contraction].cpu().numpy())
        atomic_json(
            progress_path,
            {
                "contract_version": CONTRACT_VERSION,
                "batch_size": args.batch_size,
                "noise_mode": args.noise_mode,
                "parameter_projection": args.parameter_projection,
                "query_ids": list(TRACIN_DAS_QUERY_IDS),
                "completed_timestamps": completed_timestamps,
            },
        )
        print(f"[checkpoint] timestamps={len(completed_timestamps)}/{len(selected)}", flush=True)

    for contraction, value in scores.items():
        atomic_numpy(shard_root / f"{contraction}.npy", value.cpu().numpy())
    atomic_json(
        done_path,
        {
            "methods": methods,
            "query_ids": list(TRACIN_DAS_QUERY_IDS),
            "family": TRACIN_DAS_FAMILY,
            "timestamp_indices": selected,
            "timestamps": [timestamps[index] for index in selected],
            "checkpoint_transitions": 49,
            "parameter_source": "raw",
            "endpoint_source": "cached final-EMA query endpoint",
            "noise_mode": args.noise_mode,
            "parameter_projection": args.parameter_projection,
            "parameter_projection_dim": (
                TRACIN_PROJ_DIM
                if args.parameter_projection == "projected4096"
                else None
            ),
            "endpoint_noising": (
                "one independent noise per checkpoint/timestamp"
                if args.noise_mode == "checkpoint"
                else "one noise per timestamp shared across all checkpoint transitions"
            ),
            "train_loss_noise": "same term noise as query endpoint",
            "query_scalar": "dot(epsilon_current, normalize(epsilon_next-epsilon_current))",
            "lr_weighted": TRACIN_USE_LR_WEIGHTS,
            "timestamp_weight": snapshot_weight,
            "batch_size": args.batch_size,
            "contract_version": CONTRACT_VERSION,
        },
    )
    print(f"[done] {shard_root}", flush=True)


if __name__ == "__main__":
    main()
