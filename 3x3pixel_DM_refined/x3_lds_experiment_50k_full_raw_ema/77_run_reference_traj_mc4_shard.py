"""Projected TracIn-DAS around four perturbations of each reference trajectory state."""

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
from reference_traj_mc4_config import *
from x3_endpoint_das_jax_logic_pytorch import build_countsketch_specs, make_torch_generator


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


def perturbation_bank(query_id, timestamp_index, shape, device, query_mc):
    generator = make_torch_generator(
        device,
        REF_MC4_DIRECTION_SEED,
        "reference_trajectory_state_perturbation",
        int(query_id),
        int(timestamp_index),
    )
    directions = torch.randn(
        (query_mc, *shape), generator=generator,
        device=device, dtype=torch.float32,
    )
    norms = directions.reshape(query_mc, -1).norm(dim=1)
    return directions / norms.clamp_min(REF_MC4_DIRECTION_EPS).view(
        query_mc, *([1] * len(shape))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--timestamp-shard-index", type=int, required=True)
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=REF_MC4_BATCH_SIZE)
    parser.add_argument("--epsilon", type=float, default=REF_MC4_DEFAULT_EPSILON)
    parser.add_argument("--train-mc", type=int, default=REF_MC4_DEFAULT_TRAIN_MC)
    parser.add_argument("--query-mc", type=int, default=REF_MC4_COUNT)
    args = parser.parse_args()
    if args.epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if args.train_mc <= 0:
        raise ValueError("train MC must be positive")
    if args.query_mc <= 0:
        raise ValueError("query MC must be positive")
    if not 0 <= args.timestamp_shard_index < args.timestamp_shard_count:
        raise ValueError("invalid timestamp shard")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    records = [by_id[qid] for qid in REF_MC4_QUERY_IDS]
    if any(record["family"] != REF_MC4_FAMILY for record in records):
        raise ValueError("q00-q09 must be prompted")
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    paths = model_paths(REF_MC4_FAMILY)
    if len(paths) != 50:
        raise ValueError(f"expected 50 checkpoints, found {len(paths)}")
    trajectories = [
        np.load(Path(record["dir"]) / "trajectory_xt.npy") for record in records
    ]
    timestep_arrays = [
        np.load(Path(record["dir"]) / "trajectory_t.npy") for record in records
    ]
    t_seq = timestep_arrays[0]
    if len(t_seq) != 100 or any(not np.array_equal(t_seq, ts) for ts in timestep_arrays):
        raise ValueError("q00-q09 reference trajectory timestamps differ")
    conditions = [cond_for(record, dataset, device) for record in records]
    schedule = base.make_linear_schedule(T, device=device)
    x_all, cond_all = preload_dataset(dataset, REF_MC4_FAMILY, device)
    selected = list(range(args.timestamp_shard_index, len(t_seq), args.timestamp_shard_count))
    root = ref_mc4_shard_root(
        args.timestamp_shard_index,
        args.timestamp_shard_count,
        args.epsilon,
        args.train_mc,
        args.query_mc,
    )
    done_path = root / "done.json"
    if done_path.is_file():
        print(f"[skip] {done_path}", flush=True)
        return
    methods = ref_mc4_methods(args.epsilon, args.train_mc, args.query_mc)
    partial_paths = {name: root / f"partial_{name}.npy" for name in methods}
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
        if int(progress.get("train_mc", -1)) != args.train_mc:
            raise ValueError("partial train MC differs")
        if int(progress.get("query_mc", REF_MC4_COUNT)) != args.query_mc:
            raise ValueError("partial query MC differs")
        completed = [int(value) for value in progress["completed_timestamps"]]
        scores = {
            name: torch.from_numpy(np.load(path)).to(device=device, dtype=torch.float64)
            for name, path in partial_paths.items()
        }
    else:
        scores = {name: torch.zeros(shape, device=device, dtype=torch.float64) for name in methods}

    transitions = [(index, index + 1) for index in range(49)]
    remaining = [index for index in selected if index not in set(completed)]
    timestamp_weight = 1.0 / len(t_seq)
    mc_weight = 1.0 / args.query_mc
    train_mc = int(args.train_mc)
    started = time.perf_counter()
    total_terms = len(remaining) * len(transitions)
    finished_terms = 0
    print(
        f"[ref-mc4 gpu={args.gpu}] q00-q09 epsilon={args.epsilon:g} "
        f"timestamps={len(selected)}/100 transitions=49 query_mc={args.query_mc} "
        f"train_mc={train_mc} projection={TRACIN_PROJ_DIM} batch={args.batch_size}",
        flush=True,
    )

    for timestamp_position, timestamp_index in enumerate(remaining, start=1):
        timestep = int(t_seq[timestamp_index])
        t_query = torch.tensor([timestep], device=device, dtype=torch.long)
        perturbed_states = []
        for record, trajectory in zip(records, trajectories):
            reference_state = torch.from_numpy(trajectory[timestamp_index]).to(
                device=device, dtype=torch.float32
            )
            directions = perturbation_bank(
                int(record["query_id"]), timestamp_index,
                tuple(reference_state.shape), device, args.query_mc,
            )
            perturbed_states.append(
                reference_state.unsqueeze(0) + args.epsilon * directions
            )
        timestamp_accumulator = torch.zeros(shape, device=device, dtype=torch.float64)

        for transition_position, (checkpoint_index, target_index) in enumerate(transitions, start=1):
            model, _, checkpoint = build_model(paths[checkpoint_index], "raw", device)
            target, _, _ = build_model(paths[target_index], "raw", device)
            named = dict(model.named_parameters())
            names = tuple(named)
            parameters = tuple(named.values())
            specs = build_countsketch_specs(
                list(parameters), TRACIN_PROJ_DIM, device=device,
                seed_parts=(TRAIN_SEED, "reference_traj_mc4_projection", checkpoint_index),
            )
            query_features = []
            delta_norms = []
            for states, condition in zip(perturbed_states, conditions):
                per_query = []
                for mc_index in range(args.query_mc):
                    state = states[mc_index]
                    current = model(state, t_query, condition)
                    with torch.no_grad():
                        next_value = target(state, t_query, condition)
                        delta = next_value - current.detach()
                        delta_norm = delta.norm()
                        delta = delta / delta_norm.clamp_min(REF_MC4_DIRECTION_EPS)
                    scalar = (current * delta).sum()
                    query_gradient = torch.autograd.grad(scalar, parameters)
                    per_query.append(
                        _project_gradient_tuple(
                            query_gradient, specs, TRACIN_PROJ_DIM
                        )
                    )
                    delta_norms.append(float(delta_norm))
                query_features.append(torch.stack(per_query))
            query_matrix = torch.stack(query_features).reshape(
                len(records) * args.query_mc, TRACIN_PROJ_DIM
            ).detach()
            train_timesteps = torch.full(
                (train_mc,), timestep, device=device, dtype=torch.long
            )

            def train_loss(parameter_dict, x0, condition, noises):
                x_mc = x0.unsqueeze(0).expand(train_mc, *x0.shape)
                condition_mc = condition.unsqueeze(0).expand(train_mc, condition.shape[-1])
                xt = base.q_sample(x_mc, train_timesteps, noises, schedule)
                prediction = functional_call(
                    model, parameter_dict, (xt, train_timesteps, condition_mc)
                )
                return (prediction - noises).square().reshape(train_mc, -1).mean(dim=1).mean()

            batched_gradient = vmap(grad(train_loss), in_dims=(None, 0, 0, 0))
            weight = float(tracin_lr_weight(checkpoint)) * timestamp_weight * mc_weight
            num_batches = math.ceil(N_TRAIN / args.batch_size)
            for batch_position, start in enumerate(range(0, N_TRAIN, args.batch_size), start=1):
                end = min(start + args.batch_size, N_TRAIN)
                noise_generator = make_torch_generator(
                    device, TRAIN_SEED, "reference_traj_mc4_random_train_noise",
                    checkpoint_index, timestamp_index, start, train_mc,
                )
                noises = torch.randn(
                    (end-start, train_mc, *x_all.shape[1:]),
                    generator=noise_generator, device=device, dtype=x_all.dtype,
                )
                gradients = batched_gradient(
                    named, x_all[start:end], cond_all[start:end], noises
                )
                train_features = _project_batched_grads(
                    gradients, names, specs, TRACIN_PROJ_DIM, False, 1e-8
                )
                dots = (train_features @ query_matrix.T).T.reshape(
                    len(records), args.query_mc, end-start
                ).to(torch.float64)
                scores["linear"][:, start:end] += weight * dots.sum(dim=1)
                scores["termwise_squared"][:, start:end] += weight * dots.square().sum(dim=1)
                timestamp_accumulator[:, start:end] += weight * dots.sum(dim=1)
                if batch_position == 1 or batch_position == num_batches:
                    print(
                        f"[ref-mc4 gpu={args.gpu}] timestamp={timestamp_position}/{len(remaining)} "
                        f"pair={transition_position}/49 batch={batch_position}/{num_batches}",
                        flush=True,
                    )
            finished_terms += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / finished_terms * (total_terms - finished_terms)
            print(
                f"[ref-mc4 gpu={args.gpu}] term={finished_terms}/{total_terms} "
                f"delta_norm=[{min(delta_norms):.3e},{max(delta_norms):.3e}] "
                f"eta={eta/3600:.2f}h",
                flush=True,
            )
            del model, target, named, parameters, specs, query_features, query_matrix
            del gradients, train_features, dots, noises, batched_gradient
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
                "batch_size": args.batch_size,
                "epsilon": args.epsilon,
                "train_mc": train_mc,
                "query_mc": args.query_mc,
                "completed_timestamps": completed,
            },
        )

    for name, values in scores.items():
        atomic_numpy(root / f"{name}.npy", values.cpu().numpy())
    atomic_json(
        done_path,
        {
            "methods": methods,
            "query_ids": list(REF_MC4_QUERY_IDS),
            "timestamp_indices": selected,
            "epsilon": args.epsilon,
            "query_mc": args.query_mc,
            "query_perturbation": "unit-L2 Gaussian directions around cached reference trajectory state",
            "perturbation_sharing": "fixed across checkpoints; independent by query/timestamp/mc",
            "train_mc": train_mc,
            "train_noise": "independent from query perturbations",
            "projection": "countsketch",
            "projection_dim": TRACIN_PROJ_DIM,
            "parameter_source": "raw",
            "checkpoint_target": "next",
            "batch_size": args.batch_size,
            "contract_version": CONTRACT_VERSION,
        },
    )
    print(f"[done] {root}", flush=True)


if __name__ == "__main__":
    main()
