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
    tracin_interval_mean_lr,
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


def sample_trajectory_cone_noises(
    axes, query_mc, generator, max_angle_degrees
):
    """Keep Gaussian radii while sampling directions inside a trajectory cone."""
    flat_axes = axes.reshape(axes.shape[0], -1)
    flat_axes = flat_axes / flat_axes.norm(dim=1, keepdim=True).clamp_min(
        TRACIN_DAS_DIRECTION_EPS
    )
    shape = (axes.shape[0], int(query_mc), flat_axes.shape[1])
    gaussian = torch.randn(
        shape, generator=generator, device=axes.device, dtype=axes.dtype
    )
    radii = gaussian.norm(dim=2, keepdim=True)
    axis_bank = flat_axes[:, None, :]
    orthogonal = gaussian - (gaussian * axis_bank).sum(
        dim=2, keepdim=True
    ) * axis_bank
    orthogonal = orthogonal / orthogonal.norm(dim=2, keepdim=True).clamp_min(
        TRACIN_DAS_DIRECTION_EPS
    )
    angles = torch.rand(
        (axes.shape[0], int(query_mc), 1),
        generator=generator,
        device=axes.device,
        dtype=axes.dtype,
    ) * math.radians(float(max_angle_degrees))
    directions = torch.cos(angles) * axis_bank + torch.sin(angles) * orthogonal
    return (radii * directions).reshape(
        axes.shape[0], int(query_mc), *axes.shape[1:]
    )


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
    parser.add_argument("--family", choices=FAMILIES, default=TRACIN_DAS_FAMILY)
    parser.add_argument(
        "--query-scope", choices=("ten", "all", "first99"), default="ten"
    )
    parser.add_argument(
        "--avg-pair-lr-multi",
        action="store_true",
        help=(
            "simultaneously save the 50/40/25/20/15/10/5 checkpoint banks "
            "with exact interval-mean learning-rate weights"
        ),
    )
    parser.add_argument(
        "--timestamp-count-multi",
        action="store_true",
        help=(
            "simultaneously save the 100/90/.../10 evenly spaced timestamp "
            "banks with the source checkpoint's saved learning-rate weight"
        ),
    )
    parser.add_argument("--noise-mode", choices=TRACIN_DAS_NOISE_MODES, default="checkpoint")
    parser.add_argument(
        "--parameter-projection",
        choices=TRACIN_DAS_PARAMETER_PROJECTIONS,
        default="exact",
    )
    parser.add_argument(
        "--train-noise-mode",
        choices=TRACIN_DAS_TRAIN_NOISE_MODES,
        default="aligned",
    )
    parser.add_argument("--query-mc", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.timestamp_shard_index < args.timestamp_shard_count:
        raise ValueError("invalid timestamp shard")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.query_mc <= 0:
        raise ValueError("query MC must be positive")
    if args.train_noise_mode == "aligned" and args.query_mc != 1:
        raise ValueError("query MC > 1 requires independent train noise")
    train_mc_count = (
        1
        if args.train_noise_mode in ("aligned", "independent-mc1")
        else int(TRACIN_TRAIN_MC)
    )
    if args.avg_pair_lr_multi and args.timestamp_count_multi:
        raise ValueError("checkpoint-count and timestamp-count multi modes conflict")
    multi_sweep = (
        args.avg_pair_lr_multi or args.timestamp_count_multi
    )
    if multi_sweep and (
        args.query_scope != "first99"
        or args.noise_mode != "checkpoint"
        or args.parameter_projection != "projected4096"
        or args.train_noise_mode != "aligned"
        or args.query_mc != 1
    ):
        raise ValueError(
            "multi sweep requires first99/checkpoint/projected4096/aligned"
        )
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    if args.query_scope in ("all", "first99"):
        requested_ids = (
            TRACIN_DAS_ALL_QUERY_IDS
            if args.query_scope == "all"
            else TRACIN_DAS_FIRST99_QUERY_IDS
        )
        records = [
            by_id[qid]
            for qid in requested_ids
            if by_id[qid]["family"] == args.family
        ]
    else:
        if args.family != TRACIN_DAS_FAMILY:
            raise ValueError("the legacy ten-query scope only supports prompted")
        records = [by_id[qid] for qid in TRACIN_DAS_QUERY_IDS]
    query_ids = [int(record["query_id"]) for record in records]
    if not records or any(record["family"] != args.family for record in records):
        raise ValueError(f"invalid query bank for family={args.family}")
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    checkpoint_paths = model_paths(args.family)
    if len(checkpoint_paths) != 50:
        raise ValueError(
            f"expected 50 old-experiment checkpoints, found {len(checkpoint_paths)}"
        )
    schedule = base.make_linear_schedule(T, device=device)
    x_all, cond_all = preload_dataset(dataset, args.family, device)
    endpoints = [
        torch.from_numpy(np.load(Path(record["dir"]) / "final_state.npy")).to(
            device=device, dtype=torch.float32
        )
        for record in records
    ]
    trajectory_axes = None
    if args.noise_mode == "trajectory-cone60":
        initial_states = [
            torch.from_numpy(
                np.load(
                    Path(record["dir"]) / "trajectory_xt.npy", mmap_mode="r"
                )[0].copy()
            ).to(device=device, dtype=torch.float32)
            for record in records
        ]
        trajectory_axes = torch.stack(
            [initial - endpoint for initial, endpoint in zip(initial_states, endpoints)]
        )
        if torch.any(
            trajectory_axes.reshape(len(records), -1).norm(dim=1)
            <= TRACIN_DAS_DIRECTION_EPS
        ):
            raise ValueError("zero endpoint-to-initial trajectory axis")
    conditions = [cond_for(record, dataset, device) for record in records]
    timestamps = tuple(int(value) for value in DAS_TIMESTEPS)
    selected = list(range(args.timestamp_shard_index, len(timestamps), args.timestamp_shard_count))
    if args.avg_pair_lr_multi:
        methods_by_group = {
            str(count): tracin_das_avg_pair_lr_methods(count)
            for count in TRACIN_DAS_AVG_LR_CHECKPOINT_COUNTS
        }
        pair_indices_by_group = {
            str(count): set(tracin_das_checkpoint_pair_indices(count))
            for count in TRACIN_DAS_AVG_LR_CHECKPOINT_COUNTS
        }
        shard_root = tracin_das_avg_pair_lr_shard_root(
            args.family,
            args.timestamp_shard_index,
            args.timestamp_shard_count,
        )
        timestamp_indices_by_group = {
            group: set(range(len(timestamps))) for group in methods_by_group
        }
    elif args.timestamp_count_multi:
        methods_by_group = {
            str(count): tracin_das_checkpoint_lr_timestamp_methods(count)
            for count in TRACIN_DAS_TIMESTAMP_COUNTS
        }
        pair_indices_by_group = {
            group: set(range(49)) for group in methods_by_group
        }
        timestamp_indices_by_group = {
            str(count): set(tracin_das_timestamp_indices(count))
            for count in TRACIN_DAS_TIMESTAMP_COUNTS
        }
        shard_root = tracin_das_timestamp_sweep_shard_root(
            args.family,
            args.timestamp_shard_index,
            args.timestamp_shard_count,
        )
    else:
        methods_by_group = {
            "legacy": tracin_das_methods(
                args.noise_mode,
                args.parameter_projection,
                args.train_noise_mode,
                args.query_mc,
            )
        }
        pair_indices_by_group = {"legacy": set(range(49))}
        timestamp_indices_by_group = {"legacy": set(range(len(timestamps)))}
        shard_root = tracin_das_shard_root(
            args.timestamp_shard_index,
            args.timestamp_shard_count,
            args.noise_mode,
            args.parameter_projection,
            args.train_noise_mode,
            args.family,
            args.query_scope,
            args.query_mc,
        )
    contractions = ("linear", "termwise_squared", "timestamp_sum_squared")
    snapshot_weight_by_group = (
        {group: 1.0 / float(group) for group in methods_by_group}
        if args.timestamp_count_multi
        else {group: 1.0 / len(timestamps) for group in methods_by_group}
    )
    done_path = shard_root / "done.json"
    if done_path.is_file():
        print(f"[skip] {done_path}", flush=True)
        return

    partial_paths = {
        group: {
            contraction: shard_root / (
                f"partial_{contraction}.npy"
                if group == "legacy"
                else f"partial_{group}_{contraction}.npy"
            )
            for contraction in contractions
        }
        for group in methods_by_group
    }
    progress_path = shard_root / "progress.json"
    completed_timestamps = []
    expected_shape = (len(records), N_TRAIN)
    all_partial_paths = [
        path
        for group_paths in partial_paths.values()
        for path in group_paths.values()
    ]
    if all(path.is_file() for path in all_partial_paths) and progress_path.is_file():
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
        if progress.get("train_noise_mode", "aligned") != args.train_noise_mode:
            raise ValueError("partial shard train-noise mode differs")
        if int(progress.get("query_mc", 1)) != args.query_mc:
            raise ValueError("partial shard query MC differs")
        if progress.get("family", TRACIN_DAS_FAMILY) != args.family:
            raise ValueError("partial shard family differs")
        if progress.get("query_scope", "ten") != args.query_scope:
            raise ValueError("partial shard query scope differs")
        if progress.get("query_ids", list(TRACIN_DAS_QUERY_IDS)) != query_ids:
            raise ValueError("partial shard query IDs differ")
        if bool(progress.get("avg_pair_lr_multi", False)) != args.avg_pair_lr_multi:
            raise ValueError("partial shard learning-rate/checkpoint mode differs")
        if bool(progress.get("timestamp_count_multi", False)) != args.timestamp_count_multi:
            raise ValueError("partial shard timestamp-count mode differs")
        completed_timestamps = [int(value) for value in progress["completed_timestamps"]]
        scores = {
            group: {
                contraction: torch.from_numpy(np.load(path)).to(
                    device=device, dtype=torch.float64
                )
                for contraction, path in group_paths.items()
            }
            for group, group_paths in partial_paths.items()
        }
        if any(
            value.shape != expected_shape
            for values in scores.values()
            for value in values.values()
        ):
            raise ValueError("partial score shape mismatch")
        print(f"[resume] timestamps={len(completed_timestamps)}/{len(selected)}", flush=True)
    else:
        scores = {
            group: {
                contraction: torch.zeros(
                    expected_shape, device=device, dtype=torch.float64
                )
                for contraction in contractions
            }
            for group in methods_by_group
        }

    transitions = [
        (index, index + 1) for index in range(len(checkpoint_paths) - 1)
    ]
    remaining = [index for index in selected if index not in set(completed_timestamps)]
    total_terms = len(remaining) * len(transitions)
    completed_terms = 0
    started = time.perf_counter()
    print(
        f"[tracin-das gpu={args.gpu}] family={args.family} "
        f"queries={query_ids[0]}..{query_ids[-1]} ({len(query_ids)}) "
        f"parameter_projection={args.parameter_projection} "
        f"timestamps={len(selected)}/100 transitions=49 batch={args.batch_size} "
        f"noise_mode={args.noise_mode} "
        f"train_noise_mode={args.train_noise_mode} "
        f"query_mc={args.query_mc} "
        f"avg_pair_lr_multi={args.avg_pair_lr_multi} "
        f"timestamp_count_multi={args.timestamp_count_multi} "
        f"query_train_noise_aligned={args.train_noise_mode == 'aligned'} "
        "output_delta_normalized=true",
        flush=True,
    )

    for shard_timestamp_position, timestamp_index in enumerate(remaining, start=1):
        timestep = timestamps[timestamp_index]
        t_query = torch.tensor([timestep], device=device, dtype=torch.long)
        timestamp_accumulators = {
            group: torch.zeros(expected_shape, device=device, dtype=torch.float64)
            for group in methods_by_group
        }

        for transition_position, (checkpoint_index, target_index) in enumerate(transitions, start=1):
            active_groups = [
                group
                for group, pair_indices in pair_indices_by_group.items()
                if checkpoint_index in pair_indices
                and timestamp_index in timestamp_indices_by_group[group]
            ]
            if not active_groups:
                continue
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
                    "tracin_das_trajectory_cone60_noise",
                    checkpoint_index,
                    timestamp_index,
                    timestep,
                )
                if args.noise_mode == "trajectory-cone60"
                else (
                    TRACIN_DAS_NOISE_SEED,
                    "tracin_das_timestamp_shared_noise",
                    timestamp_index,
                    timestep,
                )
            )
            noise_generator = make_torch_generator(device, *noise_seed_parts)
            if args.noise_mode == "trajectory-cone60":
                query_noises = sample_trajectory_cone_noises(
                    trajectory_axes,
                    args.query_mc,
                    noise_generator,
                    TRACIN_DAS_TRAJECTORY_CONE_DEGREES,
                )
            else:
                shared_noises = torch.randn(
                    (args.query_mc, *endpoints[0].shape),
                    generator=noise_generator,
                    device=device,
                    dtype=endpoints[0].dtype,
                )
                query_noises = shared_noises.unsqueeze(0).expand(
                    len(records), *shared_noises.shape
                )
            query_xt = [
                [
                    base.q_sample(
                        endpoint,
                        t_query,
                        query_noises[query_position, mc_index],
                        schedule,
                    )
                    for mc_index in range(args.query_mc)
                ]
                for query_position, endpoint in enumerate(endpoints)
            ]
            model, _, checkpoint = build_model(
                checkpoint_paths[checkpoint_index], "raw", device
            )
            target, _, target_checkpoint = build_model(
                checkpoint_paths[target_index], "raw", device
            )
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
            for xt_bank, condition in zip(query_xt, conditions):
                for xt in xt_bank:
                    current_prediction = model(xt, t_query, condition)
                    with torch.no_grad():
                        next_prediction = target(xt, t_query, condition)
                        direction = next_prediction - current_prediction.detach()
                        delta_norm = direction.norm()
                        direction = direction / delta_norm.clamp_min(
                            TRACIN_DAS_DIRECTION_EPS
                        )
                    projected_prediction = (current_prediction * direction).sum()
                    query_gradient = torch.autograd.grad(
                        projected_prediction, parameters
                    )
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

            query_dependent_aligned_noise = (
                args.noise_mode == "trajectory-cone60"
                and args.train_noise_mode == "aligned"
            )
            if args.train_noise_mode == "aligned":
                def train_loss(parameter_dict, x0, condition, aligned_noise):
                    xt = base.q_sample(
                        x0.unsqueeze(0), t_query, aligned_noise, schedule
                    )
                    prediction = functional_call(
                        model,
                        parameter_dict,
                        (xt, t_query, condition.unsqueeze(0)),
                    )
                    return (prediction - aligned_noise).square().mean()

                batched_gradient = vmap(
                    grad(train_loss), in_dims=(None, 0, 0, None)
                )
            else:
                train_mc = train_mc_count
                train_timesteps = torch.full(
                    (train_mc,), timestep, device=device, dtype=torch.long
                )

                def train_loss(parameter_dict, x0, condition, noises):
                    x_mc = x0.unsqueeze(0).expand(train_mc, *x0.shape)
                    condition_mc = condition.unsqueeze(0).expand(
                        train_mc, condition.shape[-1]
                    )
                    xt = base.q_sample(
                        x_mc, train_timesteps, noises, schedule
                    )
                    prediction = functional_call(
                        model,
                        parameter_dict,
                        (xt, train_timesteps, condition_mc),
                    )
                    return (prediction - noises).square().reshape(
                        train_mc, -1
                    ).mean(dim=1).mean()

                batched_gradient = vmap(
                    grad(train_loss), in_dims=(None, 0, 0, 0)
                )
            if args.avg_pair_lr_multi:
                checkpoint_lr = float(
                    tracin_interval_mean_lr(checkpoint, target_checkpoint)
                )
            else:
                checkpoint_lr = float(tracin_lr_weight(checkpoint))
            num_batches = math.ceil(N_TRAIN / args.batch_size)
            progress_every = max(1, num_batches // 5)
            for batch_position, start in enumerate(range(0, N_TRAIN, args.batch_size), start=1):
                end = min(start + args.batch_size, N_TRAIN)
                train_noises = None
                if args.train_noise_mode != "aligned":
                    train_noise_generator = make_torch_generator(
                        device,
                        TRAIN_SEED,
                        f"tracin_das_{args.train_noise_mode}",
                        checkpoint_index,
                        timestamp_index,
                        start,
                        train_mc,
                    )
                    train_noises = torch.randn(
                        (
                            end - start,
                            train_mc,
                            *x_all.shape[1:],
                        ),
                        generator=train_noise_generator,
                        device=device,
                        dtype=x_all.dtype,
                    )
                if query_dependent_aligned_noise:
                    dot_rows = []
                    for query_position in range(len(records)):
                        gradients = batched_gradient(
                            named,
                            x_all[start:end],
                            cond_all[start:end],
                            query_noises[query_position, 0],
                        )
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
                        dot_rows.append(
                            train_matrix @ query_matrix[query_position]
                        )
                        del gradients, train_matrix
                    dots = torch.stack(dot_rows, dim=0).unsqueeze(1).to(
                        torch.float64
                    )
                    del dot_rows
                else:
                    gradients = (
                        batched_gradient(
                            named,
                            x_all[start:end],
                            cond_all[start:end],
                            query_noises[0, 0],
                        )
                        if args.train_noise_mode == "aligned"
                        else batched_gradient(
                            named,
                            x_all[start:end],
                            cond_all[start:end],
                            train_noises,
                        )
                    )
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
                    dots = (train_matrix @ query_matrix.T).T.reshape(
                        len(records), args.query_mc, end - start
                    ).to(torch.float64)
                linear_dots = dots.mean(dim=1)
                squared_dots = dots.square().mean(dim=1)
                for group in active_groups:
                    weight = checkpoint_lr * snapshot_weight_by_group[group]
                    scores[group]["linear"][:, start:end] += weight * linear_dots
                    scores[group]["termwise_squared"][:, start:end] += (
                        weight * squared_dots
                    )
                    timestamp_accumulators[group][:, start:end] += (
                        weight * linear_dots
                    )
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
            del model, target, checkpoint, target_checkpoint
            del named, parameters, query_vectors, query_matrix
            del projection_specs
            del current_prediction, next_prediction, direction, projected_prediction, query_gradient
            del batched_gradient, dots, query_noises, query_xt
            if not query_dependent_aligned_noise:
                del gradients, train_matrix
            if args.noise_mode != "trajectory-cone60":
                del shared_noises
            del linear_dots, squared_dots
            del train_loss, train_noises
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for group in methods_by_group:
            scores[group]["timestamp_sum_squared"] += (
                timestamp_accumulators[group].square()
            )
        completed_timestamps.append(timestamp_index)
        completed_timestamps.sort()
        checkpoint_every = (
            10
            if args.timestamp_count_multi
            else 5
            if args.avg_pair_lr_multi
            else 1
        )
        should_checkpoint = (
            len(completed_timestamps) % checkpoint_every == 0
            or len(completed_timestamps) == len(selected)
        )
        if should_checkpoint:
            for group, group_paths in partial_paths.items():
                for contraction, path in group_paths.items():
                    atomic_numpy(path, scores[group][contraction].cpu().numpy())
            atomic_json(
                progress_path,
                {
                    "contract_version": CONTRACT_VERSION,
                    "batch_size": args.batch_size,
                    "noise_mode": args.noise_mode,
                    "parameter_projection": args.parameter_projection,
                    "train_noise_mode": args.train_noise_mode,
                    "query_mc": args.query_mc,
                    "query_ids": query_ids,
                    "family": args.family,
                    "query_scope": args.query_scope,
                    "avg_pair_lr_multi": args.avg_pair_lr_multi,
                    "timestamp_count_multi": args.timestamp_count_multi,
                    "checkpoint_counts": (
                        list(TRACIN_DAS_AVG_LR_CHECKPOINT_COUNTS)
                        if args.avg_pair_lr_multi
                        else None
                    ),
                    "timestamp_counts": (
                        list(TRACIN_DAS_TIMESTAMP_COUNTS)
                        if args.timestamp_count_multi
                        else None
                    ),
                    "completed_timestamps": completed_timestamps,
                },
            )
            print(
                f"[checkpoint] timestamps={len(completed_timestamps)}/{len(selected)}",
                flush=True,
            )

    for group, values in scores.items():
        for contraction, value in values.items():
            atomic_numpy(
                shard_root / (
                    f"{contraction}.npy"
                    if group == "legacy"
                    else f"{group}_{contraction}.npy"
                ),
                value.cpu().numpy(),
            )
    atomic_json(
        done_path,
        {
            "methods": (
                methods_by_group["legacy"]
                if "legacy" in methods_by_group
                else None
            ),
            "methods_by_group": methods_by_group,
            "query_ids": query_ids,
            "family": args.family,
            "query_scope": args.query_scope,
            "avg_pair_lr_multi": args.avg_pair_lr_multi,
            "timestamp_count_multi": args.timestamp_count_multi,
            "checkpoint_pair_indices": {
                group: sorted(indices)
                for group, indices in pair_indices_by_group.items()
            },
            "selected_timestamp_indices_by_group": {
                group: sorted(indices)
                for group, indices in timestamp_indices_by_group.items()
            },
            "timestamp_indices": selected,
            "timestamps": [timestamps[index] for index in selected],
            "checkpoint_transitions": 49,
            "parameter_source": "raw",
            "endpoint_source": "cached final-EMA query endpoint",
            "noise_mode": args.noise_mode,
            "parameter_projection": args.parameter_projection,
            "train_noise_mode": args.train_noise_mode,
            "query_mc": args.query_mc,
            "train_mc": (
                train_mc_count
            ),
            "parameter_projection_dim": (
                TRACIN_PROJ_DIM
                if args.parameter_projection == "projected4096"
                else None
            ),
            "endpoint_noising": tracin_das_endpoint_noising_description(
                args.noise_mode, args.query_mc
            ),
            "train_loss_noise": (
                "same term noise as query endpoint"
                if args.train_noise_mode == "aligned"
                else f"independent per-datapoint MC{train_mc_count} noises"
            ),
            "query_scalar": "dot(epsilon_current, normalize(epsilon_next-epsilon_current))",
            "query_mc_reduction": {
                "linear": "mean_m(dot_m)",
                "termwise_squared": "mean_m(dot_m^2)",
                "timestamp_sum_squared": (
                    "sum_t (sum_checkpoint eta * mean_m(dot_m))^2"
                ),
            },
            "lr_weighted": TRACIN_USE_LR_WEIGHTS,
            "learning_rate_source": (
                "exact mean scheduled LR over [current global_step, next global_step)"
                if args.avg_pair_lr_multi
                else "source checkpoint saved eta"
            ),
            "timestamp_weight_by_group": snapshot_weight_by_group,
            "batch_size": args.batch_size,
            "contract_version": CONTRACT_VERSION,
        },
    )
    print(f"[done] {shard_root}", flush=True)


if __name__ == "__main__":
    main()
