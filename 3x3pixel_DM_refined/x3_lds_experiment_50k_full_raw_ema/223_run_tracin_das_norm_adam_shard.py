"""Raw and AdamW-aware aligned TracIn-DAS with four L2 variants."""

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
from tracin_das_norm_adam_config import *
from x3_endpoint_das_jax_logic_pytorch import (
    build_countsketch_specs,
    make_torch_generator,
)


CONTRACT_VERSION = TDNA_CONTRACT_VERSION


def optimizer_state_by_name(checkpoint, names, device):
    """Map saved AdamW state to model names in optimizer parameter order."""
    optimizer_state = checkpoint["optimizer_state"]
    groups = optimizer_state["param_groups"]
    if len(groups) != 1:
        raise ValueError("the X3 AdamW-aware transform expects one parameter group")
    parameter_ids = list(groups[0]["params"])
    if len(parameter_ids) != len(names):
        raise ValueError("optimizer/model parameter count mismatch")
    states = optimizer_state["state"]
    by_name = {}
    for name, parameter_id in zip(names, parameter_ids):
        state = states[parameter_id]
        by_name[name] = {
            "step": int(torch.as_tensor(state["step"]).item()),
            "exp_avg": state["exp_avg"].to(device=device, dtype=torch.float32),
            "exp_avg_sq": state["exp_avg_sq"].to(
                device=device, dtype=torch.float32
            ),
        }
        if "max_exp_avg_sq" in state:
            by_name[name]["max_exp_avg_sq"] = state["max_exp_avg_sq"].to(
                device=device, dtype=torch.float32
            )
    group = groups[0]
    hyper = {
        "lr": float(group["lr"]),
        "beta1": float(group["betas"][0]),
        "beta2": float(group["betas"][1]),
        "eps": float(group["eps"]),
        "weight_decay": float(group.get("weight_decay", 0.0)),
        "amsgrad": bool(group.get("amsgrad", False)),
        "maximize": bool(group.get("maximize", False)),
    }
    return by_name, hyper


def clip_batched_gradients(gradients, names, max_norm):
    squared = None
    for name in names:
        value = gradients[name].float()
        term = value.square().flatten(1).sum(dim=1)
        squared = term if squared is None else squared + term
    norms = squared.sqrt()
    scales = (float(max_norm) / norms.clamp_min(TDNA_EPS)).clamp(max=1.0)
    result = {}
    for name in names:
        shape = (len(scales),) + (1,) * (gradients[name].ndim - 1)
        result[name] = gradients[name].float() * scales.reshape(shape)
    return result, norms


def adamw_full_batched(gradients, names, parameters, optimizer_state, hyper):
    """Full one-step AdamW parameter delta under the saved optimizer state."""
    clipped, unclipped_norms = clip_batched_gradients(
        gradients, names, GRAD_CLIP
    )
    beta1 = hyper["beta1"]
    beta2 = hyper["beta2"]
    eps = hyper["eps"]
    learning_rate = hyper["lr"]
    sign = -1.0 if hyper["maximize"] else 1.0
    output = {}
    for name in names:
        state = optimizer_state[name]
        gradient = sign * clipped[name]
        exp_avg = state["exp_avg"].unsqueeze(0)
        exp_avg_sq = state["exp_avg_sq"].unsqueeze(0)
        step = state["step"] + 1
        bias1 = 1.0 - beta1 ** step
        bias2 = 1.0 - beta2 ** step

        moment = beta1 * exp_avg + (1.0 - beta1) * gradient
        variance = beta2 * exp_avg_sq + (1.0 - beta2) * gradient.square()
        if hyper["amsgrad"]:
            previous_max = state["max_exp_avg_sq"].unsqueeze(0)
            variance = torch.maximum(previous_max, variance)
        denominator = variance.sqrt() / math.sqrt(bias2) + eps
        step_size = learning_rate / bias1
        decay = -learning_rate * hyper["weight_decay"] * parameters[name]
        output[name] = decay.unsqueeze(0) - step_size * moment / denominator
    return output, unclipped_norms


def normalized_dot_variants(query_matrix, train_matrix):
    raw = (train_matrix @ query_matrix.T).T
    query_norm = query_matrix.norm(dim=1).clamp_min(TDNA_EPS)
    train_norm = train_matrix.norm(dim=1).clamp_min(TDNA_EPS)
    return {
        "raw": raw,
        "query_l2": raw / query_norm[:, None],
        "train_l2": raw / train_norm[None, :],
        "query_train_l2": raw / (
            query_norm[:, None] * train_norm[None, :]
        ),
    }


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
    parser.add_argument(
        "--aligned-mc10-full-only",
        action="store_true",
        help=(
            "full AdamW only: retain ten aligned query/train noise terms "
            "through checkpoint summation before timestamp-wise squaring"
        ),
    )
    args = parser.parse_args()
    if not 0 <= args.timestamp_shard_index < args.timestamp_shard_count:
        raise ValueError("invalid timestamp shard")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.query_mc <= 0:
        raise ValueError("query MC must be positive")
    if (
        args.train_noise_mode == "aligned"
        and args.query_mc != 1
        and not args.aligned_mc10_full_only
    ):
        raise ValueError("query MC > 1 requires independent train noise")
    if args.aligned_mc10_full_only:
        if (
            args.query_scope != "all"
            or args.noise_mode != "checkpoint"
            or args.parameter_projection != "projected4096"
            or args.train_noise_mode != "aligned"
            or args.query_mc != TDNA_MC10
            or args.avg_pair_lr_multi
            or args.timestamp_count_multi
        ):
            raise ValueError(
                "aligned MC10 full mode requires "
                "all/checkpoint/projected4096/aligned/query-mc=10"
            )
    elif (
        args.query_scope != "all"
        or args.noise_mode != "checkpoint"
        or args.parameter_projection != "projected4096"
        or args.train_noise_mode != "aligned"
        or args.query_mc != 1
        or args.avg_pair_lr_multi
        or args.timestamp_count_multi
    ):
        raise ValueError(
            "this worker requires all/checkpoint/projected4096/aligned/MC1"
        )
    if (
        args.noise_mode == "trajectory-cone60"
        and args.train_noise_mode == "aligned"
    ):
        raise ValueError(
            "trajectory-cone60 is query-dependent and requires independent "
            "train noise"
        )
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
    elif args.aligned_mc10_full_only:
        methods_by_group = {
            f"adamw_full__{variant}": {
                "timestamp_sum_squared": method
            }
            for variant, method in tdna_mc10_methods().items()
        }
        pair_indices_by_group = {
            group: set(range(49)) for group in methods_by_group
        }
        timestamp_indices_by_group = {
            group: set(range(len(timestamps))) for group in methods_by_group
        }
        shard_root = tdna_mc10_shard_root(
            args.family,
            args.timestamp_shard_index,
            args.timestamp_shard_count,
        )
    else:
        nested_methods = tdna_methods()
        methods_by_group = {
            f"{transform}__{variant}": nested_methods[transform][variant]
            for transform in TDNA_TRANSFORMS
            for variant in TDNA_VARIANTS
        }
        pair_indices_by_group = {
            group: set(range(49)) for group in methods_by_group
        }
        timestamp_indices_by_group = {
            group: set(range(len(timestamps))) for group in methods_by_group
        }
        shard_root = tdna_shard_root(
            args.family,
            args.timestamp_shard_index,
            args.timestamp_shard_count,
        )
    contractions = (
        ("timestamp_sum_squared",)
        if args.aligned_mc10_full_only
        else ("linear", "termwise_squared", "timestamp_sum_squared")
    )
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
        if bool(progress.get("aligned_mc10_full_only", False)) != args.aligned_mc10_full_only:
            raise ValueError("partial shard aligned-MC10 mode differs")
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
        f"aligned_mc10_full_only={args.aligned_mc10_full_only} "
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
            group: torch.zeros(
                (
                    (len(records), args.query_mc, N_TRAIN)
                    if args.aligned_mc10_full_only
                    else expected_shape
                ),
                device=device,
                dtype=torch.float64,
            )
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
            adam_state, adam_hyper = optimizer_state_by_name(
                checkpoint, names, device
            )
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

            if args.aligned_mc10_full_only:
                def single_train_loss(parameter_dict, x0, condition, noise):
                    xt = base.q_sample(
                        x0.unsqueeze(0), t_query, noise, schedule
                    )
                    prediction = functional_call(
                        model,
                        parameter_dict,
                        (xt, t_query, condition.unsqueeze(0)),
                    )
                    return (prediction - noise).square().mean()

                per_noise_gradient = vmap(
                    grad(single_train_loss), in_dims=(None, None, None, 0)
                )
                batched_gradient = vmap(
                    per_noise_gradient, in_dims=(None, 0, 0, None)
                )
            elif args.train_noise_mode == "aligned":
                def train_loss(parameter_dict, x0, condition):
                    xt = base.q_sample(
                        x0.unsqueeze(0), t_query, query_noises[0, 0], schedule
                    )
                    prediction = functional_call(
                        model,
                        parameter_dict,
                        (xt, t_query, condition.unsqueeze(0)),
                    )
                    return (prediction - query_noises[0, 0]).square().mean()

                batched_gradient = vmap(
                    grad(train_loss), in_dims=(None, 0, 0)
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
                gradients = (
                    batched_gradient(
                        named,
                        x_all[start:end],
                        cond_all[start:end],
                        query_noises[0],
                    )
                    if args.aligned_mc10_full_only
                    else batched_gradient(
                        named,
                        x_all[start:end],
                        cond_all[start:end],
                    )
                    if args.train_noise_mode == "aligned"
                    else batched_gradient(
                        named,
                        x_all[start:end],
                        cond_all[start:end],
                        train_noises,
                    )
                )
                original_batch = end - start
                if args.aligned_mc10_full_only:
                    gradients = {
                        name: value.flatten(0, 1)
                        for name, value in gradients.items()
                    }
                train_features = {} if args.aligned_mc10_full_only else {
                    "gradient": gradients,
                }
                adam_gradients, unclipped_norms = adamw_full_batched(
                    gradients, names, named, adam_state, adam_hyper
                )
                train_features["adamw_full"] = adam_gradients
                dot_variants = {}
                train_matrices = {}
                for transform, feature in train_features.items():
                    matrix = _project_batched_grads(
                        feature,
                        names,
                        projection_specs,
                        TRACIN_PROJ_DIM,
                        False,
                        1e-8,
                    ).detach()
                    train_matrices[transform] = matrix
                    if args.aligned_mc10_full_only:
                        query_banked = query_matrix.reshape(
                            len(records), args.query_mc, -1
                        )
                        train_banked = matrix.reshape(
                            original_batch, args.query_mc, -1
                        )
                        raw = torch.einsum(
                            "qmd,bmd->qmb", query_banked, train_banked
                        )
                        query_norm = query_banked.norm(dim=2).clamp_min(TDNA_EPS)
                        train_norm = train_banked.norm(dim=2).clamp_min(TDNA_EPS)
                        dot_variants[transform] = {
                            "raw": raw,
                            "query_l2": raw / query_norm[:, :, None],
                            "train_l2": raw / train_norm.T[None, :, :],
                            "query_train_l2": raw / (
                                query_norm[:, :, None]
                                * train_norm.T[None, :, :]
                            ),
                        }
                    else:
                        dot_variants[transform] = normalized_dot_variants(
                            query_matrix, matrix
                        )
                for group in active_groups:
                    transform, variant = group.split("__", 1)
                    dots = dot_variants[transform][variant].reshape(
                        len(records), args.query_mc, end - start
                    ).to(torch.float64)
                    if args.aligned_mc10_full_only:
                        timestamp_accumulators[group][:, :, start:end] += dots
                        continue
                    linear_dots = dots.mean(dim=1)
                    squared_dots = dots.square().mean(dim=1)
                    # The full AdamW feature is already a parameter update and
                    # therefore already contains the checkpoint learning rate.
                    outer_weight = (
                        checkpoint_lr if transform == "gradient" else 1.0
                    )
                    weight = outer_weight * snapshot_weight_by_group[group]
                    scores[group]["linear"][:, start:end] += weight * linear_dots
                    scores[group]["termwise_squared"][:, start:end] += (
                        weight * squared_dots
                    )
                    timestamp_accumulators[group][:, start:end] += (
                        weight * linear_dots
                    )
                del adam_gradients, unclipped_norms
                del train_features, train_matrices, dot_variants
                del dots
                if not args.aligned_mc10_full_only:
                    del linear_dots, squared_dots
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
            del adam_state, adam_hyper
            del projection_specs
            del current_prediction, next_prediction, direction, projected_prediction, query_gradient
            del batched_gradient, gradients, query_noises, query_xt
            if args.noise_mode != "trajectory-cone60":
                del shared_noises
            if args.aligned_mc10_full_only:
                del single_train_loss, per_noise_gradient
            else:
                del train_loss
            del train_noises
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for group in methods_by_group:
            if args.aligned_mc10_full_only:
                scores[group]["timestamp_sum_squared"] += (
                    timestamp_accumulators[group].square().mean(dim=1)
                    / float(len(timestamps))
                )
            else:
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
                    "aligned_mc10_full_only": args.aligned_mc10_full_only,
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
            "aligned_mc10_full_only": args.aligned_mc10_full_only,
            "train_mc": (
                args.query_mc
                if args.aligned_mc10_full_only
                else train_mc_count
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
                (
                    "ten separately differentiated loss terms, each using "
                    "the matching query endpoint noise direction"
                    if args.aligned_mc10_full_only
                    else "same term noise as query endpoint"
                )
                if args.train_noise_mode == "aligned"
                else f"independent per-datapoint MC{train_mc_count} noises"
            ),
            "query_scalar": "dot(epsilon_current, normalize(epsilon_next-epsilon_current))",
            "feature_transforms": {
                "gradient": (
                    "projected per-example simple-loss gradient; source checkpoint "
                    "saved eta applied at score contraction"
                ),
                "adamw_full": (
                    "full one-step AdamW parameter delta for each per-example "
                    "gradient using saved m/v/step, per-example global-norm clipping, "
                    "and weight decay; zero-gradient baseline is NOT subtracted; "
                    "no additional outer learning-rate multiplier"
                ),
            },
            "normalization_variants": {
                "raw": "dot(query, train)",
                "query_l2": "dot(query/||query||, train)",
                "train_l2": "dot(query, train/||train||)",
                "query_train_l2": (
                    "dot(query/||query||, train/||train||)"
                ),
            },
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
