"""Reference-trajectory TracIn with normalized next delta and full AdamW."""

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
)
from dataset_loader import ColorGridDataset
from exp_config import *
from x3_endpoint_das_jax_logic_pytorch import (
    build_countsketch_specs,
    make_torch_generator,
)


VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")
TRAIN_MC = 10
EPS = 1e-12
CONTRACT_VERSION = 1


def method_name(variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    return (
        "traj_tracin_reference_next_delta_normalized_projected4096_"
        f"adamw_full_independent_mc10_{variant}_timestamp_sum_squared"
    )


def shard_root(family, shard_index, shard_count):
    return (
        ATTR_DIR
        / "_traj_tracin_adamw_full_delta_norm_independent_mc10_100q_shards"
        / family
        / f"shard_{shard_index:02d}_of_{shard_count:02d}"
    )


def optimizer_state_by_name(checkpoint, names, device):
    optimizer_state = checkpoint["optimizer_state"]
    groups = optimizer_state["param_groups"]
    if len(groups) != 1:
        raise ValueError("expected one AdamW parameter group")
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


def clip_batched_gradients(gradients, names):
    squared = None
    for name in names:
        term = gradients[name].float().square().flatten(1).sum(dim=1)
        squared = term if squared is None else squared + term
    norms = squared.sqrt()
    scales = (float(GRAD_CLIP) / norms.clamp_min(EPS)).clamp(max=1.0)
    output = {}
    for name in names:
        shape = (len(scales),) + (1,) * (gradients[name].ndim - 1)
        output[name] = gradients[name].float() * scales.reshape(shape)
    return output


def adamw_full_batched(gradients, names, parameters, optimizer_state, hyper):
    clipped = clip_batched_gradients(gradients, names)
    beta1 = hyper["beta1"]
    beta2 = hyper["beta2"]
    sign = -1.0 if hyper["maximize"] else 1.0
    output = {}
    for name in names:
        state = optimizer_state[name]
        gradient = sign * clipped[name]
        step = state["step"] + 1
        moment = (
            beta1 * state["exp_avg"].unsqueeze(0)
            + (1.0 - beta1) * gradient
        )
        variance = (
            beta2 * state["exp_avg_sq"].unsqueeze(0)
            + (1.0 - beta2) * gradient.square()
        )
        if hyper["amsgrad"]:
            variance = torch.maximum(
                state["max_exp_avg_sq"].unsqueeze(0), variance
            )
        denominator = (
            variance.sqrt() / math.sqrt(1.0 - beta2 ** step) + hyper["eps"]
        )
        step_size = hyper["lr"] / (1.0 - beta1 ** step)
        decay = -hyper["lr"] * hyper["weight_decay"] * parameters[name]
        output[name] = decay.unsqueeze(0) - step_size * moment / denominator
    return output


def normalized_dots(query_matrix, train_matrix):
    raw = (train_matrix @ query_matrix.T).T
    query_norm = query_matrix.norm(dim=1).clamp_min(EPS)
    train_norm = train_matrix.norm(dim=1).clamp_min(EPS)
    return {
        "raw": raw,
        "query_l2": raw / query_norm[:, None],
        "train_l2": raw / train_norm[None, :],
        "query_train_l2": raw / (query_norm[:, None] * train_norm[None, :]),
    }


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
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--timestamp-shard-index", type=int, required=True)
    parser.add_argument("--timestamp-shard-count", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not 0 <= args.timestamp_shard_index < args.timestamp_shard_count:
        raise ValueError("invalid timestamp shard")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    records = [record for record in manifest if record["family"] == args.family]
    if len(records) != 50:
        raise ValueError(f"expected 50 {args.family} queries, found {len(records)}")
    query_ids = [int(record["query_id"]) for record in records]

    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    paths = model_paths(args.family)
    if len(paths) != 50:
        raise ValueError(f"expected 50 checkpoints, found {len(paths)}")
    schedule = base.make_linear_schedule(T, device=device)
    x_all, cond_all = preload_dataset(dataset, args.family, device)
    conditions = [cond_for(record, dataset, device) for record in records]
    trajectories = [
        np.load(Path(record["dir"]) / "trajectory_xt.npy", mmap_mode="r")
        for record in records
    ]
    timestep_arrays = [
        np.load(Path(record["dir"]) / "trajectory_t.npy") for record in records
    ]
    timesteps = timestep_arrays[0]
    if len(timesteps) != 100 or any(
        not np.array_equal(timesteps, value) for value in timestep_arrays
    ):
        raise ValueError("queries do not share one 100-timestamp trajectory grid")
    selected = list(
        range(args.timestamp_shard_index, 100, args.timestamp_shard_count)
    )

    root = shard_root(
        args.family, args.timestamp_shard_index, args.timestamp_shard_count
    )
    done_path = root / "done.json"
    if done_path.is_file():
        print(f"[skip] {done_path}", flush=True)
        return
    partial_paths = {
        variant: root / f"partial_{variant}.npy" for variant in VARIANTS
    }
    progress_path = root / "progress.json"
    expected_shape = (len(records), N_TRAIN)
    completed = []
    if (
        all(path.is_file() for path in partial_paths.values())
        and progress_path.is_file()
    ):
        with open(progress_path) as handle:
            progress = json.load(handle)
        if int(progress["contract_version"]) != CONTRACT_VERSION:
            raise ValueError("partial contract version changed")
        if int(progress["batch_size"]) != args.batch_size:
            raise ValueError("partial batch size differs")
        if progress["query_ids"] != query_ids:
            raise ValueError("partial query IDs differ")
        completed = [int(value) for value in progress["completed_timestamps"]]
        scores = {
            variant: torch.from_numpy(np.load(path)).to(
                device=device, dtype=torch.float64
            )
            for variant, path in partial_paths.items()
        }
        print(f"[resume] timestamps={len(completed)}/{len(selected)}", flush=True)
    else:
        scores = {
            variant: torch.zeros(
                expected_shape, device=device, dtype=torch.float64
            )
            for variant in VARIANTS
        }

    remaining = [index for index in selected if index not in set(completed)]
    total_terms = len(remaining) * 49
    completed_terms = 0
    started = time.perf_counter()
    print(
        f"[traj-adamw gpu={args.gpu}] family={args.family} "
        f"queries={query_ids[0]}..{query_ids[-1]} "
        f"timestamps={len(selected)}/100 transitions=49 train_mc={TRAIN_MC} "
        f"train_noise=independent query=reference_trajectory "
        f"delta=normalized projection=4096 batch={args.batch_size}",
        flush=True,
    )

    for local_timestamp, timestamp_index in enumerate(remaining, start=1):
        timestep = int(timesteps[timestamp_index])
        t_query = torch.tensor([timestep], device=device, dtype=torch.long)
        timestamp_accumulators = {
            variant: torch.zeros(
                expected_shape, device=device, dtype=torch.float64
            )
            for variant in VARIANTS
        }

        for checkpoint_index in range(49):
            term_started = time.perf_counter()
            model, _, checkpoint = build_model(
                paths[checkpoint_index], "raw", device
            )
            target, _, _ = build_model(
                paths[checkpoint_index + 1], "raw", device
            )
            named = dict(model.named_parameters())
            names = tuple(named)
            parameters = tuple(named.values())
            adam_state, adam_hyper = optimizer_state_by_name(
                checkpoint, names, device
            )
            specs = build_countsketch_specs(
                list(parameters),
                TRACIN_PROJ_DIM,
                device=device,
                seed_parts=(
                    TRAIN_SEED,
                    "traj_adamw_full_delta_norm_projection",
                    checkpoint_index,
                ),
            )

            query_vectors = []
            delta_norms = []
            for trajectory, condition in zip(trajectories, conditions):
                xt_query = torch.from_numpy(
                    np.asarray(trajectory[timestamp_index]).copy()
                ).to(device=device, dtype=torch.float32)
                current_prediction = model(xt_query, t_query, condition)
                with torch.no_grad():
                    next_prediction = target(xt_query, t_query, condition)
                    direction = next_prediction - current_prediction.detach()
                    delta_norm = direction.norm()
                    direction = direction / delta_norm.clamp_min(EPS)
                scalar = (current_prediction * direction).sum()
                query_gradient = torch.autograd.grad(scalar, parameters)
                query_vectors.append(
                    _project_gradient_tuple(
                        query_gradient, specs, TRACIN_PROJ_DIM
                    )
                )
                delta_norms.append(float(delta_norm))
            query_matrix = torch.stack(query_vectors).detach().to(torch.float32)

            train_timesteps = torch.full(
                (TRAIN_MC,), timestep, device=device, dtype=torch.long
            )

            def mean_train_loss(parameter_dict, x0, condition, noises):
                x_mc = x0.unsqueeze(0).expand(TRAIN_MC, *x0.shape)
                condition_mc = condition.unsqueeze(0).expand(
                    TRAIN_MC, condition.shape[-1]
                )
                xt = base.q_sample(
                    x_mc, train_timesteps, noises, schedule
                )
                prediction = functional_call(
                    model,
                    parameter_dict,
                    (xt, train_timesteps, condition_mc),
                )
                losses = (prediction - noises).square().reshape(TRAIN_MC, -1)
                return losses.mean(dim=1).mean()

            batched_gradient = vmap(
                grad(mean_train_loss), in_dims=(None, 0, 0, 0)
            )
            num_batches = math.ceil(N_TRAIN / args.batch_size)
            progress_every = max(1, num_batches // 5)
            for batch_position, start in enumerate(
                range(0, N_TRAIN, args.batch_size), start=1
            ):
                end = min(start + args.batch_size, N_TRAIN)
                generator = make_torch_generator(
                    device,
                    TRAIN_SEED,
                    "traj_adamw_full_independent_mc10",
                    checkpoint_index,
                    timestamp_index,
                    start,
                )
                noises = torch.randn(
                    (end - start, TRAIN_MC, *x_all.shape[1:]),
                    generator=generator,
                    device=device,
                    dtype=x_all.dtype,
                )
                gradients = batched_gradient(
                    named,
                    x_all[start:end],
                    cond_all[start:end],
                    noises,
                )
                updates = adamw_full_batched(
                    gradients, names, named, adam_state, adam_hyper
                )
                train_matrix = _project_batched_grads(
                    updates,
                    names,
                    specs,
                    TRACIN_PROJ_DIM,
                    False,
                    1e-8,
                ).detach()
                dots = normalized_dots(query_matrix, train_matrix)
                for variant in VARIANTS:
                    timestamp_accumulators[variant][:, start:end] += (
                        dots[variant].to(torch.float64)
                    )
                if (
                    batch_position == 1
                    or batch_position % progress_every == 0
                    or batch_position == num_batches
                ):
                    print(
                        f"[traj-adamw gpu={args.gpu}] "
                        f"timestamp={local_timestamp}/{len(remaining)} "
                        f"global_t={timestamp_index+1}/100 "
                        f"pair={checkpoint_index+1}/49 "
                        f"batch={batch_position}/{num_batches}",
                        flush=True,
                    )
            completed_terms += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / completed_terms * (total_terms - completed_terms)
            print(
                f"[traj-adamw gpu={args.gpu}] term={completed_terms}/{total_terms} "
                f"delta_norm=[{min(delta_norms):.3e},{max(delta_norms):.3e}] "
                f"term_elapsed={(time.perf_counter()-term_started)/60:.1f}m "
                f"eta={eta/3600:.2f}h",
                flush=True,
            )
            del model, target, named, parameters, adam_state, adam_hyper
            del query_vectors, query_matrix, query_gradient, scalar
            del current_prediction, next_prediction, direction
            del batched_gradient, gradients, updates, train_matrix, dots, noises
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for variant in VARIANTS:
            scores[variant] += (
                timestamp_accumulators[variant].square() / 100.0
            )
        completed.append(timestamp_index)
        completed.sort()
        for variant, path in partial_paths.items():
            atomic_numpy(path, scores[variant].cpu().numpy())
        atomic_json(
            progress_path,
            {
                "contract_version": CONTRACT_VERSION,
                "family": args.family,
                "batch_size": args.batch_size,
                "query_ids": query_ids,
                "completed_timestamps": completed,
            },
        )
        print(
            f"[checkpoint] timestamps={len(completed)}/{len(selected)}",
            flush=True,
        )

    for variant, path in partial_paths.items():
        atomic_numpy(path, scores[variant].cpu().numpy())
    atomic_json(
        done_path,
        {
            "contract_version": CONTRACT_VERSION,
            "family": args.family,
            "query_ids": query_ids,
            "timestamp_indices": selected,
            "checkpoint_transitions": 49,
            "query_input": "cached reference trajectory x_t",
            "query_delta": "L2-normalized next-minus-current predicted noise",
            "train_mc": TRAIN_MC,
            "train_noise": "independent of query trajectory and per datapoint",
            "parameter_transform": "full AdamW with saved m/v/step and clipping",
            "parameter_projection": "CountSketch4096",
            "variants": list(VARIANTS),
            "contraction": (
                "mean_t square(sum_checkpoint dot(query, AdamW(train_grad)))"
            ),
        },
    )
    print(f"[done] {done_path}", flush=True)


if __name__ == "__main__":
    main()
