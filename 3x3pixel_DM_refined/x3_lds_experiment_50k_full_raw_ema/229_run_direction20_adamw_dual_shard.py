"""Run X3 direction20/mean100t full-AdamW attribution.

Each checkpoint/direction train feature is the full saved-state AdamW update
obtained from the gradient of one loss averaged over 100 timesteps.  One
Gaussian axis is shared across those timesteps.  The feature is reused for:

* reference trajectory queries (no query/train noise alignment), and
* endpoint-polluted TracIn-DAS queries (direction aligned).

Checkpoint contributions are summed as vectors before each direction/timestamp
term is squared.  The final score averages the 20 x 100 squared terms.
"""

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
    build_model,
    cond_for,
    model_paths,
    preload_dataset,
)
from dataset_loader import ColorGridDataset
from direction20_adamw_dual_config import *
from x3_endpoint_das_jax_logic_pytorch import (
    build_countsketch_specs,
    make_torch_generator,
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
    return by_name, {
        "lr": float(group["lr"]),
        "beta1": float(group["betas"][0]),
        "beta2": float(group["betas"][1]),
        "eps": float(group["eps"]),
        "weight_decay": float(group.get("weight_decay", 0.0)),
        "amsgrad": bool(group.get("amsgrad", False)),
        "maximize": bool(group.get("maximize", False)),
    }


def adamw_full_batched(gradients, names, parameters, state, hyper):
    squared = None
    for name in names:
        term = gradients[name].float().square().flatten(1).sum(dim=1)
        squared = term if squared is None else squared + term
    scales = (float(GRAD_CLIP) / squared.sqrt().clamp_min(D20_EPS)).clamp(max=1.0)
    output = {}
    beta1, beta2 = hyper["beta1"], hyper["beta2"]
    sign = -1.0 if hyper["maximize"] else 1.0
    for name in names:
        shape = (len(scales),) + (1,) * (gradients[name].ndim - 1)
        gradient = sign * gradients[name].float() * scales.reshape(shape)
        item = state[name]
        step = item["step"] + 1
        moment = beta1 * item["exp_avg"].unsqueeze(0) + (1.0 - beta1) * gradient
        variance = (
            beta2 * item["exp_avg_sq"].unsqueeze(0)
            + (1.0 - beta2) * gradient.square()
        )
        if hyper["amsgrad"]:
            variance = torch.maximum(
                item["max_exp_avg_sq"].unsqueeze(0), variance
            )
        denominator = variance.sqrt() / math.sqrt(1.0 - beta2**step) + hyper["eps"]
        step_size = hyper["lr"] / (1.0 - beta1**step)
        decay = -hyper["lr"] * hyper["weight_decay"] * parameters[name]
        output[name] = decay.unsqueeze(0) - step_size * moment / denominator
    return output


def normalized_dots(query_matrix, train_matrix):
    raw = (train_matrix @ query_matrix.T).T
    query_norm = query_matrix.norm(dim=1).clamp_min(D20_EPS)
    train_norm = train_matrix.norm(dim=1).clamp_min(D20_EPS)
    return {
        "raw": raw,
        "query_l2": raw / query_norm[:, None],
        "train_l2": raw / train_norm[None, :],
        "query_train_l2": raw / (query_norm[:, None] * train_norm[None, :]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--direction-shard-index", type=int, required=True)
    parser.add_argument("--direction-shard-count", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--query-term-batch-size", type=int, default=128)
    args = parser.parse_args()
    if not 0 <= args.direction_shard_index < args.direction_shard_count:
        raise ValueError("invalid direction shard")
    if args.batch_size <= 0 or args.query_term_batch_size <= 0:
        raise ValueError("batch sizes must be positive")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    records = [record for record in manifest if record["family"] == args.family]
    records.sort(key=lambda item: int(item["query_id"]))
    query_ids = [int(record["query_id"]) for record in records]
    if not records:
        raise ValueError(f"no queries for family={args.family}")

    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    paths = model_paths(args.family)
    if len(paths) != 50:
        raise ValueError(f"expected 50 checkpoints, found {len(paths)}")
    schedule = base.make_linear_schedule(T, device=device)
    x_all, cond_all = preload_dataset(dataset, args.family, device)
    conditions = torch.cat(
        [cond_for(record, dataset, device) for record in records], dim=0
    )
    endpoints = torch.cat(
        [
            torch.from_numpy(np.load(Path(record["dir"]) / "final_state.npy")).to(
                device=device, dtype=torch.float32
            )
            for record in records
        ],
        dim=0,
    )
    trajectories = torch.stack(
        [
            torch.from_numpy(
                np.load(Path(record["dir"]) / "trajectory_xt.npy").copy()
            ).to(device=device, dtype=torch.float32)
            for record in records
        ],
        dim=0,
    )
    trajectory_t = [
        np.load(Path(record["dir"]) / "trajectory_t.npy") for record in records
    ]
    if len(trajectory_t[0]) != 100 or any(
        not np.array_equal(value, trajectory_t[0]) for value in trajectory_t
    ):
        raise ValueError("queries do not share one cached 100-t trajectory grid")

    trajectory_query_t = torch.tensor(
        trajectory_t[0], device=device, dtype=torch.long
    )
    endpoint_query_t = torch.tensor(
        D20_QUERY_TIMESTAMPS, device=device, dtype=torch.long
    )
    train_t = torch.tensor(D20_TRAIN_TIMESTAMPS, device=device, dtype=torch.long)
    query_count = len(records)
    timestamp_count = len(trajectory_query_t)
    directions = list(
        range(
            args.direction_shard_index,
            D20_DIRECTION_COUNT,
            args.direction_shard_count,
        )
    )
    root = d20_shard_root(
        args.family, args.direction_shard_index, args.direction_shard_count
    )
    done_path = root / "done.json"
    if done_path.is_file():
        print(f"[skip] {done_path}", flush=True)
        return

    expected_shape = (query_count, N_TRAIN)
    partial_paths = {
        mode: {
            variant: root / f"partial_{mode}_{variant}.npy"
            for variant in D20_VARIANTS
        }
        for mode in D20_QUERY_MODES
    }
    progress_path = root / "progress.json"
    completed = []
    all_paths = [path for values in partial_paths.values() for path in values.values()]
    if all(path.is_file() for path in all_paths) and progress_path.is_file():
        with open(progress_path) as handle:
            progress = json.load(handle)
        if int(progress["contract_version"]) != D20_CONTRACT_VERSION:
            raise ValueError("partial contract changed")
        if int(progress["batch_size"]) != args.batch_size:
            raise ValueError("partial batch size differs")
        if progress["query_ids"] != query_ids:
            raise ValueError("partial query IDs differ")
        completed = [int(value) for value in progress["completed_directions"]]
        scores = {
            mode: {
                variant: torch.from_numpy(np.load(path)).to(
                    device=device, dtype=torch.float64
                )
                for variant, path in values.items()
            }
            for mode, values in partial_paths.items()
        }
        print(f"[resume] directions={len(completed)}/{len(directions)}", flush=True)
    else:
        scores = {
            mode: {
                variant: torch.zeros(expected_shape, device=device, dtype=torch.float64)
                for variant in D20_VARIANTS
            }
            for mode in D20_QUERY_MODES
        }

    remaining = [value for value in directions if value not in set(completed)]
    started = time.perf_counter()
    print(
        f"[direction20 gpu={args.gpu}] family={args.family} queries={query_ids[0]}..{query_ids[-1]} "
        f"directions={len(directions)}/20 train_t=100(mean) query_t=100 "
        f"full_adamw=true projection=4096 batch={args.batch_size}",
        flush=True,
    )

    for local_direction, direction_index in enumerate(remaining, start=1):
        direction_started = time.perf_counter()
        accumulators = {
            mode: {
                variant: torch.zeros(
                    (query_count, timestamp_count, N_TRAIN),
                    device=device,
                    dtype=torch.float32,
                )
                for variant in D20_VARIANTS
            }
            for mode in D20_QUERY_MODES
        }

        for checkpoint_index in range(49):
            pair_started = time.perf_counter()
            model, _, checkpoint = build_model(paths[checkpoint_index], "raw", device)
            target, _, _ = build_model(paths[checkpoint_index + 1], "raw", device)
            named = dict(model.named_parameters())
            names = tuple(named)
            parameters = tuple(named.values())
            adam_state, adam_hyper = optimizer_state_by_name(
                checkpoint, names, device
            )
            specs = build_countsketch_specs(
                list(parameters),
                D20_PROJECTION_DIM,
                device=device,
                seed_parts=(
                    TRAIN_SEED,
                    "direction20_adamw_full_projection",
                    checkpoint_index,
                ),
            )
            generator = make_torch_generator(
                device,
                D20_NOISE_SEED,
                "direction20_checkpoint_noise",
                checkpoint_index,
                direction_index,
            )
            noise = torch.randn(
                endpoints.shape[1:],
                generator=generator,
                device=device,
                dtype=endpoints.dtype,
            )

            endpoint_bank = endpoints[:, None].expand(
                query_count, timestamp_count, *endpoints.shape[1:]
            ).reshape(query_count * timestamp_count, *endpoints.shape[1:])
            trajectory_timestep_bank = trajectory_query_t[None].expand(
                query_count, timestamp_count
            ).reshape(-1)
            endpoint_timestep_bank = endpoint_query_t[None].expand(
                query_count, timestamp_count
            ).reshape(-1)
            condition_bank = conditions[:, None].expand(
                query_count, timestamp_count, conditions.shape[-1]
            ).reshape(-1, conditions.shape[-1])
            query_inputs = {
                "trajectory": (
                    trajectories.reshape(
                        query_count * timestamp_count, *trajectories.shape[2:]
                    ),
                    trajectory_timestep_bank,
                ),
                "endpoint": (
                    base.q_sample(
                        endpoint_bank,
                        endpoint_timestep_bank,
                        noise.unsqueeze(0).expand_as(endpoint_bank),
                        schedule,
                    ),
                    endpoint_timestep_bank,
                ),
            }

            def query_scalar(parameter_dict, xt, timestep, condition, output_direction):
                prediction = functional_call(
                    model,
                    parameter_dict,
                    (xt.unsqueeze(0), timestep.reshape(1), condition.unsqueeze(0)),
                )
                return (prediction.squeeze(0) * output_direction).sum()

            query_grad_fn = vmap(grad(query_scalar), in_dims=(None, 0, 0, 0, 0))
            query_matrices = {}
            delta_ranges = {}
            for mode, (inputs, timestep_bank) in query_inputs.items():
                with torch.no_grad():
                    current = model(inputs, timestep_bank, condition_bank)
                    following = target(inputs, timestep_bank, condition_bank)
                    delta = following - current
                    delta_norm = delta.flatten(1).norm(dim=1)
                    output_direction = delta / delta_norm.clamp_min(D20_EPS).reshape(
                        -1, *([1] * (delta.ndim - 1))
                    )
                chunks = []
                for start in range(0, len(inputs), args.query_term_batch_size):
                    end = min(start + args.query_term_batch_size, len(inputs))
                    gradients = query_grad_fn(
                        named,
                        inputs[start:end],
                        timestep_bank[start:end],
                        condition_bank[start:end],
                        output_direction[start:end],
                    )
                    chunks.append(
                        _project_batched_grads(
                            gradients,
                            names,
                            specs,
                            D20_PROJECTION_DIM,
                            False,
                            1e-8,
                        )
                    )
                    del gradients
                query_matrices[mode] = torch.cat(chunks, dim=0)
                delta_ranges[mode] = (float(delta_norm.min()), float(delta_norm.max()))
                del current, following, delta, delta_norm, output_direction, chunks

            def mean100_loss(parameter_dict, x0, condition, aligned_noise):
                count = train_t.shape[0]
                x_bank = x0.unsqueeze(0).expand(count, *x0.shape)
                condition_local = condition.unsqueeze(0).expand(count, condition.shape[-1])
                noise_bank = aligned_noise.unsqueeze(0).expand_as(x_bank)
                xt = base.q_sample(x_bank, train_t, noise_bank, schedule)
                prediction = functional_call(
                    model, parameter_dict, (xt, train_t, condition_local)
                )
                return (prediction - noise_bank).square().mean()

            train_grad_fn = vmap(grad(mean100_loss), in_dims=(None, 0, 0, None))
            num_batches = math.ceil(N_TRAIN / args.batch_size)
            progress_every = max(1, num_batches // 10)
            for batch_position, start in enumerate(
                range(0, N_TRAIN, args.batch_size), start=1
            ):
                end = min(start + args.batch_size, N_TRAIN)
                gradients = train_grad_fn(
                    named, x_all[start:end], cond_all[start:end], noise
                )
                updates = adamw_full_batched(
                    gradients, names, named, adam_state, adam_hyper
                )
                train_matrix = _project_batched_grads(
                    updates,
                    names,
                    specs,
                    D20_PROJECTION_DIM,
                    False,
                    1e-8,
                ).detach()
                for mode in D20_QUERY_MODES:
                    variants = normalized_dots(query_matrices[mode], train_matrix)
                    for variant in D20_VARIANTS:
                        accumulators[mode][variant][:, :, start:end] += variants[
                            variant
                        ].reshape(query_count, timestamp_count, end - start)
                    del variants
                del gradients, updates, train_matrix
                if (
                    batch_position == 1
                    or batch_position % progress_every == 0
                    or batch_position == num_batches
                ):
                    print(
                        f"[direction20 gpu={args.gpu}] direction={direction_index+1}/20 "
                        f"pair={checkpoint_index+1}/49 batch={batch_position}/{num_batches}",
                        flush=True,
                    )

            print(
                f"[direction20 gpu={args.gpu}] direction={direction_index+1}/20 "
                f"pair={checkpoint_index+1}/49 elapsed={(time.perf_counter()-pair_started)/60:.1f}m "
                f"traj_delta=[{delta_ranges['trajectory'][0]:.2e},{delta_ranges['trajectory'][1]:.2e}] "
                f"endpoint_delta=[{delta_ranges['endpoint'][0]:.2e},{delta_ranges['endpoint'][1]:.2e}]",
                flush=True,
            )
            del model, target, checkpoint, named, names, parameters
            del adam_state, adam_hyper, specs, noise, endpoint_bank
            del trajectory_timestep_bank, endpoint_timestep_bank
            del timestep_bank, condition_bank, query_inputs, query_matrices
            del query_grad_fn, train_grad_fn, query_scalar, mean100_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        weight = 1.0 / float(D20_DIRECTION_COUNT * timestamp_count)
        for mode in D20_QUERY_MODES:
            for variant in D20_VARIANTS:
                scores[mode][variant] += weight * accumulators[mode][variant].square().sum(
                    dim=1
                ).to(torch.float64)
        completed.append(direction_index)
        completed.sort()
        for mode in D20_QUERY_MODES:
            for variant, path in partial_paths[mode].items():
                atomic_numpy(path, scores[mode][variant].cpu().numpy())
        atomic_json(
            progress_path,
            {
                "contract_version": D20_CONTRACT_VERSION,
                "family": args.family,
                "query_ids": query_ids,
                "batch_size": args.batch_size,
                "query_term_batch_size": args.query_term_batch_size,
                "completed_directions": completed,
            },
        )
        elapsed = time.perf_counter() - started
        eta = elapsed / local_direction * (len(remaining) - local_direction)
        print(
            f"[direction checkpoint gpu={args.gpu}] completed={len(completed)}/{len(directions)} "
            f"direction_elapsed={(time.perf_counter()-direction_started)/3600:.2f}h "
            f"eta={eta/3600:.2f}h",
            flush=True,
        )
        del accumulators
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for mode in D20_QUERY_MODES:
        for variant in D20_VARIANTS:
            atomic_numpy(root / f"{mode}_{variant}.npy", scores[mode][variant].cpu().numpy())
    atomic_json(
        done_path,
        {
            "contract_version": D20_CONTRACT_VERSION,
            "family": args.family,
            "query_ids": query_ids,
            "direction_indices": directions,
            "direction_count": D20_DIRECTION_COUNT,
            "train_timesteps": list(D20_TRAIN_TIMESTAMPS),
            "trajectory_query_timesteps": [int(value) for value in trajectory_t[0]],
            "endpoint_query_timesteps": list(D20_QUERY_TIMESTAMPS),
            "checkpoint_transitions": 49,
            "train_feature": "full saved-state AdamW update of mean100t aligned-direction loss",
            "query_modes": list(D20_QUERY_MODES),
            "contraction": "checkpoint-sum then square; mean over direction and timestamp",
            "projection_dim": D20_PROJECTION_DIM,
            "batch_size": args.batch_size,
        },
    )
    print(f"[done] {done_path}", flush=True)


if __name__ == "__main__":
    main()
