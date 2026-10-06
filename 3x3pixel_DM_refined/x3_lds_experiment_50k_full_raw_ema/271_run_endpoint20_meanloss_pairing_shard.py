"""Run endpoint-near per-t aligned vs mean-noise/mean-loss TracIn-DAS."""

import argparse
import json
import math
import os
import time
from importlib import import_module
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
from endpoint20_meanloss_pairing_config import *
from x3_endpoint_das_jax_logic_pytorch import build_countsketch_specs


pairing_worker = import_module("235_run_tracin_das_noise_pairing_ablation_shard")
optimizer_state_by_name = pairing_worker.optimizer_state_by_name
adamw_full_batched = pairing_worker.adamw_full_batched


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def atomic_npz(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez(handle, **values)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--checkpoint-shard-index", type=int, required=True)
    parser.add_argument("--checkpoint-shard-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-microbatch-size", type=int, default=4)
    parser.add_argument("--train-t-chunk-size", type=int, default=4)
    parser.add_argument("--query-term-batch-size", type=int, default=64)
    args = parser.parse_args()
    if not 0 <= args.checkpoint_shard_index < args.checkpoint_shard_count:
        raise ValueError("invalid checkpoint shard")
    if min(
        args.batch_size,
        args.grad_microbatch_size,
        args.train_t_chunk_size,
        args.query_term_batch_size,
    ) <= 0:
        raise ValueError("batch sizes must be positive")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    with open(QUERY_DIR / "manifest.json") as handle:
        by_id = {int(item["query_id"]): item for item in json.load(handle)}
    records = [by_id[query_id] for query_id in E20_QUERY_IDS]
    if any(record["family"] != "prompted" for record in records):
        raise ValueError("q00-q09 must be prompted queries")

    endpoints = torch.cat(
        [
            torch.from_numpy(
                np.load(QUERY_DIR / f"q{query_id:02d}" / "final_state.npy")
            )
            for query_id in E20_QUERY_IDS
        ],
        dim=0,
    ).to(device=device, dtype=torch.float32)
    trajectories = [
        np.load(Path(record["dir"]) / "trajectory_xt.npy", mmap_mode="r")
        for record in records
    ]
    trajectory_times = [
        np.load(Path(record["dir"]) / "trajectory_t.npy")
        for record in records
    ]
    if any(not np.array_equal(trajectory_times[0], value) for value in trajectory_times):
        raise ValueError("query trajectory timestamp grids differ")
    trajectory_times = np.asarray(trajectory_times[0], dtype=np.int64)

    paths = model_paths("prompted")
    bootstrap, dataset, _ = build_model(paths[0], "raw", device)
    x_all, cond_all = preload_dataset(dataset, "prompted", device)
    conditions = torch.cat(
        [cond_for(record, dataset, device) for record in records], dim=0
    )
    del bootstrap
    schedule = base.make_linear_schedule(T, device=device)

    timesteps = torch.tensor(E20_TIMESTEPS, device=device, dtype=torch.long)
    reference_by_t = []
    resolved_trajectory_timesteps = []
    for timestep in E20_TIMESTEPS:
        nearest = int(np.argmin(np.abs(trajectory_times - int(timestep))))
        resolved_trajectory_timesteps.append(int(trajectory_times[nearest]))
        reference_by_t.append(
            torch.cat(
                [
                    torch.from_numpy(np.asarray(trajectory[nearest]).copy())
                    for trajectory in trajectories
                ],
                dim=0,
            )
        )
    reference_states = torch.stack(reference_by_t, dim=1).to(
        device=device, dtype=torch.float32
    )
    a = schedule.sqrt_alpha_bars[timesteps].reshape(1, -1, 1, 1, 1)
    b = schedule.sqrt_one_minus_alpha_bars[timesteps].reshape(1, -1, 1, 1, 1)
    implied_noises = (reference_states - a * endpoints[:, None]) / b.clamp_min(
        NPA_EPS
    )
    mean_noises = implied_noises.mean(dim=1)

    selected_pairs = list(
        NPA_CHECKPOINT_PAIRS[
            args.checkpoint_shard_index :: args.checkpoint_shard_count
        ]
    )
    root = e20_shard_root(
        args.checkpoint_shard_index, args.checkpoint_shard_count
    )
    done_path = root / "done.json"
    if done_path.is_file():
        print(f"[skip] {done_path}", flush=True)
        return

    score_shape = (len(E20_QUERY_IDS), N_TRAIN)
    response_shape = (len(E20_QUERY_IDS), len(E20_TIMESTEPS), N_TRAIN)
    linear = {
        mode: torch.zeros(score_shape, device=device, dtype=torch.float64)
        for mode in E20_MODES
    }
    termwise = {
        mode: torch.zeros(score_shape, device=device, dtype=torch.float64)
        for mode in E20_MODES
    }
    timestamp_response = {
        mode: torch.zeros(response_shape, device=device, dtype=torch.float32)
        for mode in E20_MODES
    }
    partial_path = root / "partial_scores.npz"
    progress_path = root / "progress.json"
    completed = []
    if partial_path.is_file() and progress_path.is_file():
        with open(progress_path) as handle:
            progress = json.load(handle)
        if int(progress["version"]) != E20_VERSION:
            raise ValueError("partial version differs")
        completed = [int(value) for value in progress["completed_checkpoint_pairs"]]
        with np.load(partial_path) as partial:
            for mode in E20_MODES:
                linear[mode].copy_(torch.from_numpy(partial[f"{mode}__linear"]).to(device))
                termwise[mode].copy_(
                    torch.from_numpy(partial[f"{mode}__termwise_squared"]).to(device)
                )
                timestamp_response[mode].copy_(
                    torch.from_numpy(partial[f"{mode}__timestamp_response"]).to(device)
                )
        print(f"[resume] checkpoint pairs={len(completed)}/{len(selected_pairs)}", flush=True)

    remaining_pairs = [value for value in selected_pairs if value not in set(completed)]
    started = time.perf_counter()
    print(
        f"[endpoint20 gpu={args.gpu}] queries={list(E20_QUERY_IDS)} "
        f"pairs={selected_pairs} timestamps={list(E20_TIMESTEPS)} "
        f"resolved_trajectory_t={resolved_trajectory_timesteps} "
        f"modes={list(E20_MODES)} full_adamw=true projected4096",
        flush=True,
    )

    for pair_position, checkpoint_index in enumerate(remaining_pairs, start=1):
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
            NPA_PROJECTION_DIM,
            device=device,
            seed_parts=(TRAIN_SEED, "endpoint20_meanloss_projection", checkpoint_index),
        )

        query_count = len(E20_QUERY_IDS)
        timestamp_count = len(E20_TIMESTEPS)
        query_xt = reference_states.reshape(
            query_count * timestamp_count, *reference_states.shape[2:]
        )
        query_t = timesteps[None].expand(query_count, -1).reshape(-1)
        query_condition = conditions[:, None].expand(
            query_count, timestamp_count, conditions.shape[-1]
        ).reshape(-1, conditions.shape[-1])
        with torch.no_grad():
            current = model(query_xt, query_t, query_condition)
            following = target(query_xt, query_t, query_condition)
            delta = following - current
            delta_norm = delta.flatten(1).norm(dim=1)
            output_direction = delta / delta_norm.clamp_min(NPA_EPS).reshape(
                -1, *([1] * (delta.ndim - 1))
            )

        def query_scalar(parameter_dict, xt, t_value, condition, direction):
            prediction = functional_call(
                model,
                parameter_dict,
                (xt.unsqueeze(0), t_value.reshape(1), condition.unsqueeze(0)),
            )
            return (prediction.squeeze(0) * direction).sum()

        query_grad_fn = vmap(grad(query_scalar), in_dims=(None, 0, 0, 0, 0))
        query_chunks = []
        for start in range(0, len(query_xt), args.query_term_batch_size):
            end = min(start + args.query_term_batch_size, len(query_xt))
            gradients = query_grad_fn(
                named,
                query_xt[start:end],
                query_t[start:end],
                query_condition[start:end],
                output_direction[start:end],
            )
            query_chunks.append(
                _project_batched_grads(
                    gradients,
                    names,
                    specs,
                    NPA_PROJECTION_DIM,
                    False,
                    NPA_EPS,
                )
            )
            del gradients
        query_matrix = torch.cat(query_chunks).reshape(
            query_count, timestamp_count, NPA_PROJECTION_DIM
        ).detach()

        def single_loss(parameter_dict, x0, condition, t_value, noise):
            xt = base.q_sample(
                x0.unsqueeze(0),
                t_value.reshape(1),
                noise.unsqueeze(0),
                schedule,
            )
            prediction = functional_call(
                model,
                parameter_dict,
                (xt, t_value.reshape(1), condition.unsqueeze(0)),
            )
            return (prediction - noise.unsqueeze(0)).square().mean()

        def mean_loss(parameter_dict, x0, condition, noise):
            count = len(E20_TIMESTEPS)
            xb = x0.unsqueeze(0).expand(count, *x0.shape)
            nb = noise.unsqueeze(0).expand(count, *noise.shape)
            cb = condition.unsqueeze(0).expand(count, condition.shape[-1])
            xt = base.q_sample(xb, timesteps, nb, schedule)
            prediction = functional_call(model, parameter_dict, (xt, timesteps, cb))
            return (prediction - nb).square().mean()

        single_grad_fn = vmap(grad(single_loss), in_dims=(None, 0, 0, 0, 0))
        mean_grad_fn = vmap(grad(mean_loss), in_dims=(None, 0, 0, 0))
        num_batches = math.ceil(N_TRAIN / args.batch_size)
        progress_every = max(1, num_batches // 5)
        for batch_position, outer_start in enumerate(
            range(0, N_TRAIN, args.batch_size), start=1
        ):
            outer_end = min(outer_start + args.batch_size, N_TRAIN)
            for micro_start in range(
                outer_start, outer_end, args.grad_microbatch_size
            ):
                micro_end = min(micro_start + args.grad_microbatch_size, outer_end)
                count = micro_end - micro_start
                x_micro = x_all[micro_start:micro_end]
                c_micro = cond_all[micro_start:micro_end]
                for query_position in range(query_count):
                    # Mode 1: each t uses its own trajectory-implied noise.
                    dots = torch.empty(
                        (timestamp_count, count),
                        device=device,
                        dtype=torch.float32,
                    )
                    for t_start in range(
                        0, timestamp_count, args.train_t_chunk_size
                    ):
                        t_end = min(
                            t_start + args.train_t_chunk_size, timestamp_count
                        )
                        t_count = t_end - t_start
                        xb = x_micro[:, None].expand(
                            count, t_count, *x_micro.shape[1:]
                        ).reshape(-1, *x_micro.shape[1:])
                        cb = c_micro[:, None].expand(
                            count, t_count, c_micro.shape[-1]
                        ).reshape(-1, c_micro.shape[-1])
                        tb = timesteps[t_start:t_end][None].expand(
                            count, -1
                        ).reshape(-1)
                        nb = implied_noises[
                            query_position, t_start:t_end
                        ][None].expand(
                            count,
                            t_count,
                            *implied_noises.shape[2:],
                        ).reshape(-1, *implied_noises.shape[2:])
                        gradients = single_grad_fn(named, xb, cb, tb, nb)
                        updates = adamw_full_batched(
                            gradients, names, named, adam_state, adam_hyper
                        )
                        train_matrix = _project_batched_grads(
                            updates,
                            names,
                            specs,
                            NPA_PROJECTION_DIM,
                            False,
                            NPA_EPS,
                        ).reshape(
                            count, t_count, NPA_PROJECTION_DIM
                        ).detach()
                        dots[t_start:t_end] = torch.einsum(
                            "tp,btp->tb",
                            query_matrix[query_position, t_start:t_end],
                            train_matrix,
                        )
                        del xb, cb, tb, nb, gradients, updates, train_matrix
                    mode = "per_timestamp_aligned"
                    linear[mode][query_position, micro_start:micro_end] += (
                        dots.mean(dim=0).double()
                    )
                    termwise[mode][query_position, micro_start:micro_end] += (
                        dots.square().mean(dim=0).double()
                    )
                    timestamp_response[mode][
                        query_position, :, micro_start:micro_end
                    ] += dots
                    del dots

                    # Mode 2: one mean-noise, 20-t mean loss gradient, paired
                    # with every timestamp-specific query term.
                    nb_mean = mean_noises[query_position][None].expand(
                        count, *mean_noises.shape[1:]
                    )
                    gradients = mean_grad_fn(
                        named, x_micro, c_micro, nb_mean
                    )
                    updates = adamw_full_batched(
                        gradients, names, named, adam_state, adam_hyper
                    )
                    train_matrix = _project_batched_grads(
                        updates,
                        names,
                        specs,
                        NPA_PROJECTION_DIM,
                        False,
                        NPA_EPS,
                    ).detach()
                    dots = torch.einsum(
                        "tp,bp->tb", query_matrix[query_position], train_matrix
                    )
                    mode = "mean_noise_mean_loss"
                    linear[mode][query_position, micro_start:micro_end] += (
                        dots.mean(dim=0).double()
                    )
                    termwise[mode][query_position, micro_start:micro_end] += (
                        dots.square().mean(dim=0).double()
                    )
                    timestamp_response[mode][
                        query_position, :, micro_start:micro_end
                    ] += dots
                    del nb_mean, gradients, updates, train_matrix, dots

            if (
                batch_position == 1
                or batch_position % progress_every == 0
                or batch_position == num_batches
            ):
                print(
                    f"[endpoint20 gpu={args.gpu}] pair="
                    f"{pair_position}/{len(remaining_pairs)} "
                    f"checkpoint_index={checkpoint_index} "
                    f"batch={batch_position}/{num_batches}",
                    flush=True,
                )

        completed.append(checkpoint_index)
        completed.sort()
        partial = {}
        for mode in E20_MODES:
            partial[f"{mode}__linear"] = linear[mode].cpu().numpy()
            partial[f"{mode}__termwise_squared"] = termwise[mode].cpu().numpy()
            partial[f"{mode}__timestamp_response"] = (
                timestamp_response[mode].cpu().numpy()
            )
        atomic_npz(partial_path, partial)
        atomic_json(
            progress_path,
            {
                "version": E20_VERSION,
                "completed_checkpoint_pairs": completed,
                "selected_checkpoint_pairs": selected_pairs,
            },
        )
        elapsed = time.perf_counter() - pair_started
        total_elapsed = time.perf_counter() - started
        completed_count = pair_position
        eta = total_elapsed / completed_count * (
            len(remaining_pairs) - completed_count
        )
        print(
            f"[checkpoint] pair={pair_position}/{len(remaining_pairs)} "
            f"elapsed={elapsed/60:.1f}m eta={eta/3600:.2f}h",
            flush=True,
        )
        del model, target, checkpoint, named, names, parameters
        del adam_state, adam_hyper, specs, current, following, delta
        del delta_norm, output_direction, query_chunks, query_matrix
        del query_grad_fn, single_grad_fn, mean_grad_fn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    atomic_json(
        done_path,
        {
            "version": E20_VERSION,
            "query_ids": list(E20_QUERY_IDS),
            "checkpoint_pairs": selected_pairs,
            "all_checkpoint_pairs": list(NPA_CHECKPOINT_PAIRS),
            "timestamp_positions": list(E20_TIMESTAMP_POSITIONS),
            "timesteps": list(E20_TIMESTEPS),
            "resolved_trajectory_timesteps": resolved_trajectory_timesteps,
            "modes": list(E20_MODES),
            "parameter_source": "raw checkpoint pair",
            "optimizer_transform": "full AdamW",
            "projection_dim": NPA_PROJECTION_DIM,
            "query_direction": "normalized next-checkpoint predicted-noise delta",
        },
    )
    print(f"[done] {root}", flush=True)


if __name__ == "__main__":
    main()
