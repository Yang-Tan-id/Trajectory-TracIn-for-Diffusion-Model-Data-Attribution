"""Run query-dependent trajectory-bridge aligned TracIn-DAS."""

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
from trajectory_bridge_alignment_config import *
from x3_endpoint_das_jax_logic_pytorch import (
    build_countsketch_specs,
    make_torch_generator,
)


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


def score_key(contraction, group):
    return "__".join((contraction, group))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--family", choices=("prompted", "unprompted"), default="prompted")
    parser.add_argument("--timestamp-shard-index", type=int, required=True)
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-microbatch-size", type=int, default=4)
    parser.add_argument("--query-term-batch-size", type=int, default=128)
    args = parser.parse_args()
    if not 0 <= args.timestamp_shard_index < args.timestamp_shard_count:
        raise ValueError("invalid timestamp shard")
    if min(args.batch_size, args.grad_microbatch_size, args.query_term_batch_size) <= 0:
        raise ValueError("batch sizes must be positive")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    with open(QUERY_DIR / "manifest.json") as handle:
        by_id = {int(item["query_id"]): item for item in json.load(handle)}
    records = [by_id[query_id] for query_id in TBA_QUERY_IDS]
    if any(record["family"] != args.family for record in records):
        raise ValueError(
            f"trajectory-bridge query family mismatch: expected {args.family}"
        )

    endpoints = torch.cat(
        [
            torch.from_numpy(np.load(QUERY_DIR / f"q{query_id:02d}" / "final_state.npy"))
            for query_id in TBA_QUERY_IDS
        ],
        dim=0,
    ).to(device=device, dtype=torch.float32)
    trajectories = [
        np.load(Path(record["dir"]) / "trajectory_xt.npy", mmap_mode="r")
        for record in records
    ]
    trajectory_times = [
        np.load(Path(record["dir"]) / "trajectory_t.npy") for record in records
    ]
    if any(not np.array_equal(trajectory_times[0], values) for values in trajectory_times):
        raise ValueError("query trajectory timestamp grids differ")
    trajectory_times = np.asarray(trajectory_times[0], dtype=np.int64)

    paths = model_paths(args.family)
    bootstrap, dataset, _ = build_model(paths[0], "raw", device)
    x_all, cond_all = preload_dataset(dataset, args.family, device)
    conditions = torch.cat(
        [cond_for(record, dataset, device) for record in records], dim=0
    )
    del bootstrap
    schedule = base.make_linear_schedule(T, device=device)

    random_generator = make_torch_generator(
        device, NPA_NOISE_SEED, "trajectory_bridge_random_origins"
    )
    random_origins = torch.randn(
        (len(TBA_QUERY_IDS), TBA_DIRECTION_COUNT, *endpoints.shape[1:]),
        generator=random_generator,
        device=device,
        dtype=endpoints.dtype,
    )

    selected = list(
        NPA_TIMESTAMP_INDICES[
            args.timestamp_shard_index :: args.timestamp_shard_count
        ]
    )
    root = tba_shard_root(args.timestamp_shard_index, args.timestamp_shard_count)
    done_path = root / "done.json"
    if done_path.is_file():
        print(f"[skip] {done_path}", flush=True)
        return
    expected_shape = (len(TBA_QUERY_IDS), N_TRAIN)
    scores = {
        score_key(contraction, group): torch.zeros(
            expected_shape, device=device, dtype=torch.float64
        )
        for contraction in TBA_CONTRACTIONS
        for group in NPA_TIMESTAMP_GROUPS
    }
    partial_path = root / "partial_scores.npz"
    progress_path = root / "progress.json"
    completed = []
    if partial_path.is_file() and progress_path.is_file():
        with open(progress_path) as handle:
            progress = json.load(handle)
        if int(progress["version"]) != TBA_VERSION:
            raise ValueError("partial version differs")
        completed = [int(value) for value in progress["completed_timestamp_indices"]]
        with np.load(partial_path) as partial:
            for key in scores:
                scores[key].copy_(torch.from_numpy(partial[key]).to(device))
        print(f"[resume] timestamps={len(completed)}/{len(selected)}", flush=True)

    remaining = [value for value in selected if value not in set(completed)]
    total_terms = len(remaining) * len(NPA_CHECKPOINT_PAIRS)
    completed_terms = 0
    started = time.perf_counter()
    print(
        f"[trajectory-bridge gpu={args.gpu}] queries={list(TBA_QUERY_IDS)} "
        f"timestamps={len(selected)}/{len(DAS_TIMESTEPS)} "
        f"pairs={len(NPA_CHECKPOINT_PAIRS)} directions={TBA_DIRECTION_COUNT} "
        f"query-dependent aligned loss full_adamw=true "
        f"projection=4096 batch={args.batch_size} microbatch={args.grad_microbatch_size}",
        flush=True,
    )

    for local_timestamp, timestamp_index in enumerate(remaining, start=1):
        timestep = int(DAS_TIMESTEPS[timestamp_index])
        nearest = int(np.argmin(np.abs(trajectory_times - timestep)))
        if int(trajectory_times[nearest]) != timestep:
            print(
                f"[trajectory-bridge] DAS t={timestep} uses nearest trajectory "
                f"t={int(trajectory_times[nearest])}",
                flush=True,
            )
        reference_states = torch.cat(
            [
                torch.from_numpy(np.asarray(trajectory[nearest]).copy())
                for trajectory in trajectories
            ],
            dim=0,
        ).to(device=device, dtype=torch.float32)
        a = schedule.sqrt_alpha_bars[timestep]
        b = schedule.sqrt_one_minus_alpha_bars[timestep]
        implied_noise = (reference_states - a * endpoints) / b.clamp_min(NPA_EPS)
        bridge_weight = float(timestep) / float(T - 1)
        random_norm = random_origins.flatten(2).norm(dim=2, keepdim=True)
        implied_bank = implied_noise[:, None].expand_as(random_origins)
        implied_norm = implied_bank.flatten(2).norm(dim=2, keepdim=True)
        random_unit = random_origins / random_norm.clamp_min(NPA_EPS).reshape(
            *random_norm.shape, 1, 1
        )
        implied_unit = implied_bank / implied_norm.clamp_min(NPA_EPS).reshape(
            *implied_norm.shape, 1, 1
        )
        mixed_unit = (1.0 - bridge_weight) * random_unit + bridge_weight * implied_unit
        mixed_norm = mixed_unit.flatten(2).norm(dim=2, keepdim=True)
        mixed_unit = mixed_unit / mixed_norm.clamp_min(NPA_EPS).reshape(
            *mixed_norm.shape, 1, 1
        )
        bridge_radius = (
            (1.0 - bridge_weight) * random_norm + bridge_weight * implied_norm
        )
        path_noises = mixed_unit * bridge_radius.reshape(
            *bridge_radius.shape, 1, 1
        )
        if TBA_PURE_IMPLIED_NOISE:
            # Exact query-dependent direction inferred from endpoint -> x_t.
            # Unlike the bridge experiment, no random direction is mixed in.
            path_noises = implied_bank
        query_count = len(TBA_QUERY_IDS)
        endpoint_bank = endpoints[:, None].expand(
            query_count, TBA_DIRECTION_COUNT, *endpoints.shape[1:]
        ).reshape(-1, *endpoints.shape[1:])
        query_noise_bank = path_noises.reshape(-1, *endpoints.shape[1:])
        query_t = torch.full(
            (query_count * TBA_DIRECTION_COUNT,), timestep,
            device=device, dtype=torch.long,
        )
        query_condition_bank = conditions[:, None].expand(
            query_count, TBA_DIRECTION_COUNT, conditions.shape[-1]
        ).reshape(-1, conditions.shape[-1])
        query_xt = base.q_sample(endpoint_bank, query_t, query_noise_bank, schedule)
        timestamp_accumulator = torch.zeros(
            (query_count, TBA_DIRECTION_COUNT, N_TRAIN),
            device=device,
            dtype=torch.float32,
        )
        groups = [
            group for group, indices in NPA_TIMESTAMP_GROUPS.items()
            if timestamp_index in indices
        ]

        for checkpoint_position, checkpoint_index in enumerate(
            NPA_CHECKPOINT_PAIRS, start=1
        ):
            term_started = time.perf_counter()
            model, _, checkpoint = build_model(paths[checkpoint_index], "raw", device)
            target, _, _ = build_model(paths[checkpoint_index + 1], "raw", device)
            named = dict(model.named_parameters())
            names = tuple(named)
            parameters = tuple(named.values())
            adam_state, adam_hyper = optimizer_state_by_name(
                checkpoint, names, device
            )
            specs = build_countsketch_specs(
                list(parameters), NPA_PROJECTION_DIM, device=device,
                seed_parts=(TRAIN_SEED, "trajectory_bridge_projection", checkpoint_index),
            )
            with torch.no_grad():
                current = model(query_xt, query_t, query_condition_bank)
                following = target(query_xt, query_t, query_condition_bank)
                delta = following - current
                delta_norm = delta.flatten(1).norm(dim=1)
                output_direction = delta / delta_norm.clamp_min(NPA_EPS).reshape(
                    -1, *([1] * (delta.ndim - 1))
                )

            def query_scalar(parameter_dict, xt, t_value, condition, direction):
                prediction = functional_call(
                    model, parameter_dict,
                    (xt.unsqueeze(0), t_value.reshape(1), condition.unsqueeze(0)),
                )
                return (prediction.squeeze(0) * direction).sum()

            query_grad_fn = vmap(grad(query_scalar), in_dims=(None, 0, 0, 0, 0))
            query_chunks = []
            for start in range(0, len(query_xt), args.query_term_batch_size):
                end = min(start + args.query_term_batch_size, len(query_xt))
                gradients = query_grad_fn(
                    named, query_xt[start:end], query_t[start:end],
                    query_condition_bank[start:end], output_direction[start:end],
                )
                query_chunks.append(
                    _project_batched_grads(
                        gradients, names, specs, NPA_PROJECTION_DIM, False, NPA_EPS
                    )
                )
                del gradients
            query_matrix = torch.cat(query_chunks).reshape(
                query_count, TBA_DIRECTION_COUNT, NPA_PROJECTION_DIM
            ).detach()

            independent_train_noises = None
            if TBA_TRAIN_NOISE_MODE == "independent":
                independent_generator = make_torch_generator(
                    device,
                    NPA_NOISE_SEED,
                    "trajectory_implied_independent_train_noise",
                    args.family,
                    checkpoint_index,
                    timestamp_index,
                )
                independent_train_noises = torch.randn(
                    (TBA_DIRECTION_COUNT, *endpoints.shape[1:]),
                    generator=independent_generator,
                    device=device,
                    dtype=endpoints.dtype,
                )
            elif TBA_TRAIN_NOISE_MODE != "aligned":
                raise ValueError(
                    f"unknown trajectory train-noise mode: {TBA_TRAIN_NOISE_MODE}"
                )

            def train_loss(parameter_dict, x0, condition, t_value, noise):
                xt = base.q_sample(
                    x0.unsqueeze(0), t_value.reshape(1), noise.unsqueeze(0), schedule
                )
                prediction = functional_call(
                    model, parameter_dict,
                    (xt, t_value.reshape(1), condition.unsqueeze(0)),
                )
                return (prediction - noise.unsqueeze(0)).square().mean()

            train_grad_fn = vmap(grad(train_loss), in_dims=(None, 0, 0, 0, 0))
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
                    for query_position in range(query_count):
                        xb = x_all[micro_start:micro_end, None].expand(
                            count, TBA_DIRECTION_COUNT, *x_all.shape[1:]
                        ).reshape(-1, *x_all.shape[1:])
                        cb = cond_all[micro_start:micro_end, None].expand(
                            count, TBA_DIRECTION_COUNT, cond_all.shape[-1]
                        ).reshape(-1, cond_all.shape[-1])
                        tb = torch.full(
                            (count * TBA_DIRECTION_COUNT,), timestep,
                            device=device, dtype=torch.long,
                        )
                        noise_bank = (
                            path_noises[query_position]
                            if TBA_TRAIN_NOISE_MODE == "aligned"
                            else independent_train_noises
                        )
                        nb = noise_bank[None].expand(
                            count, TBA_DIRECTION_COUNT, *endpoints.shape[1:]
                        ).reshape(-1, *endpoints.shape[1:])
                        gradients = train_grad_fn(named, xb, cb, tb, nb)
                        updates = adamw_full_batched(
                            gradients, names, named, adam_state, adam_hyper
                        )
                        train_matrix = _project_batched_grads(
                            updates, names, specs, NPA_PROJECTION_DIM, False, NPA_EPS
                        ).reshape(
                            count, TBA_DIRECTION_COUNT, NPA_PROJECTION_DIM
                        ).detach()
                        dots = torch.einsum(
                            "mp,bmp->mb", query_matrix[query_position], train_matrix
                        )
                        timestamp_accumulator[
                            query_position, :, micro_start:micro_end
                        ] += dots
                        for group in groups:
                            weight = 1.0 / len(NPA_TIMESTAMP_GROUPS[group])
                            scores[score_key("linear", group)][
                                query_position, micro_start:micro_end
                            ] += weight * dots.mean(dim=0).double()
                            scores[score_key("termwise_squared", group)][
                                query_position, micro_start:micro_end
                            ] += weight * dots.square().mean(dim=0).double()
                        del gradients, updates, train_matrix, dots
                if (
                    batch_position == 1
                    or batch_position % progress_every == 0
                    or batch_position == num_batches
                ):
                    print(
                        f"[trajectory-bridge gpu={args.gpu}] "
                        f"timestamp={local_timestamp}/{len(remaining)} t={timestep} "
                        f"pair={checkpoint_position}/{len(NPA_CHECKPOINT_PAIRS)} "
                        f"batch={batch_position}/{num_batches}",
                        flush=True,
                    )

            completed_terms += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / completed_terms * (total_terms - completed_terms)
            print(
                f"[trajectory-bridge gpu={args.gpu}] term={completed_terms}/{total_terms} "
                f"bridge={bridge_weight:.3f} "
                f"noise_norm=[{float(path_noises.flatten(2).norm(dim=2).min()):.3f},"
                f"{float(path_noises.flatten(2).norm(dim=2).max()):.3f}] "
                f"term_elapsed={(time.perf_counter()-term_started)/60:.1f}m "
                f"eta={eta/3600:.2f}h",
                flush=True,
            )
            del model, target, checkpoint, named, names, parameters
            del adam_state, adam_hyper, specs, current, following, delta
            del delta_norm, output_direction, query_chunks, query_matrix
            del query_grad_fn, train_grad_fn, query_scalar, train_loss
            del independent_train_noises
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        reduced = timestamp_accumulator.square().mean(dim=1).double()
        for group in groups:
            weight = 1.0 / len(NPA_TIMESTAMP_GROUPS[group])
            scores[score_key("timestamp_sum_squared", group)] += weight * reduced
        completed.append(timestamp_index)
        completed.sort()
        atomic_npz(
            partial_path, {key: value.cpu().numpy() for key, value in scores.items()}
        )
        atomic_json(
            progress_path,
            {
                "version": TBA_VERSION,
                "completed_timestamp_indices": completed,
                "selected_timestamp_indices": selected,
            },
        )
        del reference_states, implied_noise, implied_bank, path_noises
        del random_norm, implied_norm, random_unit, implied_unit, mixed_unit
        del mixed_norm, bridge_radius, endpoint_bank
        del query_noise_bank, query_t, query_condition_bank, query_xt
        del timestamp_accumulator, reduced
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    atomic_json(
        done_path,
        {
            "version": TBA_VERSION,
            "query_ids": list(TBA_QUERY_IDS),
            "timestamp_indices": selected,
            "checkpoint_pairs": list(NPA_CHECKPOINT_PAIRS),
            "direction_count": TBA_DIRECTION_COUNT,
            "noise_path": (
                "pure endpoint-to-reference-state implied noise"
                if TBA_PURE_IMPLIED_NOISE
                else "linear bridge from fixed random origins to trajectory-implied noise"
            ),
            "train_noise": (
                "exactly aligned to each query path direction"
                if TBA_TRAIN_NOISE_MODE == "aligned"
                else "independent Gaussian noise per checkpoint/timestamp"
            ),
            "train_noise_mode": TBA_TRAIN_NOISE_MODE,
        },
    )
    print(f"[done] {root}", flush=True)


if __name__ == "__main__":
    main()
