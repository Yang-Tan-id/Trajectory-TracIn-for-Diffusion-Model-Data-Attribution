"""Direction-aligned TracIn-DAS with an exact all-1000-t train loss.

For checkpoint transition c -> c+1 and Gaussian direction r, the train
feature for datapoint i is

    grad_theta mean_{t=0..999} ||eps_theta(x_i,t,eps_r) - eps_r||^2.

The same eps_r pollutes every query endpoint at the 100 evaluation
timestamps.  The query scalar is the current predicted noise projected onto
the normalized current-to-next-checkpoint predicted-noise delta direction.
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
    tracin_lr_weight,
)
from dataset_loader import ColorGridDataset
from direction_integrated_tracin_das_config import *
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--direction-shard-index", type=int, required=True)
    parser.add_argument("--direction-shard-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-t-chunk-size", type=int, default=50)
    parser.add_argument(
        "--train-t-count", type=int, default=DITD_DEFAULT_TRAIN_T_COUNT
    )
    parser.add_argument("--query-term-batch-size", type=int, default=100)
    parser.add_argument("--direction-count", type=int, default=DITD_DIRECTION_COUNT)
    args = parser.parse_args()
    if not 0 <= args.direction_shard_index < args.direction_shard_count:
        raise ValueError("invalid direction shard")
    if args.batch_size <= 0 or args.train_t_chunk_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.query_term_batch_size <= 0 or args.direction_count <= 0:
        raise ValueError("query batch and direction count must be positive")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    records = [by_id[qid] for qid in DITD_QUERY_IDS]
    if any(record["family"] != DITD_FAMILY for record in records):
        raise ValueError("q00-q09 must be prompted queries")

    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    paths = model_paths(DITD_FAMILY)
    if len(paths) != 50:
        raise ValueError(f"expected 50 checkpoints, found {len(paths)}")
    schedule = base.make_linear_schedule(T, device=device)
    x_all, cond_all = preload_dataset(dataset, DITD_FAMILY, device)
    endpoints = torch.cat(
        [
            torch.from_numpy(np.load(Path(record["dir"]) / "final_state.npy")).to(
                device=device, dtype=torch.float32
            )
            for record in records
        ]
    )
    conditions = torch.cat(
        [cond_for(record, dataset, device) for record in records], dim=0
    )
    query_t = torch.tensor(
        DITD_QUERY_TIMESTEPS, device=device, dtype=torch.long
    )
    train_t_values = ditd_train_timesteps(args.train_t_count)
    train_t = torch.tensor(train_t_values, device=device, dtype=torch.long)
    train_t_count = len(train_t_values)
    methods = ditd_methods(train_t_count)
    query_count = len(records)
    query_timestamp_count = len(query_t)
    selected_directions = list(
        range(
            args.direction_shard_index,
            args.direction_count,
            args.direction_shard_count,
        )
    )
    shard_root = ditd_shard_root(
        args.direction_shard_index, args.direction_shard_count, train_t_count
    )
    done_path = shard_root / "done.json"
    if done_path.is_file():
        print(f"[skip] {done_path}", flush=True)
        return

    expected_shape = (query_count, N_TRAIN)
    partial_paths = {
        contraction: shard_root / f"partial_{contraction}.npy"
        for contraction in methods
    }
    progress_path = shard_root / "progress.json"
    completed_directions = []
    if all(path.is_file() for path in partial_paths.values()) and progress_path.is_file():
        with open(progress_path) as handle:
            progress = json.load(handle)
        expected = {
            "contract_version": DITD_CONTRACT_VERSION,
            "batch_size": args.batch_size,
            "train_t_chunk_size": args.train_t_chunk_size,
            "train_t_count": train_t_count,
            "query_term_batch_size": args.query_term_batch_size,
            "direction_count": args.direction_count,
        }
        for key, value in expected.items():
            if int(progress[key]) != int(value):
                raise ValueError(f"partial shard {key} differs")
        completed_directions = [int(value) for value in progress["completed_directions"]]
        scores = {
            contraction: torch.from_numpy(np.load(path)).to(
                device=device, dtype=torch.float64
            )
            for contraction, path in partial_paths.items()
        }
        if any(value.shape != expected_shape for value in scores.values()):
            raise ValueError("partial score shape mismatch")
        print(
            f"[resume] directions={len(completed_directions)}/{len(selected_directions)}",
            flush=True,
        )
    else:
        scores = {
            contraction: torch.zeros(
                expected_shape, device=device, dtype=torch.float64
            )
            for contraction in methods
        }

    remaining = [
        value for value in selected_directions if value not in set(completed_directions)
    ]
    started = time.perf_counter()
    print(
        f"[direction-integrated gpu={args.gpu}] directions={len(selected_directions)} "
        f"of {args.direction_count}; checkpoints=49; train_t={train_t_count} evenly-spaced; "
        f"query_t={query_timestamp_count}; queries={list(DITD_QUERY_IDS)}; "
        f"train_batch={args.batch_size}; train_t_chunk={args.train_t_chunk_size}; "
        f"projection={DITD_PROJECTION_DIM}",
        flush=True,
    )

    for local_position, direction_index in enumerate(remaining, start=1):
        direction_started = time.perf_counter()
        # Keep each query timestamp separate until all checkpoint transitions
        # have been summed.  This is required for timestamp-sum-square.
        timestamp_accumulator = torch.zeros(
            (query_count, query_timestamp_count, N_TRAIN),
            device=device,
            dtype=torch.float32,
        )
        direction_linear = torch.zeros(
            expected_shape, device=device, dtype=torch.float64
        )
        direction_term_square = torch.zeros_like(direction_linear)

        for checkpoint_index in range(49):
            checkpoint_started = time.perf_counter()
            model, _, checkpoint = build_model(paths[checkpoint_index], "raw", device)
            target, _, _ = build_model(paths[checkpoint_index + 1], "raw", device)
            named = dict(model.named_parameters())
            names = tuple(named)
            parameters = tuple(named.values())
            projection_specs = build_countsketch_specs(
                list(parameters),
                DITD_PROJECTION_DIM,
                device=device,
                seed_parts=(
                    TRAIN_SEED,
                    "direction_integrated_tracin_das_projection",
                    checkpoint_index,
                ),
            )
            noise_generator = make_torch_generator(
                device,
                DITD_NOISE_SEED,
                "direction_integrated_noise",
                checkpoint_index,
                direction_index,
            )
            noise = torch.randn(
                endpoints.shape[1:],
                generator=noise_generator,
                device=device,
                dtype=endpoints.dtype,
            )

            endpoint_bank = endpoints[:, None].expand(
                query_count, query_timestamp_count, *endpoints.shape[1:]
            ).reshape(query_count * query_timestamp_count, *endpoints.shape[1:])
            timestep_bank = query_t[None].expand(
                query_count, query_timestamp_count
            ).reshape(-1)
            condition_bank = conditions[:, None].expand(
                query_count, query_timestamp_count, conditions.shape[-1]
            ).reshape(-1, conditions.shape[-1])
            noise_bank = noise.unsqueeze(0).expand_as(endpoint_bank)
            query_xt = base.q_sample(
                endpoint_bank, timestep_bank, noise_bank, schedule
            )
            with torch.no_grad():
                current_prediction = model(
                    query_xt, timestep_bank, condition_bank
                )
                next_prediction = target(
                    query_xt, timestep_bank, condition_bank
                )
                delta = next_prediction - current_prediction
                delta_norm = delta.reshape(delta.shape[0], -1).norm(
                    dim=1, keepdim=True
                )
                direction = delta / delta_norm.clamp_min(
                    DITD_DIRECTION_EPS
                ).reshape(-1, *([1] * (delta.ndim - 1)))

            def query_scalar(parameter_dict, xt, timestep, condition, output_direction):
                prediction = functional_call(
                    model,
                    parameter_dict,
                    (
                        xt.unsqueeze(0),
                        timestep.reshape(1),
                        condition.unsqueeze(0),
                    ),
                )
                return (prediction.squeeze(0) * output_direction).sum()

            batched_query_gradient = vmap(
                grad(query_scalar), in_dims=(None, 0, 0, 0, 0)
            )
            query_projected_chunks = []
            for query_start in range(
                0, len(query_xt), args.query_term_batch_size
            ):
                query_end = min(
                    query_start + args.query_term_batch_size, len(query_xt)
                )
                query_gradients = batched_query_gradient(
                    named,
                    query_xt[query_start:query_end],
                    timestep_bank[query_start:query_end],
                    condition_bank[query_start:query_end],
                    direction[query_start:query_end],
                )
                query_projected_chunks.append(
                    _project_batched_grads(
                        query_gradients,
                        names,
                        projection_specs,
                        DITD_PROJECTION_DIM,
                        False,
                        1e-8,
                    )
                )
                del query_gradients
            query_matrix = torch.cat(query_projected_chunks, dim=0)

            def train_loss_chunk(parameter_dict, x0, condition, timesteps, aligned_noise):
                count = timesteps.shape[0]
                x_bank = x0.unsqueeze(0).expand(count, *x0.shape)
                condition_bank_local = condition.unsqueeze(0).expand(
                    count, condition.shape[-1]
                )
                noise_bank_local = aligned_noise.unsqueeze(0).expand_as(x_bank)
                xt = base.q_sample(
                    x_bank, timesteps, noise_bank_local, schedule
                )
                prediction = functional_call(
                    model,
                    parameter_dict,
                    (xt, timesteps, condition_bank_local),
                )
                return (prediction - noise_bank_local).square().mean()

            batched_train_gradient = vmap(
                grad(train_loss_chunk), in_dims=(None, 0, 0, None, None)
            )
            checkpoint_lr = float(tracin_lr_weight(checkpoint))
            num_batches = math.ceil(N_TRAIN / args.batch_size)
            # Roughly twenty heartbeat lines per checkpoint pair.  The old
            # five-line cadence left long silent periods for integrated-t
            # gradients and looked stalled in screen logs.
            progress_every = max(1, num_batches // 20)
            for batch_position, train_start in enumerate(
                range(0, N_TRAIN, args.batch_size), start=1
            ):
                train_end = min(train_start + args.batch_size, N_TRAIN)
                train_matrix = torch.zeros(
                    (train_end - train_start, DITD_PROJECTION_DIM),
                    device=device,
                    dtype=torch.float32,
                )
                for t_start in range(0, train_t_count, args.train_t_chunk_size):
                    t_end = min(t_start + args.train_t_chunk_size, train_t_count)
                    t_chunk = train_t[t_start:t_end]
                    gradients = batched_train_gradient(
                        named,
                        x_all[train_start:train_end],
                        cond_all[train_start:train_end],
                        t_chunk,
                        noise,
                    )
                    projected = _project_batched_grads(
                        gradients,
                        names,
                        projection_specs,
                        DITD_PROJECTION_DIM,
                        False,
                        1e-8,
                    )
                    train_matrix += projected * (
                        float(len(t_chunk)) / float(train_t_count)
                    )
                    del gradients, projected

                dots = (query_matrix @ train_matrix.T).reshape(
                    query_count,
                    query_timestamp_count,
                    train_end - train_start,
                )
                weighted = checkpoint_lr * dots
                timestamp_accumulator[
                    :, :, train_start:train_end
                ] += weighted
                direction_linear[:, train_start:train_end] += (
                    weighted.sum(dim=1).to(torch.float64)
                    / float(query_timestamp_count)
                )
                direction_term_square[:, train_start:train_end] += (
                    checkpoint_lr
                    * dots.square().sum(dim=1).to(torch.float64)
                    / float(query_timestamp_count)
                )
                del train_matrix, dots, weighted
                if (
                    batch_position == 1
                    or batch_position % progress_every == 0
                    or batch_position == num_batches
                ):
                    print(
                        f"[direction-integrated gpu={args.gpu}] "
                        f"direction={direction_index + 1}/{args.direction_count} "
                        f"pair={checkpoint_index + 1}/49 "
                        f"batch={batch_position}/{num_batches}",
                        flush=True,
                    )

            pair_elapsed = time.perf_counter() - checkpoint_started
            print(
                f"[direction-integrated gpu={args.gpu}] "
                f"direction={direction_index + 1}/{args.direction_count} "
                f"pair={checkpoint_index + 1}/49 "
                f"delta_norm=[{float(delta_norm.min()):.3e},"
                f"{float(delta_norm.max()):.3e}] elapsed={pair_elapsed / 60:.1f}m",
                flush=True,
            )
            del model, target, checkpoint, named, parameters, projection_specs
            del noise, endpoint_bank, timestep_bank, condition_bank, noise_bank
            del query_xt, current_prediction, next_prediction, delta, delta_norm
            del direction, query_projected_chunks, query_matrix
            del batched_query_gradient, batched_train_gradient
            del query_scalar, train_loss_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        direction_weight = 1.0 / float(args.direction_count)
        scores["linear"] += direction_weight * direction_linear
        scores["termwise_squared"] += direction_weight * direction_term_square
        scores["timestamp_sum_squared"] += direction_weight * (
            timestamp_accumulator.square().sum(dim=1).to(torch.float64)
            / float(query_timestamp_count)
        )
        completed_directions.append(direction_index)
        completed_directions.sort()
        for contraction, path in partial_paths.items():
            atomic_numpy(path, scores[contraction].cpu().numpy())
        atomic_json(
            progress_path,
            {
                "contract_version": DITD_CONTRACT_VERSION,
                "batch_size": args.batch_size,
                "train_t_chunk_size": args.train_t_chunk_size,
                "train_t_count": train_t_count,
                "query_term_batch_size": args.query_term_batch_size,
                "direction_count": args.direction_count,
                "completed_directions": completed_directions,
            },
        )
        elapsed = time.perf_counter() - started
        completed_here = local_position
        eta = elapsed / completed_here * (len(remaining) - completed_here)
        print(
            f"[direction checkpoint gpu={args.gpu}] "
            f"completed={len(completed_directions)}/{len(selected_directions)} "
            f"direction_elapsed={(time.perf_counter()-direction_started)/3600:.2f}h "
            f"eta={eta/3600:.2f}h",
            flush=True,
        )
        del timestamp_accumulator, direction_linear, direction_term_square
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for contraction in methods:
        atomic_numpy(shard_root / f"{contraction}.npy", scores[contraction].cpu().numpy())
    atomic_json(
        done_path,
        {
            "contract_version": DITD_CONTRACT_VERSION,
            "query_ids": list(DITD_QUERY_IDS),
            "family": DITD_FAMILY,
            "direction_shard_index": args.direction_shard_index,
            "direction_shard_count": args.direction_shard_count,
            "direction_indices": selected_directions,
            "direction_count": args.direction_count,
            "train_timesteps": list(train_t_values),
            "query_timesteps": list(DITD_QUERY_TIMESTEPS),
            "train_t_reduction": (
                f"arithmetic mean over {train_t_count} evenly spaced diffusion "
                "indices including 0 and 999"
            ),
            "direction_alignment": (
                "one Gaussian noise direction shared by train loss and every "
                "query endpoint/timestamp within each checkpoint-direction term"
            ),
            "checkpoint_noise_policy": "independent 100-direction bank per checkpoint",
            "checkpoint_transitions": 49,
            "projection_dim": DITD_PROJECTION_DIM,
            "batch_size": args.batch_size,
            "train_t_chunk_size": args.train_t_chunk_size,
            "train_t_count": train_t_count,
            "query_term_batch_size": args.query_term_batch_size,
        },
    )
    print(f"[done] {done_path}", flush=True)


if __name__ == "__main__":
    main()
