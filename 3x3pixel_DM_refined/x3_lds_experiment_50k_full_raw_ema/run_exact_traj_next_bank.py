"""Exact, unprojected first-order next-checkpoint Traj-TracIn family bank."""

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
    build_model,
    cond_for,
    model_paths,
    preload_dataset,
    tracin_lr_weight,
)
from dataset_loader import ColorGridDataset
from exp_config import *
from x3_endpoint_das_jax_logic_pytorch import make_torch_generator


METHOD = "traj_next_raw_exact_aligned_100q"
SHARD_NAMESPACE = "_exact_traj_next_raw_aligned_shards"
SCORE_CONTRACT_VERSION = 2


def flatten_gradient_tuple(values):
    return torch.cat([value.reshape(-1) for value in values], dim=0)


def flatten_batched_gradients(values, names):
    batch = values[names[0]].shape[0]
    return torch.cat([values[name].reshape(batch, -1) for name in names], dim=1)


def atomic_numpy_save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def atomic_json_save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timestamp-shard-index", type=int, required=True)
    parser.add_argument("--timestamp-shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=TRACIN_PROJECTED_BATCH_SIZE)
    args = parser.parse_args()
    if args.timestamp_shard_count <= 0:
        raise ValueError("--timestamp-shard-count must be positive")
    if not 0 <= args.timestamp_shard_index < args.timestamp_shard_count:
        raise ValueError("timestamp shard index is outside the shard count")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    with open(QUERY_DIR / "manifest.json") as handle:
        records = [item for item in json.load(handle) if item["family"] == args.family]
    if not records:
        raise ValueError(f"no queries found for family={args.family}")

    shard_root = (
        ATTR_DIR / SHARD_NAMESPACE / args.family
        / f"shard_{args.timestamp_shard_index:02d}_of_{args.timestamp_shard_count:02d}"
    )
    done_path = shard_root / "done.json"
    if done_path.is_file():
        print(f"[skip] exact Traj shard complete: {done_path}", flush=True)
        return

    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    paths = model_paths(args.family)
    schedule = base.make_linear_schedule(T, device=device)
    x_all, cond_all = preload_dataset(dataset, args.family, device)
    trajectories = [np.load(Path(item["dir"]) / "trajectory_xt.npy") for item in records]
    timestep_arrays = [np.load(Path(item["dir"]) / "trajectory_t.npy") for item in records]
    t_seq = timestep_arrays[0]
    if len(t_seq) != 100 or any(
        not np.array_equal(t_seq, values) for values in timestep_arrays
    ):
        raise ValueError("all family queries must share the same 100 timestamps")
    selected_timestamps = list(
        range(args.timestamp_shard_index, len(t_seq), args.timestamp_shard_count)
    )
    conditions = [cond_for(item, dataset, device) for item in records]
    shard_root.mkdir(parents=True, exist_ok=True)
    partial_path = shard_root / "partial_linear.npy"
    progress_path = shard_root / "progress.json"
    completed_timestamps = []
    if partial_path.is_file() and progress_path.is_file():
        with open(progress_path) as handle:
            progress = json.load(handle)
        expected_ids = [int(item["query_id"]) for item in records]
        if progress.get("query_ids") != expected_ids:
            raise ValueError(f"query IDs changed since {progress_path} was written")
        if progress.get("score_contract_version") != SCORE_CONTRACT_VERSION:
            raise ValueError(f"score contract changed since {progress_path} was written")
        completed_timestamps = [int(value) for value in progress["completed_timestamps"]]
        partial = np.load(partial_path)
        expected_shape = (len(records), N_TRAIN)
        if partial.shape != expected_shape:
            raise ValueError(
                f"{partial_path} has shape {partial.shape}, expected {expected_shape}"
            )
        scores = torch.from_numpy(partial).to(device=device, dtype=torch.float64)
        print(
            f"[resume] exact Traj shard has {len(completed_timestamps)}/"
            f"{len(selected_timestamps)} timestamps complete",
            flush=True,
        )
    else:
        scores = torch.zeros(
            (len(records), N_TRAIN), device=device, dtype=torch.float64
        )
    mc_count = int(TRACIN_TRAIN_MC)
    snapshot_weight = 1.0 / float(len(t_seq))
    transitions = [(index, index + 1) for index in range(len(paths) - 1)]
    remaining_timestamps = [
        value for value in selected_timestamps if value not in set(completed_timestamps)
    ]
    total_terms = len(remaining_timestamps) * len(transitions)
    completed_terms = 0
    started = time.perf_counter()

    print(
        f"[exact-traj-next {args.family}] queries={len(records)} "
        f"parameters=full/no_projection transitions={len(transitions)} "
        f"timestamps={len(selected_timestamps)}/{len(t_seq)} "
        f"shard={args.timestamp_shard_index}/{args.timestamp_shard_count} "
        f"batch={args.batch_size} train_mc={mc_count}",
        flush=True,
    )

    for shard_timestamp_index, snapshot_index in enumerate(
        remaining_timestamps, start=1
    ):
        timestep = int(t_seq[snapshot_index])
        for transition_index, (checkpoint_index, target_index) in enumerate(
            transitions, start=1
        ):
            term_started = time.perf_counter()
            model, model_dataset, checkpoint = build_model(
                paths[checkpoint_index], "raw", device
            )
            target, target_dataset, target_checkpoint = build_model(
                paths[target_index], "raw", device
            )
            named = dict(model.named_parameters())
            names = tuple(named)
            parameters = tuple(named.values())
            parameter_dict = dict(named)

            query_vectors = []
            query_timestep = torch.tensor([timestep], device=device, dtype=torch.long)
            for trajectory, condition in zip(trajectories, conditions):
                xt_query = torch.from_numpy(trajectory[snapshot_index]).to(
                    device=device, dtype=torch.float32
                )
                with torch.no_grad():
                    epsilon_target = target(
                        xt_query, query_timestep, condition
                    ).detach()
                query_loss = (
                    model(xt_query, query_timestep, condition) - epsilon_target
                ).pow(2).sum()
                query_gradient = torch.autograd.grad(query_loss, parameters)
                query_vectors.append(flatten_gradient_tuple(query_gradient))
            query_matrix = torch.stack(query_vectors).to(dtype=torch.float32).detach()
            parameter_count = int(query_matrix.shape[1])

            train_timestep = torch.full(
                (mc_count,), timestep, device=device, dtype=torch.long
            )

            def single_mean_loss(params, x0, condition, noises):
                x_mc = x0.unsqueeze(0).expand(mc_count, *x0.shape)
                condition_mc = condition.unsqueeze(0).expand(
                    mc_count, condition.shape[-1]
                )
                xt = base.q_sample(x_mc, train_timestep, noises, schedule)
                prediction = functional_call(
                    model, params, (xt, train_timestep, condition_mc)
                )
                return (
                    (prediction - noises).pow(2).reshape(mc_count, -1)
                    .mean(dim=1).mean()
                )

            batched_gradient = vmap(
                grad(single_mean_loss), in_dims=(None, 0, 0, 0)
            )
            weight = float(tracin_lr_weight(checkpoint)) * snapshot_weight
            num_batches = math.ceil(N_TRAIN / args.batch_size)
            progress_every = max(1, num_batches // 10)
            for batch_index, start in enumerate(
                range(0, N_TRAIN, args.batch_size), start=1
            ):
                end = min(start + args.batch_size, N_TRAIN)
                x_batch, condition_batch = x_all[start:end], cond_all[start:end]
                generator = make_torch_generator(
                    device, TRAIN_SEED, "projected_traj_train", checkpoint_index,
                    snapshot_index, start, mc_count,
                )
                noises = torch.randn(
                    (end - start, mc_count, *x_batch.shape[1:]),
                    generator=generator,
                    device=device,
                    dtype=x_batch.dtype,
                )
                gradients = batched_gradient(
                    parameter_dict, x_batch, condition_batch, noises
                )
                train_matrix = flatten_batched_gradients(gradients, names).detach()
                # The projected worker divides both CountSketch vectors by
                # sqrt(d), so its dot has expected scale full_dot / d. Keep
                # that positive scalar here for numerical comparability; it
                # does not affect descending top-k ranks.
                dots = (
                    torch.matmul(train_matrix, query_matrix.T)
                    / float(TRACIN_PROJ_DIM)
                ).T.to(torch.float64)
                scores[:, start:end] += weight * dots
                if (
                    batch_index == 1
                    or batch_index % progress_every == 0
                    or batch_index == num_batches
                ):
                    print(
                        f"[exact-traj-next {args.family}] "
                        f"timestamp={snapshot_index + 1}/100 "
                        f"shard_timestamp={shard_timestamp_index}/"
                        f"{len(remaining_timestamps)} "
                        f"transition={transition_index}/{len(transitions)} "
                        f"batch={batch_index}/{num_batches} points={end}/{N_TRAIN}",
                        flush=True,
                    )

            completed_terms += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / completed_terms * (total_terms - completed_terms)
            print(
                f"[exact-traj-next {args.family}] term={completed_terms}/{total_terms} "
                f"full_parameters={parameter_count} "
                f"term_elapsed={(time.perf_counter() - term_started)/60:.1f}m "
                f"elapsed={elapsed/3600:.2f}h eta≈{eta/3600:.2f}h",
                flush=True,
            )
            del (
                model, target, model_dataset, target_dataset, checkpoint,
                target_checkpoint, named, parameters, parameter_dict,
                query_vectors, query_matrix, query_gradient, query_loss,
                epsilon_target, batched_gradient, single_mean_loss,
                gradients, train_matrix, dots, noises,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        completed_timestamps.append(snapshot_index)
        completed_timestamps.sort()
        atomic_numpy_save(partial_path, scores.cpu().numpy())
        atomic_json_save(
            progress_path,
            {
                "family": args.family,
                "query_ids": [int(item["query_id"]) for item in records],
                "timestamp_shard_index": args.timestamp_shard_index,
                "timestamp_shard_count": args.timestamp_shard_count,
                "completed_timestamps": completed_timestamps,
                "score_contract_version": SCORE_CONTRACT_VERSION,
            },
        )
        print(
            f"[checkpoint] exact Traj shard timestamps="
            f"{len(completed_timestamps)}/{len(selected_timestamps)}",
            flush=True,
        )

    if sorted(completed_timestamps) != sorted(selected_timestamps):
        raise RuntimeError("not all assigned timestamps completed")
    atomic_numpy_save(shard_root / "linear.npy", scores.cpu().numpy())
    atomic_json_save(
        done_path,
        {
            "family": args.family,
            "method": METHOD,
            "query_ids": [int(item["query_id"]) for item in records],
            "timestamp_indices": selected_timestamps,
            "timestamp_shard_index": args.timestamp_shard_index,
            "timestamp_shard_count": args.timestamp_shard_count,
            "target": "next",
            "param_source": "raw",
            "projection": None,
            "train_mc": mc_count,
            "batch_size": args.batch_size,
            "score_contract_version": SCORE_CONTRACT_VERSION,
            "train_noise_seed_contract": (
                "TRAIN_SEED/projected_traj_train/checkpoint/timestamp/batch/mc"
            ),
            "score_scale": f"full_gradient_dot/{int(TRACIN_PROJ_DIM)}",
        },
    )
    print(f"[done] exact Traj shard saved: {shard_root}", flush=True)


if __name__ == "__main__":
    main()
