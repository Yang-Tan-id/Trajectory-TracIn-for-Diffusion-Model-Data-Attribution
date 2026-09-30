"""Compute checkpoint-sharded Bundle TracIn response vectors.

For every checkpoint interval and training example, this worker replays the
four ``(t, epsilon)`` events actually used during the four training epochs in
that interval and averages their loss gradients.  These are exact training
events rather than newly sampled Monte Carlo draws.

For LDS we exploit linearity and aggregate projected training gradients with
the fixed membership matrix before applying the query output Jacobian.  In the
shared projected space this is exactly equivalent to computing every
``a_i = J_q (-eta g_i)`` first and summing ``a_i`` within each subset.
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
from bundle_tracin_config import *
from dataset_loader import ColorGridDataset
from forward_loss_alignment_config import replay_noise_path, replay_t_path
from x3_endpoint_das_jax_logic_pytorch import build_countsketch_specs


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


def projected_output_jacobian(model, parameters, names, specs, xt, timestep, condition):
    """Return all 27 output-gradient rows in the shared CountSketch space."""
    prediction = model(xt, timestep, condition).reshape(-1)
    output_dim = prediction.numel()
    if output_dim != BUNDLE_TRACIN_OUTPUT_DIM:
        raise ValueError(
            f"predicted-noise output has {output_dim} values, "
            f"expected {BUNDLE_TRACIN_OUTPUT_DIM}"
        )
    basis = torch.eye(output_dim, device=prediction.device, dtype=prediction.dtype)
    rows = torch.autograd.grad(
        prediction,
        parameters,
        grad_outputs=basis,
        is_grads_batched=True,
    )
    rows_by_name = {name: value for name, value in zip(names, rows)}
    return _project_batched_grads(
        rows_by_name,
        names,
        specs,
        BUNDLE_TRACIN_PROJ_DIM,
        False,
        1e-8,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint-shard-index", type=int, required=True)
    parser.add_argument("--checkpoint-shard-count", type=int, required=True)
    parser.add_argument("--query-ids", default="0-9")
    parser.add_argument("--batch-size", type=int, default=BUNDLE_TRACIN_BATCH_SIZE)
    args = parser.parse_args()
    if args.checkpoint_shard_count <= 0:
        raise ValueError("checkpoint shard count must be positive")
    if not 0 <= args.checkpoint_shard_index < args.checkpoint_shard_count:
        raise ValueError("checkpoint shard index is outside the shard count")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    query_ids = parse_query_ids(args.query_ids)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    missing = [query_id for query_id in query_ids if query_id not in by_id]
    if missing:
        raise ValueError(f"query IDs missing from manifest: {missing}")
    records = [by_id[query_id] for query_id in query_ids if by_id[query_id]["family"] == args.family]
    if not records:
        print(f"[skip] no requested queries for family={args.family}", flush=True)
        return

    paths = model_paths(args.family)
    if len(paths) != 50:
        raise ValueError(f"expected 50 {args.family} checkpoints, found {len(paths)}")
    checkpoint_indices = list(
        range(args.checkpoint_shard_index, len(paths), args.checkpoint_shard_count)
    )
    shard_root = (
        ATTR_DIR
        / BUNDLE_TRACIN_SHARD_NAMESPACE
        / args.family
        / f"shard_{args.checkpoint_shard_index:02d}_of_{args.checkpoint_shard_count:02d}"
    )
    done_path = shard_root / "done.json"
    if done_path.is_file():
        print(f"[skip] completed Bundle TracIn shard: {done_path}", flush=True)
        return
    shard_root.mkdir(parents=True, exist_ok=True)

    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    x_all, condition_all = preload_dataset(dataset, args.family, device)
    schedule = base.make_linear_schedule(T, device=device)
    membership_np = np.load(MASK_DIR / "membership.npy").astype(np.float32)
    expected_masks = SUBSETS_PER_SEED * len(LDS_SEEDS)
    if membership_np.shape != (expected_masks, N_TRAIN):
        raise ValueError(
            f"membership shape={membership_np.shape}, expected={(expected_masks, N_TRAIN)}"
        )
    membership = torch.from_numpy(membership_np).to(device=device)
    replay_t = np.load(replay_t_path(), mmap_mode="r")
    replay_noise = np.load(replay_noise_path(), mmap_mode="r")
    expected_t_shape = (50, N_TRAIN, BUNDLE_TRACIN_TRAIN_EVENTS)
    expected_noise_shape = expected_t_shape + (3, 3, 3)
    if replay_t.shape != expected_t_shape:
        raise ValueError(f"replay t shape={replay_t.shape}, expected={expected_t_shape}")
    if replay_noise.shape != expected_noise_shape:
        raise ValueError(
            f"replay noise shape={replay_noise.shape}, expected={expected_noise_shape}"
        )
    trajectories = {
        int(record["query_id"]): np.load(Path(record["dir"]) / "trajectory_xt.npy")
        for record in records
    }
    timestep_arrays = {
        int(record["query_id"]): np.load(Path(record["dir"]) / "trajectory_t.npy")
        for record in records
    }
    reference_timestamps = next(iter(timestep_arrays.values()))
    if len(reference_timestamps) != TRAJ_SNAPSHOTS or any(
        not np.array_equal(reference_timestamps, values)
        for values in timestep_arrays.values()
    ):
        raise ValueError("all requested queries must share the same 100 timestamps")
    conditions = {
        int(record["query_id"]): cond_for(record, dataset, device)
        for record in records
    }

    response_shape = (
        len(records),
        expected_masks,
        len(reference_timestamps),
        BUNDLE_TRACIN_OUTPUT_DIM,
    )
    progress_path = shard_root / "progress.json"
    completed = []
    responses = np.zeros(response_shape, dtype=np.float64)
    if progress_path.is_file():
        with open(progress_path) as handle:
            progress = json.load(handle)
        expected_ids = [int(record["query_id"]) for record in records]
        if progress.get("query_ids") != expected_ids:
            raise ValueError("query IDs changed since the shard was started")
        completed = [int(value) for value in progress["completed_checkpoint_indices"]]
        for query_position, query_id in enumerate(expected_ids):
            partial_path = shard_root / f"q{query_id:02d}_bundle_vectors.npy"
            if not partial_path.is_file():
                raise FileNotFoundError(partial_path)
            responses[query_position] = np.load(partial_path)
        print(
            f"[resume] family={args.family} shard={args.checkpoint_shard_index} "
            f"completed={len(completed)}/{len(checkpoint_indices)} checkpoints",
            flush=True,
        )

    remaining = [index for index in checkpoint_indices if index not in set(completed)]
    started = time.perf_counter()
    for local_position, checkpoint_index in enumerate(remaining, start=1):
        checkpoint_started = time.perf_counter()
        model, _, checkpoint = build_model(
            paths[checkpoint_index], BUNDLE_TRACIN_PARAM_SOURCE, device
        )
        named = dict(model.named_parameters())
        names = tuple(named)
        parameters = tuple(named.values())
        parameter_dict = dict(named)
        specs = build_countsketch_specs(
            list(parameters),
            BUNDLE_TRACIN_PROJ_DIM,
            device=device,
            seed_parts=(
                TRAIN_SEED,
                "bundle_tracin_projection",
                args.family,
                checkpoint_index,
            ),
        )

        event_count = BUNDLE_TRACIN_TRAIN_EVENTS

        def single_event_loss(params, x0, condition, timestep, noise):
            xt = base.q_sample(
                x0.unsqueeze(0), timestep.unsqueeze(0), noise.unsqueeze(0), schedule
            )
            prediction = functional_call(
                model,
                params,
                (xt, timestep.unsqueeze(0), condition.unsqueeze(0)),
            )
            return (prediction - noise.unsqueeze(0)).pow(2).mean()

        batched_gradient = vmap(
            grad(single_event_loss), in_dims=(None, 0, 0, 0, 0)
        )
        subset_features = torch.zeros(
            (expected_masks, BUNDLE_TRACIN_PROJ_DIM),
            device=device,
            dtype=torch.float32,
        )
        num_batches = math.ceil(N_TRAIN / args.batch_size)
        progress_every = max(1, num_batches // 10)
        for batch_position, start in enumerate(
            range(0, N_TRAIN, args.batch_size), start=1
        ):
            end = min(start + args.batch_size, N_TRAIN)
            x_batch = x_all[start:end]
            condition_batch = condition_all[start:end]
            point_count = end - start
            x_events = (
                x_batch[:, None]
                .expand(point_count, event_count, *x_batch.shape[1:])
                .reshape(point_count * event_count, *x_batch.shape[1:])
            )
            condition_events = (
                condition_batch[:, None]
                .expand(point_count, event_count, condition_batch.shape[-1])
                .reshape(point_count * event_count, condition_batch.shape[-1])
            )
            train_timesteps = torch.from_numpy(
                np.array(
                    replay_t[checkpoint_index, start:end], copy=True
                ).reshape(-1)
            ).to(device=device, dtype=torch.long)
            noises = torch.from_numpy(
                np.array(
                    replay_noise[checkpoint_index, start:end], copy=True
                ).reshape(point_count * event_count, *x_batch.shape[1:])
            ).to(device=device, dtype=x_batch.dtype)
            event_gradients = batched_gradient(
                parameter_dict,
                x_events,
                condition_events,
                train_timesteps,
                noises,
            )
            projected_events = _project_batched_grads(
                event_gradients,
                names,
                specs,
                BUNDLE_TRACIN_PROJ_DIM,
                False,
                1e-8,
            )
            projected = projected_events.reshape(
                point_count, event_count, BUNDLE_TRACIN_PROJ_DIM
            ).mean(dim=1)
            subset_features.add_(membership[:, start:end] @ projected)
            if (
                batch_position == 1
                or batch_position % progress_every == 0
                or batch_position == num_batches
            ):
                print(
                    f"[bundle-tracin gpu={args.gpu} family={args.family}] "
                    f"checkpoint={checkpoint_index + 1}/50 train-batch="
                    f"{batch_position}/{num_batches} points={end}/{N_TRAIN}",
                    flush=True,
                )

        learning_rate = float(tracin_lr_weight(checkpoint))
        for query_position, record in enumerate(records):
            query_id = int(record["query_id"])
            trajectory = trajectories[query_id]
            condition = conditions[query_id]
            for timestamp_position, timestep_value in enumerate(reference_timestamps):
                timestep = torch.tensor(
                    [int(timestep_value)], device=device, dtype=torch.long
                )
                xt = torch.from_numpy(trajectory[timestamp_position]).to(
                    device=device, dtype=torch.float32
                )
                query_jacobian = projected_output_jacobian(
                    model,
                    parameters,
                    names,
                    specs,
                    xt,
                    timestep,
                    condition,
                )
                # The signed SGD contribution is -eta * J_q g_i.  The common
                # sign disappears after the final squared norm but is retained
                # in the saved vectors so that future cross-checkpoint/vector
                # analyses remain semantically correct.
                value = (-learning_rate) * (subset_features @ query_jacobian.T)
                responses[query_position, :, timestamp_position, :] += (
                    value.detach().cpu().numpy().astype(np.float64)
                )
            print(
                f"[bundle-tracin gpu={args.gpu} family={args.family}] "
                f"checkpoint={checkpoint_index + 1}/50 q{query_id:02d} "
                f"timestamps={len(reference_timestamps)} done",
                flush=True,
            )

        completed.append(checkpoint_index)
        completed.sort()
        for query_position, record in enumerate(records):
            query_id = int(record["query_id"])
            atomic_numpy_save(
                shard_root / f"q{query_id:02d}_bundle_vectors.npy",
                responses[query_position].astype(np.float32),
            )
        atomic_json_save(
            progress_path,
            {
                "method": BUNDLE_TRACIN_METHOD,
                "family": args.family,
                "query_ids": [int(record["query_id"]) for record in records],
                "checkpoint_shard_index": args.checkpoint_shard_index,
                "checkpoint_shard_count": args.checkpoint_shard_count,
                "assigned_checkpoint_indices": checkpoint_indices,
                "completed_checkpoint_indices": completed,
                "train_events_per_checkpoint": event_count,
                "training_event_source": "exact_replay_cache",
                "batch_size": args.batch_size,
                "projection_dim": BUNDLE_TRACIN_PROJ_DIM,
            },
        )
        elapsed = time.perf_counter() - started
        print(
            f"[bundle-tracin gpu={args.gpu} family={args.family}] checkpoint="
            f"{checkpoint_index + 1}/50 shard-progress={local_position}/{len(remaining)} "
            f"checkpoint-elapsed={(time.perf_counter() - checkpoint_started)/60:.1f}m "
            f"shard-elapsed={elapsed/3600:.2f}h",
            flush=True,
        )
        del (
            model,
            checkpoint,
            named,
            parameters,
            parameter_dict,
            specs,
            batched_gradient,
            subset_features,
            event_gradients,
            projected_events,
            projected,
            noises,
            train_timesteps,
            query_jacobian,
            value,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if sorted(completed) != sorted(checkpoint_indices):
        raise RuntimeError("not all assigned checkpoints completed")
    atomic_json_save(
        done_path,
        {
            "method": BUNDLE_TRACIN_METHOD,
            "family": args.family,
            "query_ids": [int(record["query_id"]) for record in records],
            "checkpoint_shard_index": args.checkpoint_shard_index,
            "checkpoint_shard_count": args.checkpoint_shard_count,
            "checkpoint_indices": checkpoint_indices,
            "train_gradient": "mean_of_four_exact_replayed_training_event_gradients",
            "train_t_noise_alignment": "exact_training_events_not_query_aligned",
            "train_events_per_checkpoint": BUNDLE_TRACIN_TRAIN_EVENTS,
            "training_event_t_cache": str(replay_t_path()),
            "training_event_noise_cache": str(replay_noise_path()),
            "parameter_source": BUNDLE_TRACIN_PARAM_SOURCE,
            "projection": "countsketch_shared_by_train_and_query_jacobian",
            "projection_dim": BUNDLE_TRACIN_PROJ_DIM,
            "learning_rate_weighted": True,
            "response_vector": "-eta_c * J_query_checkpoint_timestamp * mean_train_gradient",
            "response_shape_per_query": [
                expected_masks,
                len(reference_timestamps),
                BUNDLE_TRACIN_OUTPUT_DIM,
            ],
        },
    )
    print(f"[done] {done_path}", flush=True)


if __name__ == "__main__":
    main()
