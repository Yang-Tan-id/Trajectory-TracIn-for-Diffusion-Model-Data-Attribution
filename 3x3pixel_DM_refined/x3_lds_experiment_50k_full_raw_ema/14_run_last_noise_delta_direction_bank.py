"""Score points using the last-checkpoint predicted-noise delta direction."""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

import x3pixel_DM_training as base
from attribution_one_query import (
    build_model, cond_for, model_paths, preload_dataset, tracin_lr_weight,
)
from checkpoint_counterfactual_config import *
from dataset_loader import ColorGridDataset
from exp_config import *
from run_exact_traj_next_bank import flatten_batched_gradients, flatten_gradient_tuple
from x3_endpoint_das_jax_logic_pytorch import make_torch_generator


SCORE_VERSION = 1
SHARD_NAMESPACE = "_last_noise_delta_direction_shards"


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
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--timestamp-shard-index", type=int, required=True)
    parser.add_argument("--timestamp-shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=CF_GRAD_BATCH_SIZE)
    args = parser.parse_args()
    if not 0 <= args.timestamp_shard_index < args.timestamp_shard_count:
        raise ValueError("invalid timestamp shard")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    with open(QUERY_DIR / "manifest.json") as handle:
        records = [item for item in json.load(handle) if item["family"] == args.family]
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    paths = model_paths(args.family)
    if len(paths) < 2:
        raise ValueError(f"need at least two checkpoints, found {len(paths)}")
    schedule = base.make_linear_schedule(T, device=device)
    x_all, cond_all = preload_dataset(dataset, args.family, device)
    trajectories = [np.load(item["dir"] + "/trajectory_xt.npy") for item in records]
    timestep_arrays = [np.load(item["dir"] + "/trajectory_t.npy") for item in records]
    t_seq = timestep_arrays[0]
    if len(t_seq) != TRAJ_SNAPSHOTS or any(
        not np.array_equal(t_seq, values) for values in timestep_arrays
    ):
        raise ValueError("query trajectory timestamps differ")
    conditions = [cond_for(item, dataset, device) for item in records]
    selected_timestamps = list(
        range(args.timestamp_shard_index, len(t_seq), args.timestamp_shard_count)
    )
    shard_root = (
        ATTR_DIR / SHARD_NAMESPACE / args.family
        / f"shard_{args.timestamp_shard_index:02d}_of_{args.timestamp_shard_count:02d}"
    )
    done_path = shard_root / "done.json"
    if done_path.is_file():
        print(f"[skip] {done_path}", flush=True)
        return
    partial_path = shard_root / "partial_linear.npy"
    progress_path = shard_root / "progress.json"
    query_ids = [int(item["query_id"]) for item in records]
    completed_timestamps = []
    if partial_path.is_file() and progress_path.is_file():
        with open(progress_path) as handle:
            progress = json.load(handle)
        if progress["query_ids"] != query_ids or progress["score_version"] != SCORE_VERSION:
            raise ValueError("partial score contract no longer matches")
        completed_timestamps = [int(value) for value in progress["completed_timestamps"]]
        scores = torch.from_numpy(np.load(partial_path)).to(device, torch.float64)
        print(f"[resume] timestamps={len(completed_timestamps)}/{len(selected_timestamps)}", flush=True)
    else:
        scores = torch.zeros((len(records), N_TRAIN), device=device, dtype=torch.float64)

    final_model, _, _ = build_model(paths[-1], "raw", device)
    final_model.requires_grad_(False)
    transitions = list(enumerate(paths[:-1]))
    remaining = [value for value in selected_timestamps if value not in set(completed_timestamps)]
    total_terms = len(remaining) * len(transitions)
    completed_terms = 0
    started = time.perf_counter()
    snapshot_weight = 1.0 / float(len(t_seq))

    print(
        f"[delta-direction {args.family}] queries={len(records)} checkpoints={len(transitions)} "
        f"timestamps={len(selected_timestamps)}/{len(t_seq)} batch={args.batch_size} "
        f"train_mc={CF_TRAIN_MC} output_delta_normalized={CF_DELTA_NORMALIZE}",
        flush=True,
    )
    for snapshot_index in remaining:
        timestep = int(t_seq[snapshot_index])
        query_timestep = torch.tensor([timestep], device=device, dtype=torch.long)
        for checkpoint_index, path in transitions:
            term_started = time.perf_counter()
            model, _, checkpoint = build_model(path, "raw", device)
            named = dict(model.named_parameters())
            names = tuple(named)
            parameters = tuple(named.values())
            query_vectors = []
            for trajectory, condition in zip(trajectories, conditions):
                x_query = torch.from_numpy(trajectory[snapshot_index]).to(
                    device=device, dtype=torch.float32
                )
                current_epsilon = model(x_query, query_timestep, condition)
                with torch.no_grad():
                    last_epsilon = final_model(x_query, query_timestep, condition)
                    direction = last_epsilon - current_epsilon.detach()
                    if CF_DELTA_NORMALIZE:
                        direction = direction / direction.norm().clamp_min(1e-12)
                projected_noise = (current_epsilon * direction).sum()
                query_gradient = torch.autograd.grad(projected_noise, parameters)
                query_vectors.append(flatten_gradient_tuple(query_gradient))
            query_matrix = torch.stack(query_vectors).detach()

            train_timestep = torch.full(
                (CF_TRAIN_MC,), timestep, device=device, dtype=torch.long
            )

            def one_loss(pdict, x0, condition, noises):
                x_mc = x0.unsqueeze(0).expand(CF_TRAIN_MC, *x0.shape)
                c_mc = condition.unsqueeze(0).expand(CF_TRAIN_MC, condition.shape[-1])
                x_t = base.q_sample(x_mc, train_timestep, noises, schedule)
                prediction = functional_call(model, pdict, (x_t, train_timestep, c_mc))
                return (prediction - noises).pow(2).reshape(CF_TRAIN_MC, -1).mean()

            batched_gradient = vmap(grad(one_loss), in_dims=(None, 0, 0, 0))
            weight = float(tracin_lr_weight(checkpoint)) * snapshot_weight
            num_batches = math.ceil(N_TRAIN / args.batch_size)
            progress_every = max(1, num_batches // 10)
            for batch_index, start in enumerate(range(0, N_TRAIN, args.batch_size), start=1):
                end = min(start + args.batch_size, N_TRAIN)
                x_batch, condition_batch = x_all[start:end], cond_all[start:end]
                generator = make_torch_generator(
                    device, TRAIN_SEED, "projected_traj_train", checkpoint_index,
                    snapshot_index, start, CF_TRAIN_MC,
                )
                noises = torch.randn(
                    (end - start, CF_TRAIN_MC, *x_batch.shape[1:]),
                    generator=generator, device=device, dtype=x_batch.dtype,
                )
                gradients = batched_gradient(named, x_batch, condition_batch, noises)
                train_matrix = flatten_batched_gradients(gradients, names).detach()
                dots = torch.matmul(train_matrix, query_matrix.T).T.to(torch.float64)
                scores[:, start:end] += weight * dots
                if batch_index == 1 or batch_index % progress_every == 0 or batch_index == num_batches:
                    print(
                        f"[delta-direction {args.family}] timestamp={snapshot_index+1}/100 "
                        f"checkpoint={checkpoint_index+1}/{len(transitions)} "
                        f"batch={batch_index}/{num_batches} points={end}/{N_TRAIN}",
                        flush=True,
                    )
            completed_terms += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / completed_terms * (total_terms - completed_terms)
            print(
                f"[delta-direction {args.family}] term={completed_terms}/{total_terms} "
                f"term_elapsed={(time.perf_counter()-term_started)/60:.1f}m "
                f"elapsed={elapsed/3600:.2f}h eta≈{eta/3600:.2f}h",
                flush=True,
            )
            del model, checkpoint, named, parameters, query_vectors, query_matrix
            del current_epsilon, last_epsilon, direction, projected_noise, query_gradient
            del batched_gradient, gradients, train_matrix, dots, noises
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        completed_timestamps.append(snapshot_index)
        completed_timestamps.sort()
        atomic_numpy(partial_path, scores.cpu().numpy())
        atomic_json(
            progress_path,
            {
                "family": args.family,
                "query_ids": query_ids,
                "completed_timestamps": completed_timestamps,
                "score_version": SCORE_VERSION,
            },
        )
        print(f"[checkpoint] timestamps={len(completed_timestamps)}/{len(selected_timestamps)}", flush=True)

    if sorted(completed_timestamps) != sorted(selected_timestamps):
        raise RuntimeError("not all timestamps completed")
    atomic_numpy(shard_root / "linear.npy", scores.cpu().numpy())
    atomic_json(
        done_path,
        {
            "method": CF_DIRECTION_SCORE_METHOD,
            "family": args.family,
            "query_ids": query_ids,
            "timestamp_indices": selected_timestamps,
            "timestamp_shard_index": args.timestamp_shard_index,
            "timestamp_shard_count": args.timestamp_shard_count,
            "current_param_source": "raw",
            "reference_param_source": "raw",
            "reference_checkpoint": str(paths[-1]),
            "current_checkpoints": len(transitions),
            "query_scalar": "dot(epsilon_current, normalize(epsilon_last-epsilon_current))",
            "delta_normalized": CF_DELTA_NORMALIZE,
            "parameter_projection": None,
            "train_mc": CF_TRAIN_MC,
            "score_version": SCORE_VERSION,
        },
    )
    print(f"[done] {shard_root}", flush=True)


if __name__ == "__main__":
    main()
