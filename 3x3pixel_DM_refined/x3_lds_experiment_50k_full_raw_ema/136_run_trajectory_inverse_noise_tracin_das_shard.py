"""Score query-dependent inverse-noise training losses on trajectory states."""

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
    tracin_lr_weight,
)
from dataset_loader import ColorGridDataset
from trajectory_inverse_noise_tracin_das_config import *
from x3_endpoint_das_jax_logic_pytorch import build_countsketch_specs


CONTRACT_VERSION = 1


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
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not 0 <= args.timestamp_shard_index < args.timestamp_shard_count:
        raise ValueError("invalid timestamp shard")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    with open(QUERY_DIR / "manifest.json") as handle:
        by_id = {int(record["query_id"]): record for record in json.load(handle)}
    records = [by_id[query_id] for query_id in INVERSE_NOISE_QUERY_IDS]
    if any(record["family"] != INVERSE_NOISE_FAMILY for record in records):
        raise ValueError("q00-q09 must all be prompted queries")

    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    checkpoint_paths = model_paths(INVERSE_NOISE_FAMILY)
    if len(checkpoint_paths) != 50:
        raise ValueError(f"expected 50 checkpoints, found {len(checkpoint_paths)}")
    schedule = base.make_linear_schedule(T, device=device)
    x_all, cond_all = preload_dataset(dataset, INVERSE_NOISE_FAMILY, device)
    conditions = [cond_for(record, dataset, device) for record in records]
    trajectories = [
        np.load(Path(record["dir"]) / "trajectory_xt.npy") for record in records
    ]
    timestep_banks = [
        np.load(Path(record["dir"]) / "trajectory_t.npy") for record in records
    ]
    trajectory_timesteps = tuple(int(value) for value in timestep_banks[0])
    if len(trajectory_timesteps) != 100 or any(
        not np.array_equal(timestep_banks[0], values)
        for values in timestep_banks[1:]
    ):
        raise ValueError("q00-q09 trajectory timestamp banks differ")

    # Exclude index 99: final generated x0 at diffusion timestamp zero.
    included_timestamp_indices = list(range(99))
    selected = included_timestamp_indices[
        args.timestamp_shard_index :: args.timestamp_shard_count
    ]
    snapshot_weight = 1.0 / float(len(included_timestamp_indices))
    root = inverse_noise_shard_root(
        args.timestamp_shard_index, args.timestamp_shard_count
    )
    done_path = root / "done.json"
    if done_path.is_file():
        print(f"[skip] {done_path}", flush=True)
        return

    score_paths = {
        contraction: root / f"partial_{contraction}.npy"
        for contraction in INVERSE_NOISE_CONTRACTIONS
    }
    progress_path = root / "progress.json"
    completed_timestamps = []
    score_shape = (len(records), N_TRAIN)
    if all(path.is_file() for path in score_paths.values()) and progress_path.is_file():
        with open(progress_path) as handle:
            progress = json.load(handle)
        if int(progress.get("contract_version", 0)) != CONTRACT_VERSION:
            raise ValueError("partial shard contract changed")
        if int(progress["batch_size"]) != args.batch_size:
            raise ValueError("partial shard batch size differs")
        completed_timestamps = [
            int(value) for value in progress["completed_timestamps"]
        ]
        scores = {
            contraction: torch.from_numpy(np.load(path)).to(
                device=device, dtype=torch.float64
            )
            for contraction, path in score_paths.items()
        }
        if any(value.shape != score_shape for value in scores.values()):
            raise ValueError("partial score shape mismatch")
        print(
            f"[resume] timestamps={len(completed_timestamps)}/{len(selected)}",
            flush=True,
        )
    else:
        scores = {
            contraction: torch.zeros(
                score_shape, device=device, dtype=torch.float64
            )
            for contraction in INVERSE_NOISE_CONTRACTIONS
        }

    remaining = [
        index for index in selected if index not in set(completed_timestamps)
    ]
    transitions = tuple((index, index + 1) for index in range(49))
    total_terms = len(remaining) * len(transitions)
    completed_terms = 0
    started = time.perf_counter()
    print(
        f"[inverse-noise gpu={args.gpu}] q00-q09 timestamps="
        f"{len(selected)}/99 transitions=49 batch={args.batch_size} "
        f"projection={INVERSE_NOISE_PROJ_DIM} endpoint_excluded=True",
        flush=True,
    )

    for shard_position, timestamp_index in enumerate(remaining, start=1):
        timestep = int(trajectory_timesteps[timestamp_index])
        if timestep <= 0:
            raise ValueError("inverse-noise experiment must not include t=0")
        t_query = torch.tensor([timestep], device=device, dtype=torch.long)
        alpha_bar = schedule.alpha_bars[t_query].reshape(())
        sqrt_alpha = alpha_bar.sqrt()
        sqrt_one_minus = (1.0 - alpha_bar).sqrt().clamp_min(1e-12)
        query_states = [
            torch.from_numpy(trajectory[timestamp_index]).to(
                device=device, dtype=torch.float32
            )
            for trajectory in trajectories
        ]
        timestamp_accumulator = torch.zeros(
            score_shape, device=device, dtype=torch.float64
        )

        for transition_position, (checkpoint_index, target_index) in enumerate(
            transitions, start=1
        ):
            term_started = time.perf_counter()
            model, _, checkpoint = build_model(
                checkpoint_paths[checkpoint_index], "raw", device
            )
            target, _, _ = build_model(
                checkpoint_paths[target_index], "raw", device
            )
            named = dict(model.named_parameters())
            names = tuple(named)
            parameters = tuple(named.values())
            specs = build_countsketch_specs(
                list(parameters),
                INVERSE_NOISE_PROJ_DIM,
                device=device,
                seed_parts=(
                    TRAIN_SEED,
                    "trajectory_inverse_noise_projection",
                    checkpoint_index,
                ),
            )

            query_vectors = []
            delta_norms = []
            for query_state, query_condition in zip(query_states, conditions):
                current_prediction = model(
                    query_state, t_query, query_condition
                )
                with torch.no_grad():
                    next_prediction = target(
                        query_state, t_query, query_condition
                    )
                    direction = next_prediction - current_prediction.detach()
                    delta_norm = direction.norm()
                    direction = direction / delta_norm.clamp_min(1e-12)
                scalar = (current_prediction * direction).sum()
                query_gradient = torch.autograd.grad(scalar, parameters)
                query_vectors.append(
                    _project_gradient_tuple(
                        query_gradient, specs, INVERSE_NOISE_PROJ_DIM
                    )
                )
                delta_norms.append(float(delta_norm))
            query_matrix = torch.stack(query_vectors).detach()

            def inverse_noise_loss(parameter_dict, x0, condition, query_state):
                query_single = query_state.squeeze(0)
                inferred_noise = (
                    query_single - sqrt_alpha * x0
                ) / sqrt_one_minus
                prediction = functional_call(
                    model,
                    parameter_dict,
                    (query_state, t_query, condition.unsqueeze(0)),
                ).squeeze(0)
                return (prediction - inferred_noise).square().mean()

            batched_gradient = vmap(
                grad(inverse_noise_loss), in_dims=(None, 0, 0, None)
            )
            checkpoint_lr = float(tracin_lr_weight(checkpoint))
            weight = checkpoint_lr * snapshot_weight
            num_batches = math.ceil(N_TRAIN / args.batch_size)
            progress_every = max(1, num_batches // 5)
            for batch_position, start in enumerate(
                range(0, N_TRAIN, args.batch_size), start=1
            ):
                end = min(start + args.batch_size, N_TRAIN)
                for query_position, query_state in enumerate(query_states):
                    gradients = batched_gradient(
                        named,
                        x_all[start:end],
                        cond_all[start:end],
                        query_state,
                    )
                    train_matrix = _project_batched_grads(
                        gradients,
                        names,
                        specs,
                        INVERSE_NOISE_PROJ_DIM,
                        False,
                        1e-8,
                    ).detach()
                    dots = (
                        train_matrix @ query_matrix[query_position]
                    ).to(torch.float64)
                    scores["linear"][query_position, start:end] += weight * dots
                    scores["termwise_squared"][query_position, start:end] += (
                        weight * dots.square()
                    )
                    timestamp_accumulator[
                        query_position, start:end
                    ] += weight * dots
                if (
                    batch_position == 1
                    or batch_position % progress_every == 0
                    or batch_position == num_batches
                ):
                    print(
                        f"[inverse-noise gpu={args.gpu}] timestamp="
                        f"{shard_position}/{len(remaining)} "
                        f"global={timestamp_index + 1}/99 "
                        f"pair={transition_position}/49 "
                        f"batch={batch_position}/{num_batches}",
                        flush=True,
                    )

            completed_terms += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / completed_terms * (total_terms - completed_terms)
            print(
                f"[inverse-noise gpu={args.gpu}] term={completed_terms}/"
                f"{total_terms} t={timestep} delta_norm="
                f"[{min(delta_norms):.3e},{max(delta_norms):.3e}] "
                f"term_elapsed={(time.perf_counter()-term_started)/60:.1f}m "
                f"eta={eta/3600:.2f}h",
                flush=True,
            )
            del model, target, named, parameters, specs
            del query_vectors, query_matrix, query_gradient
            del current_prediction, next_prediction, direction, scalar
            del batched_gradient, gradients, train_matrix, dots
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        scores["timestamp_sum_squared"] += timestamp_accumulator.square()
        completed_timestamps.append(timestamp_index)
        completed_timestamps.sort()
        for contraction, path in score_paths.items():
            atomic_numpy(path, scores[contraction].cpu().numpy())
        atomic_json(
            progress_path,
            {
                "contract_version": CONTRACT_VERSION,
                "batch_size": args.batch_size,
                "query_ids": list(INVERSE_NOISE_QUERY_IDS),
                "completed_timestamps": completed_timestamps,
                "included_timestamp_indices": included_timestamp_indices,
                "endpoint_excluded": True,
            },
        )
        print(
            f"[checkpoint] timestamps={len(completed_timestamps)}/{len(selected)}",
            flush=True,
        )

    for contraction, values in scores.items():
        atomic_numpy(root / f"{contraction}.npy", values.cpu().numpy())
    atomic_json(
        done_path,
        {
            "contract_version": CONTRACT_VERSION,
            "query_ids": list(INVERSE_NOISE_QUERY_IDS),
            "timestamp_indices": selected,
            "included_timestamp_indices": included_timestamp_indices,
            "trajectory_timesteps": list(trajectory_timesteps),
            "endpoint_excluded": True,
            "timestamp_weight": snapshot_weight,
            "checkpoint_pairs": [list(value) for value in transitions],
            "checkpoint_count": len(checkpoint_paths),
            "parameter_source": "raw",
            "projection_dim": INVERSE_NOISE_PROJ_DIM,
            "batch_size": args.batch_size,
            "loss_definition": (
                "for each (query trajectory state, training point), infer "
                "epsilon=(x_t_query-sqrt(alpha_bar_t)*x_i)/"
                "sqrt(1-alpha_bar_t), then differentiate MSE at x_t_query"
            ),
        },
    )
    print(f"[done] {root}", flush=True)


if __name__ == "__main__":
    main()
