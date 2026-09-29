"""Run final-EMA DAS with query-dependent inverse-noise trajectory losses."""

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
from trajectory_inverse_noise_das_config import *
from x3_endpoint_das_jax_logic_pytorch import (
    build_countsketch_specs,
    make_torch_generator,
    sample_output_probe,
)


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
    parser.add_argument("--query-scope", choices=("ten", "family"), default="ten")
    parser.add_argument("--family", choices=FAMILIES, default="prompted")
    parser.add_argument("--query-shard-index", type=int, default=0)
    parser.add_argument("--query-shard-count", type=int, default=1)
    parser.add_argument("--condition-batch-size", type=int, default=64)
    args = parser.parse_args()
    if not 0 <= args.timestamp_shard_index < args.timestamp_shard_count:
        raise ValueError("invalid timestamp shard")
    if not 0 <= args.query_shard_index < args.query_shard_count:
        raise ValueError("invalid query shard")
    if args.condition_batch_size <= 0:
        raise ValueError("condition batch size must be positive")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    with open(QUERY_DIR / "manifest.json") as handle:
        by_id = {int(record["query_id"]): record for record in json.load(handle)}
    if args.query_scope == "ten":
        query_ids = TRAJECTORY_INVERSE_DAS_QUERY_IDS
        family = TRAJECTORY_INVERSE_DAS_FAMILY
    else:
        family = args.family
        family_ids = trajectory_inverse_das_family_query_ids(family)
        query_ids = family_ids[
            args.query_shard_index :: args.query_shard_count
        ]
    records = [by_id[query_id] for query_id in query_ids]
    if not records or any(record["family"] != family for record in records):
        raise ValueError(f"invalid query bank for family={family}")

    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    model, _, checkpoint = build_model(
        model_paths(family)[-1], "ema", device
    )
    named = dict(model.named_parameters())
    names = tuple(named)
    active = tuple(named.values())
    schedule = base.make_linear_schedule(T, device=device)
    x_all, cond_all = preload_dataset(
        dataset, family, device
    )
    unique_conditions, condition_inverse, condition_counts = torch.unique(
        cond_all,
        dim=0,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    query_conditions = [cond_for(record, dataset, device) for record in records]
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

    included_timestamp_indices = list(range(99))
    selected = (
        included_timestamp_indices[
            args.timestamp_shard_index :: args.timestamp_shard_count
        ]
        if args.query_scope == "ten"
        else included_timestamp_indices
    )
    term_weight = 1.0 / (len(included_timestamp_indices) * int(DAS_NUM_MC))
    root = (
        trajectory_inverse_das_shard_root(
            args.timestamp_shard_index, args.timestamp_shard_count
        )
        if args.query_scope == "ten"
        else trajectory_inverse_das_100q_shard_root(
            family, args.query_shard_index, args.query_shard_count
        )
    )
    method = (
        TRAJECTORY_INVERSE_DAS_METHOD
        if args.query_scope == "ten"
        else TRAJECTORY_INVERSE_DAS_100Q_METHOD
    )
    done_path = root / "done.json"
    if done_path.is_file():
        print(f"[skip] {done_path}", flush=True)
        return

    partial_paths = {
        float(lam): root / f"partial_lambda_{lambda_tag(lam)}.npy"
        for lam in DAS_LAMBDAS
    }
    progress_path = root / "progress.json"
    completed_timestamps = []
    score_shape = (len(records), N_TRAIN)
    if all(path.is_file() for path in partial_paths.values()) and progress_path.is_file():
        with open(progress_path) as handle:
            progress = json.load(handle)
        if int(progress.get("contract_version", 0)) != CONTRACT_VERSION:
            raise ValueError("partial shard contract changed")
        if int(progress["condition_batch_size"]) != args.condition_batch_size:
            raise ValueError("partial shard condition batch size differs")
        completed_timestamps = [
            int(value) for value in progress["completed_timestamps"]
        ]
        scores = {
            lam: torch.from_numpy(np.load(path)).to(
                device=device, dtype=torch.float64
            )
            for lam, path in partial_paths.items()
        }
        print(
            f"[resume] timestamps={len(completed_timestamps)}/{len(selected)}",
            flush=True,
        )
    else:
        scores = {
            float(lam): torch.zeros(
                score_shape, device=device, dtype=torch.float64
            )
            for lam in DAS_LAMBDAS
        }

    remaining = [
        index for index in selected if index not in set(completed_timestamps)
    ]
    dimension = int(TRAJECTORY_INVERSE_DAS_PROJ_DIM)
    eye = torch.eye(dimension, device=device, dtype=torch.float32)
    total_terms = len(remaining) * len(records) * int(DAS_NUM_MC)
    completed_terms = 0
    started = time.perf_counter()
    print(
        f"[inverse-das gpu={args.gpu}] family={family} query_ids="
        f"{list(query_ids)} timestamps={len(selected)}/99 "
        f"outer_probes={DAS_NUM_MC} unique_conditions={len(unique_conditions)} "
        f"condition_batch={args.condition_batch_size} projection={dimension} "
        f"final_ema=True endpoint_excluded=True",
        flush=True,
    )

    for shard_position, timestamp_index in enumerate(remaining, start=1):
        timestep = int(trajectory_timesteps[timestamp_index])
        if timestep <= 0:
            raise ValueError("inverse-noise DAS must not include t=0")
        t_query = torch.tensor([timestep], device=device, dtype=torch.long)
        alpha_bar = schedule.alpha_bars[t_query].reshape(())
        sqrt_alpha = alpha_bar.sqrt()
        sqrt_one_minus = (1.0 - alpha_bar).sqrt().clamp_min(1e-12)

        for query_position, (record, trajectory, query_condition) in enumerate(
            zip(records, trajectories, query_conditions)
        ):
            query_state = torch.from_numpy(trajectory[timestamp_index]).to(
                device=device, dtype=torch.float32
            )
            query_single = query_state.squeeze(0)
            inferred_noise = (
                query_single.unsqueeze(0) - sqrt_alpha * x_all
            ) / sqrt_one_minus

            for mc_index in range(int(DAS_NUM_MC)):
                term_started = time.perf_counter()
                specs = build_countsketch_specs(
                    list(active),
                    dimension,
                    device=device,
                    seed_parts=(
                        812,
                        "trajectory_inverse_noise_das_projection",
                        timestamp_index,
                        mc_index,
                    ),
                )
                probe_generator = make_torch_generator(
                    device,
                    812,
                    "trajectory_inverse_noise_das_output_probe",
                    timestamp_index,
                    mc_index,
                )
                output_probe = sample_output_probe(
                    tuple(query_state.shape),
                    device=device,
                    rng=probe_generator,
                )
                probe_single = output_probe[0]
                scalar_denominator = math.sqrt(float(probe_single.numel()))

                current_query_prediction = model(
                    query_state, t_query, query_condition
                )
                query_scalar = (
                    (current_query_prediction * output_probe).sum()
                    / scalar_denominator
                )
                query_gradient = torch.autograd.grad(query_scalar, active)
                query_gradient_dict = {
                    name: value.unsqueeze(0)
                    for name, value in zip(names, query_gradient)
                }
                query_feature = _project_batched_grads(
                    query_gradient_dict,
                    names,
                    specs,
                    dimension,
                    bool(DAS_NORMALIZE_PROJECTED_GRADS),
                    1e-8,
                )[0].detach()

                def condition_scalar(parameter_dict, condition):
                    prediction = functional_call(
                        model,
                        parameter_dict,
                        (query_state, t_query, condition.unsqueeze(0)),
                    )
                    return (prediction * output_probe).sum() / scalar_denominator

                condition_gradient = vmap(
                    grad(condition_scalar), in_dims=(None, 0)
                )
                feature_unique = torch.empty(
                    (len(unique_conditions), dimension),
                    device=device,
                    dtype=torch.float32,
                )
                prediction_unique = torch.empty(
                    (len(unique_conditions), *query_single.shape),
                    device=device,
                    dtype=torch.float32,
                )
                for condition_start in range(
                    0, len(unique_conditions), args.condition_batch_size
                ):
                    condition_end = min(
                        condition_start + args.condition_batch_size,
                        len(unique_conditions),
                    )
                    condition_batch = unique_conditions[
                        condition_start:condition_end
                    ]
                    gradients = condition_gradient(named, condition_batch)
                    feature_unique[condition_start:condition_end] = (
                        _project_batched_grads(
                            gradients,
                            names,
                            specs,
                            dimension,
                            bool(DAS_NORMALIZE_PROJECTED_GRADS),
                            1e-8,
                        )
                    )
                    with torch.no_grad():
                        count = condition_end - condition_start
                        prediction_unique[condition_start:condition_end] = model(
                            query_state.expand(count, *query_state.shape[1:]),
                            t_query.expand(count),
                            condition_batch,
                        )

                weighted_features = feature_unique * condition_counts.to(
                    torch.float32
                ).sqrt().unsqueeze(1)
                gram = weighted_features.T @ weighted_features
                prediction_all = prediction_unique[condition_inverse]
                residual = (
                    ((prediction_all - inferred_noise) * probe_single)
                    .reshape(N_TRAIN, -1)
                    .sum(dim=1)
                    / scalar_denominator
                )

                for lam_raw in DAS_LAMBDAS:
                    lam = float(lam_raw)
                    solved_query = torch.linalg.solve(
                        gram + lam * eye, query_feature
                    )
                    coefficient_unique = feature_unique @ solved_query
                    raw = residual * coefficient_unique[condition_inverse]
                    if DAS_USE_SM_DENOMINATOR:
                        solved_unique = torch.linalg.solve(
                            gram + lam * eye, feature_unique.T
                        ).T
                        denominator_unique = 1.0 - (
                            feature_unique * solved_unique
                        ).sum(dim=1)
                        denominator_unique = torch.where(
                            denominator_unique.abs() < 1e-6,
                            denominator_unique.sign() * 1e-6,
                            denominator_unique,
                        )
                        raw = raw / denominator_unique[condition_inverse]
                    scores[lam][query_position] += (
                        term_weight * raw.to(torch.float64).square()
                    )

                completed_terms += 1
                elapsed = time.perf_counter() - started
                eta = elapsed / completed_terms * (total_terms - completed_terms)
                print(
                    f"[inverse-das gpu={args.gpu}] timestamp="
                    f"{shard_position}/{len(remaining)} global="
                    f"{timestamp_index + 1}/99 q={record['query_id']:02d} "
                    f"probe={mc_index + 1}/{DAS_NUM_MC} "
                    f"term_elapsed={(time.perf_counter()-term_started)/60:.1f}m "
                    f"eta={eta/3600:.2f}h",
                    flush=True,
                )
                del specs, output_probe, current_query_prediction
                del query_scalar, query_gradient, query_gradient_dict, query_feature
                del condition_gradient, feature_unique, prediction_unique
                del weighted_features, gram, prediction_all, residual
                del solved_query, coefficient_unique, raw, gradients
                if DAS_USE_SM_DENOMINATOR:
                    del solved_unique, denominator_unique
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        completed_timestamps.append(timestamp_index)
        completed_timestamps.sort()
        for lam, path in partial_paths.items():
            atomic_numpy(path, scores[lam].cpu().numpy())
        atomic_json(
            progress_path,
            {
                "contract_version": CONTRACT_VERSION,
                "condition_batch_size": args.condition_batch_size,
                "query_ids": list(query_ids),
                "query_scope": args.query_scope,
                "family": family,
                "completed_timestamps": completed_timestamps,
                "included_timestamp_indices": included_timestamp_indices,
                "endpoint_excluded": True,
            },
        )
        print(
            f"[checkpoint] timestamps={len(completed_timestamps)}/{len(selected)}",
            flush=True,
        )

    for lam, values in scores.items():
        atomic_numpy(root / f"lambda_{lambda_tag(lam)}.npy", values.cpu().numpy())
    atomic_json(
        done_path,
        {
            "contract_version": CONTRACT_VERSION,
            "method": method,
            "query_ids": list(query_ids),
            "query_scope": args.query_scope,
            "family": family,
            "timestamp_indices": selected,
            "included_timestamp_indices": included_timestamp_indices,
            "trajectory_timesteps": list(trajectory_timesteps),
            "endpoint_excluded": True,
            "term_weight": term_weight,
            "outer_probe_count": int(DAS_NUM_MC),
            "parameter_source": "final EMA",
            "checkpoint_epoch": int(checkpoint.get("epoch", EPOCHS)),
            "projection_dim": dimension,
            "normalize_features": bool(DAS_NORMALIZE_PROJECTED_GRADS),
            "sm_denominator": bool(DAS_USE_SM_DENOMINATOR),
            "unique_condition_count": int(len(unique_conditions)),
            "condition_batch_size": args.condition_batch_size,
            "train_loss_is_query_dependent": True,
            "loss_definition": (
                "epsilon*=(x_t_query-sqrt(alpha_bar_t)*x_i)/"
                "sqrt(1-alpha_bar_t); residual and features are evaluated "
                "at x_t_query with training condition c_i"
            ),
        },
    )
    print(f"[done] {root}", flush=True)


if __name__ == "__main__":
    main()
