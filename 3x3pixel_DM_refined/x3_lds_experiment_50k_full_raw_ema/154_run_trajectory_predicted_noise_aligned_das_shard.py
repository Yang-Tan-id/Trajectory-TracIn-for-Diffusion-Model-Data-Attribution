"""One timestamp shard of query-dependent predicted-noise aligned DAS."""

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
from trajectory_predicted_noise_aligned_das_config import *
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
    parser.add_argument("--feature-batch-size", type=int, default=64)
    args = parser.parse_args()
    if not 0 <= args.timestamp_shard_index < args.timestamp_shard_count:
        raise ValueError("invalid timestamp shard")
    if args.feature_batch_size <= 0:
        raise ValueError("feature batch size must be positive")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    with open(QUERY_DIR / "manifest.json") as handle:
        by_id = {int(record["query_id"]): record for record in json.load(handle)}
    records = [by_id[query_id] for query_id in TPNA_DAS_QUERY_IDS]
    if any(record["family"] != TPNA_DAS_FAMILY for record in records):
        raise ValueError("q00-q09 must be prompted")
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    model, _, checkpoint = build_model(
        model_paths(TPNA_DAS_FAMILY)[-1], "ema", device
    )
    named = dict(model.named_parameters())
    names = tuple(named)
    active = tuple(named.values())
    schedule = base.make_linear_schedule(T, device=device)
    x_all, cond_all = preload_dataset(dataset, TPNA_DAS_FAMILY, device)
    query_conditions = [cond_for(record, dataset, device) for record in records]
    trajectories = [
        np.load(Path(record["dir"]) / "trajectory_xt.npy", mmap_mode="r")
        for record in records
    ]
    timestep_banks = [
        np.load(Path(record["dir"]) / "trajectory_t.npy") for record in records
    ]
    trajectory_timesteps = tuple(int(value) for value in timestep_banks[0])
    if len(trajectory_timesteps) != 100 or any(
        not np.array_equal(timestep_banks[0], values)
        for values in timestep_banks[1:]
    ):
        raise ValueError("trajectory timestamp banks differ")
    included_timestamp_indices = list(range(99))
    selected = included_timestamp_indices[
        args.timestamp_shard_index :: args.timestamp_shard_count
    ]
    term_weight = 1.0 / (99.0 * float(DAS_NUM_MC))
    root = tpna_das_shard_root(
        args.timestamp_shard_index, args.timestamp_shard_count
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
    shape = (len(records), N_TRAIN)
    if all(path.is_file() for path in partial_paths.values()) and progress_path.is_file():
        with open(progress_path) as handle:
            progress = json.load(handle)
        if int(progress.get("contract_version", 0)) != CONTRACT_VERSION:
            raise ValueError("partial contract changed")
        if int(progress["feature_batch_size"]) != args.feature_batch_size:
            raise ValueError("partial feature batch size differs")
        completed_timestamps = [
            int(value) for value in progress["completed_timestamps"]
        ]
        scores = {
            lam: torch.from_numpy(np.load(path)).to(device, torch.float64)
            for lam, path in partial_paths.items()
        }
        print(f"[resume] timestamps={len(completed_timestamps)}", flush=True)
    else:
        scores = {
            float(lam): torch.zeros(shape, device=device, dtype=torch.float64)
            for lam in DAS_LAMBDAS
        }
    remaining = [
        index for index in selected if index not in set(completed_timestamps)
    ]
    dimension = int(TPNA_DAS_PROJ_DIM)
    eye = torch.eye(dimension, device=device, dtype=torch.float32)
    total_terms = len(remaining) * len(records) * int(DAS_NUM_MC)
    completed_terms = 0
    started = time.perf_counter()
    print(
        f"[tpna-das gpu={args.gpu}] timestamps={len(selected)}/99 "
        f"queries=10 probes={DAS_NUM_MC} feature_batch={args.feature_batch_size} "
        f"projection={dimension} final_ema=True",
        flush=True,
    )

    for shard_position, timestamp_index in enumerate(remaining, start=1):
        timestep = int(trajectory_timesteps[timestamp_index])
        if timestep <= 0:
            raise ValueError("endpoint t=0 must be excluded")
        t_query = torch.tensor([timestep], device=device, dtype=torch.long)
        for query_position, (record, trajectory, query_condition) in enumerate(
            zip(records, trajectories, query_conditions)
        ):
            query_state = torch.from_numpy(
                np.asarray(trajectory[timestamp_index]).copy()
            ).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                query_predicted_noise = model(
                    query_state, t_query, query_condition
                ).detach()
            target_noise = query_predicted_noise[0]

            for probe_index in range(int(DAS_NUM_MC)):
                term_started = time.perf_counter()
                specs = build_countsketch_specs(
                    list(active),
                    dimension,
                    device=device,
                    seed_parts=(
                        TPNA_DAS_SEED,
                        "parameter_projection",
                        timestamp_index,
                        probe_index,
                    ),
                )
                probe_generator = make_torch_generator(
                    device,
                    TPNA_DAS_SEED,
                    "output_probe",
                    timestamp_index,
                    probe_index,
                )
                output_probe = sample_output_probe(
                    tuple(query_state.shape), device=device, rng=probe_generator
                )
                probe_single = output_probe[0]
                scalar_denominator = math.sqrt(float(probe_single.numel()))

                query_prediction = model(query_state, t_query, query_condition)
                query_scalar = (
                    (query_prediction * output_probe).sum()
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

                def training_scalar(parameter_dict, x0, condition):
                    xt = base.q_sample(
                        x0.unsqueeze(0), t_query, target_noise, schedule
                    )
                    prediction = functional_call(
                        model,
                        parameter_dict,
                        (xt, t_query, condition.unsqueeze(0)),
                    )
                    return (
                        (prediction * output_probe).sum()
                        / scalar_denominator
                    )

                batched_gradient = vmap(
                    grad(training_scalar), in_dims=(None, 0, 0)
                )
                feature_cache = torch.empty(
                    (N_TRAIN, dimension), device=device, dtype=torch.float32
                )
                residual_cache = torch.empty(
                    N_TRAIN, device=device, dtype=torch.float32
                )
                gram = torch.zeros(
                    (dimension, dimension), device=device, dtype=torch.float32
                )
                num_batches = math.ceil(N_TRAIN / args.feature_batch_size)
                for batch_position, start in enumerate(
                    range(0, N_TRAIN, args.feature_batch_size), start=1
                ):
                    end = min(start + args.feature_batch_size, N_TRAIN)
                    xb = x_all[start:end]
                    cb = cond_all[start:end]
                    gradients = batched_gradient(named, xb, cb)
                    feature = _project_batched_grads(
                        gradients,
                        names,
                        specs,
                        dimension,
                        bool(DAS_NORMALIZE_PROJECTED_GRADS),
                        1e-8,
                    )
                    feature_cache[start:end] = feature
                    gram.addmm_(feature.T, feature)
                    with torch.no_grad():
                        batch_size = end - start
                        xt = base.q_sample(
                            xb,
                            t_query.expand(batch_size),
                            target_noise.expand(batch_size, -1, -1, -1),
                            schedule,
                        )
                        prediction = model(
                            xt, t_query.expand(batch_size), cb
                        )
                        residual_cache[start:end] = (
                            ((prediction - target_noise) * probe_single)
                            .reshape(batch_size, -1)
                            .sum(dim=1)
                            / scalar_denominator
                        )
                    if batch_position in (1, num_batches):
                        print(
                            f"[tpna-das gpu={args.gpu}] t="
                            f"{shard_position}/{len(remaining)} q="
                            f"{record['query_id']:02d} probe="
                            f"{probe_index+1}/{DAS_NUM_MC} batch="
                            f"{batch_position}/{num_batches}",
                            flush=True,
                        )

                for lam_raw in DAS_LAMBDAS:
                    lam = float(lam_raw)
                    solved_query = torch.linalg.solve(
                        gram + lam * eye, query_feature
                    )
                    raw = residual_cache * (feature_cache @ solved_query)
                    scores[lam][query_position] += (
                        term_weight * raw.double().square()
                    )

                completed_terms += 1
                elapsed = time.perf_counter() - started
                eta = elapsed / completed_terms * (total_terms - completed_terms)
                print(
                    f"[tpna-das gpu={args.gpu}] term="
                    f"{completed_terms}/{total_terms} global_t="
                    f"{timestamp_index+1}/99 q={record['query_id']:02d} "
                    f"elapsed={elapsed/3600:.2f}h eta={eta/3600:.2f}h",
                    flush=True,
                )
                del specs, output_probe, query_prediction, query_scalar
                del query_gradient, query_gradient_dict, query_feature
                del batched_gradient, feature_cache, residual_cache, gram
                del gradients, feature, prediction, raw, solved_query, xt
                torch.cuda.empty_cache()

        completed_timestamps.append(timestamp_index)
        completed_timestamps.sort()
        for lam, path in partial_paths.items():
            atomic_numpy(path, scores[lam].cpu().numpy())
        atomic_json(
            progress_path,
            {
                "contract_version": CONTRACT_VERSION,
                "feature_batch_size": args.feature_batch_size,
                "completed_timestamps": completed_timestamps,
                "query_ids": list(TPNA_DAS_QUERY_IDS),
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
            "method": TPNA_DAS_METHOD,
            "query_ids": list(TPNA_DAS_QUERY_IDS),
            "timestamp_indices": selected,
            "trajectory_timesteps": list(trajectory_timesteps),
            "endpoint_excluded": True,
            "term_weight": term_weight,
            "outer_probe_count": int(DAS_NUM_MC),
            "parameter_source": "final EMA",
            "checkpoint_epoch": int(checkpoint.get("epoch", EPOCHS)),
            "projection_dim": dimension,
            "normalize_features": bool(DAS_NORMALIZE_PROJECTED_GRADS),
            "sm_denominator": False,
            "feature_batch_size": args.feature_batch_size,
            "train_loss_is_query_dependent": True,
            "alignment_definition": (
                "At cached query state x_t, detach eps_q=eps_theta(x_t,t,c_q); "
                "construct every training state as sqrt(alpha_bar_t)*x_i + "
                "sqrt(1-alpha_bar_t)*eps_q and use eps_q as its residual target"
            ),
        },
    )
    print(f"[done] {root}", flush=True)


if __name__ == "__main__":
    main()
