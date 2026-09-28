"""Final-EMA DAS with exactly aligned query/train noise for q00-q98."""

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
from exp_config import *
from tracin_das_config import TRACIN_DAS_FIRST99_QUERY_IDS
from x3_endpoint_das_jax_logic_pytorch import (
    build_countsketch_specs,
    compute_projected_eps_feature,
    make_torch_generator,
    sample_noise,
    sample_output_probe,
)


METHOD = "das_ema_aligned_noise"
SHARD_NAMESPACE = "_das_ema_aligned_noise_99q_shards"
CONTRACT_VERSION = 1


def tag(value):
    return str(float(value)).replace(".", "p")


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
    parser.add_argument("--timestamp-shard-count", type=int, default=2)
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
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    records = [
        by_id[query_id]
        for query_id in TRACIN_DAS_FIRST99_QUERY_IDS
        if by_id[query_id]["family"] == args.family
    ]
    query_ids = [int(record["query_id"]) for record in records]
    if not records:
        raise ValueError(f"no q00-q98 records for family={args.family}")

    shard_root = (
        ATTR_DIR
        / SHARD_NAMESPACE
        / args.family
        / f"shard_{args.timestamp_shard_index:02d}_of_{args.timestamp_shard_count:02d}"
    )
    done_path = shard_root / "done.json"
    if done_path.is_file():
        print(f"[skip] {done_path}", flush=True)
        return

    final_checkpoint = model_paths(args.family)[-1]
    model, dataset, _ = build_model(final_checkpoint, "ema", device)
    schedule = base.make_linear_schedule(T, device=device)
    x_all, cond_all = preload_dataset(dataset, args.family, device)
    endpoints = [
        torch.from_numpy(np.load(Path(record["dir"]) / "final_state.npy")).to(
            device=device, dtype=torch.float32
        )
        for record in records
    ]
    conditions = [cond_for(record, dataset, device) for record in records]
    named = dict(model.named_parameters())
    names = tuple(named)
    active = tuple(named.values())
    dimension = int(DAS_PROJ_DIM)
    normalize = bool(DAS_NORMALIZE_PROJECTED_GRADS)
    normalize_eps = 1e-8
    selected_timestamp_indices = list(
        range(
            args.timestamp_shard_index,
            len(DAS_TIMESTEPS),
            args.timestamp_shard_count,
        )
    )
    scores = {
        float(lam): torch.zeros(
            (len(records), N_TRAIN), device=device, dtype=torch.float64
        )
        for lam in DAS_LAMBDAS
    }
    total_terms = len(selected_timestamp_indices) * int(DAS_NUM_MC)
    completed_terms = 0
    started = time.perf_counter()
    print(
        f"[das-aligned gpu={args.gpu}] family={args.family} "
        f"queries={query_ids[0]}..{query_ids[-1]} ({len(query_ids)}) "
        f"timestamps={len(selected_timestamp_indices)}/100 mc={DAS_NUM_MC} "
        f"batch={args.batch_size} projection={dimension}",
        flush=True,
    )

    for timestamp_index in selected_timestamp_indices:
        timestep = int(DAS_TIMESTEPS[timestamp_index])
        t_query = torch.tensor([timestep], device=device, dtype=torch.long)
        for mc_index in range(int(DAS_NUM_MC)):
            specs = build_countsketch_specs(
                list(active),
                dimension,
                device=device,
                seed_parts=(808, "pdas_gradient_projection", 0, timestep, mc_index),
            )
            probe_generator = make_torch_generator(
                device, 808, "pdas_output_probe", 0, timestep, mc_index
            )
            output_probe = sample_output_probe(
                tuple(endpoints[0].shape), device=device, rng=probe_generator
            )
            noise_generator = make_torch_generator(
                device, 808, "pdas_q", 0, timestep, mc_index
            )
            aligned_noise = sample_noise(endpoints[0], rng=noise_generator)

            query_features = []
            for endpoint, condition in zip(endpoints, conditions):
                _, feature = compute_projected_eps_feature(
                    model=model,
                    active=list(active),
                    sched=schedule,
                    x0=endpoint,
                    cond=condition,
                    t=t_query,
                    noise=aligned_noise,
                    output_probe=output_probe,
                    projection_specs=specs,
                    proj_dim=dimension,
                    device=device,
                    normalize_projected_grads=normalize,
                    normalize_eps=normalize_eps,
                )
                query_features.append(feature.to(torch.float32).detach())
            query_features = torch.stack(query_features)

            probe_single = output_probe[0]
            scalar_denominator = math.sqrt(float(probe_single.numel()))

            def single_scalar(parameter_dict, x0, condition):
                xt = base.q_sample(
                    x0.unsqueeze(0), t_query, aligned_noise, schedule
                )
                prediction = functional_call(
                    model,
                    parameter_dict,
                    (xt, t_query, condition.unsqueeze(0)),
                )
                return (prediction * probe_single).sum() / scalar_denominator

            batched_gradient = vmap(grad(single_scalar), in_dims=(None, 0, 0))
            feature_cache = torch.empty(
                (N_TRAIN, dimension), device=device, dtype=torch.float32
            )
            residual_cache = torch.empty(
                N_TRAIN, device=device, dtype=torch.float32
            )
            gram = torch.zeros(
                (dimension, dimension), device=device, dtype=torch.float32
            )
            num_batches = math.ceil(N_TRAIN / args.batch_size)
            progress_every = max(1, num_batches // 5)
            for batch_position, start in enumerate(
                range(0, N_TRAIN, args.batch_size), start=1
            ):
                end = min(start + args.batch_size, N_TRAIN)
                xb = x_all[start:end]
                cb = cond_all[start:end]
                gradients = batched_gradient(named, xb, cb)
                features = _project_batched_grads(
                    gradients,
                    names,
                    specs,
                    dimension,
                    normalize,
                    normalize_eps,
                )
                feature_cache[start:end] = features
                gram.addmm_(features.T, features)

                batch_noise = aligned_noise.expand(end - start, *aligned_noise.shape[1:])
                t_batch = torch.full(
                    (end - start,), timestep, device=device, dtype=torch.long
                )
                with torch.no_grad():
                    prediction = model(
                        base.q_sample(xb, t_batch, batch_noise, schedule),
                        t_batch,
                        cb,
                    )
                    residual_cache[start:end] = (
                        ((prediction - batch_noise) * probe_single)
                        .reshape(end - start, -1)
                        .sum(dim=1)
                        / scalar_denominator
                    )
                if (
                    batch_position == 1
                    or batch_position % progress_every == 0
                    or batch_position == num_batches
                ):
                    print(
                        f"[das-aligned gpu={args.gpu}] term={completed_terms + 1}/{total_terms} "
                        f"t={timestamp_index + 1}/100 mc={mc_index + 1}/{DAS_NUM_MC} "
                        f"batch={batch_position}/{num_batches}",
                        flush=True,
                    )

            eye = torch.eye(dimension, device=device, dtype=torch.float32)
            for lam_raw in DAS_LAMBDAS:
                lam = float(lam_raw)
                solved_queries = torch.linalg.solve(
                    gram + lam * eye, query_features.T
                )
                raw = (feature_cache @ solved_queries).to(torch.float64)
                raw *= residual_cache.to(torch.float64).unsqueeze(1)
                if DAS_USE_SM_DENOMINATOR:
                    solved_train = torch.linalg.solve(
                        gram + lam * eye, feature_cache.T
                    ).T
                    denominator = 1.0 - (
                        feature_cache * solved_train
                    ).sum(dim=1).to(torch.float64)
                    denominator = torch.where(
                        denominator.abs() < 1e-6,
                        denominator.sign() * 1e-6,
                        denominator,
                    )
                    raw /= denominator.unsqueeze(1)
                scores[lam] += raw.T.square()

            completed_terms += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / completed_terms * (total_terms - completed_terms)
            print(
                f"[das-aligned gpu={args.gpu}] completed={completed_terms}/{total_terms} "
                f"elapsed={elapsed/3600:.2f}h eta={eta/3600:.2f}h",
                flush=True,
            )
            del (
                specs,
                output_probe,
                aligned_noise,
                query_features,
                feature_cache,
                residual_cache,
                gram,
                eye,
                solved_queries,
                raw,
                gradients,
                features,
                batched_gradient,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for lam, values in scores.items():
        atomic_numpy(shard_root / f"lambda_{tag(lam)}.npy", values.cpu().numpy())
    atomic_json(
        done_path,
        {
            "contract_version": CONTRACT_VERSION,
            "method": METHOD,
            "family": args.family,
            "query_ids": query_ids,
            "timestamp_indices": selected_timestamp_indices,
            "timestamps": [
                int(DAS_TIMESTEPS[index]) for index in selected_timestamp_indices
            ],
            "num_mc_per_timestamp": int(DAS_NUM_MC),
            "term_count": total_terms,
            "parameter_source": "ema",
            "projection_dim": dimension,
            "normalize_projected_grads": normalize,
            "noise_alignment": (
                "same noise per (timestamp,mc) for every query and every "
                "training point feature/residual"
            ),
            "batch_size": args.batch_size,
        },
    )
    print(f"[done] {shard_root}", flush=True)


if __name__ == "__main__":
    main()
