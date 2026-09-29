"""Score one timestamp/MC shard for fully-unrolled trajectory DAS."""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

import x3pixel_DM_training as base
from attribution_one_query import _project_batched_grads, build_model, model_paths, preload_dataset
from dataset_loader import ColorGridDataset
from unrolled_traj_das_config import *
from x3_endpoint_das_jax_logic_pytorch import (
    build_countsketch_specs,
    make_torch_generator,
    sample_output_probe,
)


def atomic_numpy(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=DAS_FEATURE_BATCH_SIZE)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    model, _, _ = build_model(
        model_paths(UNROLLED_TRAJ_DAS_FAMILY)[-1], "ema", device
    )
    named = dict(model.named_parameters())
    names = tuple(named)
    active = tuple(named.values())
    specs = build_countsketch_specs(
        list(active),
        UNROLLED_TRAJ_DAS_PROJECTION_DIM,
        device=device,
        seed_parts=UNROLLED_TRAJ_DAS_PROJECTION_SEED,
    )
    schedule = base.make_linear_schedule(T, device=device)
    x_all, cond_all = preload_dataset(dataset, UNROLLED_TRAJ_DAS_FAMILY, device)
    query_features_np = np.load(
        UNROLLED_TRAJ_DAS_CACHE_DIR / "query_features.npy"
    )
    with open(UNROLLED_TRAJ_DAS_CACHE_DIR / "info.json") as handle:
        query_info = json.load(handle)
    if query_info["method"] != UNROLLED_TRAJ_DAS_METHOD:
        raise ValueError("query-feature method mismatch")
    if query_info["projection_seed"] != list(UNROLLED_TRAJ_DAS_PROJECTION_SEED):
        raise ValueError("query-feature projection seed mismatch")
    expected_query_shape = (
        len(UNROLLED_TRAJ_DAS_QUERY_IDS),
        UNROLLED_TRAJ_DAS_PROBES,
        UNROLLED_TRAJ_DAS_PROJECTION_DIM,
    )
    if query_features_np.shape != expected_query_shape:
        raise ValueError(
            f"query feature shape={query_features_np.shape}, "
            f"expected={expected_query_shape}"
        )
    query_features = torch.from_numpy(query_features_np).to(
        device=device, dtype=torch.float32
    ).reshape(-1, UNROLLED_TRAJ_DAS_PROJECTION_DIM)

    all_terms = tuple(
        (timestamp_index, mc_index)
        for timestamp_index in range(len(DAS_TIMESTEPS))
        for mc_index in range(int(DAS_NUM_MC))
    )
    selected_indices = list(range(args.shard_index, len(all_terms), args.shard_count))
    root = unrolled_traj_das_shard_root(args.shard_index, args.shard_count)
    done_path = root / "done.json"
    if done_path.is_file():
        print(f"[skip] {done_path}", flush=True)
        return
    scores = {
        float(lam): torch.zeros(
            (len(UNROLLED_TRAJ_DAS_QUERY_IDS), N_TRAIN),
            device=device,
            dtype=torch.float64,
        )
        for lam in DAS_LAMBDAS
    }
    dimension = int(UNROLLED_TRAJ_DAS_PROJECTION_DIM)
    train_mc = int(DAS_TRAIN_GRAD_MC)
    total_global_terms = len(all_terms)
    completed = 0
    started = time.perf_counter()
    print(
        f"[unrolled-traj-das gpu={args.gpu}] shard={args.shard_index}/"
        f"{args.shard_count} terms={len(selected_indices)}/{len(all_terms)} "
        f"queries={len(UNROLLED_TRAJ_DAS_QUERY_IDS)} probes="
        f"{UNROLLED_TRAJ_DAS_PROBES} train_mc={train_mc} "
        f"batch={args.batch_size} projection={dimension}",
        flush=True,
    )

    for term_index in selected_indices:
        timestamp_index, mc_index = all_terms[term_index]
        timestep = int(DAS_TIMESTEPS[timestamp_index])
        t_mc = torch.full(
            (train_mc,), timestep, device=device, dtype=torch.long
        )
        probe_generator = make_torch_generator(
            device, 811, "unrolled_train_output_probe", timestep, mc_index
        )
        output_probe = sample_output_probe(
            (1, *x_all.shape[1:]), device=device, rng=probe_generator
        )[0]
        scalar_denominator = math.sqrt(float(output_probe.numel()))

        def single_scalar(parameter_dict, x0, condition, noises):
            x_mc = x0.unsqueeze(0).expand(train_mc, *x0.shape)
            c_mc = condition.unsqueeze(0).expand(train_mc, condition.shape[-1])
            xt = base.q_sample(x_mc, t_mc, noises, schedule)
            prediction = functional_call(model, parameter_dict, (xt, t_mc, c_mc))
            values = (
                (prediction * output_probe)
                .reshape(train_mc, -1)
                .sum(dim=1)
                / scalar_denominator
            )
            return values.mean()

        batched_gradient = vmap(
            grad(single_scalar), in_dims=(None, 0, 0, 0)
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
        num_batches = math.ceil(N_TRAIN / args.batch_size)
        progress_every = max(1, num_batches // 5)
        for batch_position, start in enumerate(
            range(0, N_TRAIN, args.batch_size), start=1
        ):
            end = min(start + args.batch_size, N_TRAIN)
            xb = x_all[start:end]
            cb = cond_all[start:end]
            batch_size = end - start
            noise_generator = make_torch_generator(
                device,
                811,
                "unrolled_train_noise",
                timestep,
                mc_index,
                start,
            )
            noises = torch.randn(
                (batch_size, train_mc, *xb.shape[1:]),
                generator=noise_generator,
                device=device,
                dtype=xb.dtype,
            )
            gradients = batched_gradient(named, xb, cb, noises)
            features = _project_batched_grads(
                gradients,
                names,
                specs,
                dimension,
                bool(DAS_NORMALIZE_PROJECTED_GRADS),
                1e-8,
            )
            feature_cache[start:end] = features
            gram.addmm_(features.T, features)
            with torch.no_grad():
                xb_mc = xb[:, None].expand(
                    batch_size, train_mc, *xb.shape[1:]
                ).reshape(batch_size * train_mc, *xb.shape[1:])
                cb_mc = cb[:, None].expand(
                    batch_size, train_mc, cb.shape[-1]
                ).reshape(batch_size * train_mc, cb.shape[-1])
                noise_flat = noises.reshape(
                    batch_size * train_mc, *xb.shape[1:]
                )
                t_batch = torch.full(
                    (batch_size * train_mc,),
                    timestep,
                    device=device,
                    dtype=torch.long,
                )
                prediction = model(
                    base.q_sample(xb_mc, t_batch, noise_flat, schedule),
                    t_batch,
                    cb_mc,
                )
                residual_cache[start:end] = (
                    ((prediction - noise_flat) * output_probe)
                    .reshape(batch_size, train_mc, -1)
                    .sum(dim=2)
                    .mean(dim=1)
                    / scalar_denominator
                )
            if (
                batch_position == 1
                or batch_position % progress_every == 0
                or batch_position == num_batches
            ):
                print(
                    f"[unrolled-traj-das gpu={args.gpu}] "
                    f"term={completed + 1}/{len(selected_indices)} "
                    f"t={timestamp_index + 1}/100 mc={mc_index + 1}/"
                    f"{DAS_NUM_MC} batch={batch_position}/{num_batches}",
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
            per_query_probe = raw.reshape(
                N_TRAIN,
                len(UNROLLED_TRAJ_DAS_QUERY_IDS),
                UNROLLED_TRAJ_DAS_PROBES,
            )
            scores[lam] += (
                per_query_probe.square().mean(dim=2).T / total_global_terms
            )

        completed += 1
        elapsed = time.perf_counter() - started
        eta = elapsed / completed * (len(selected_indices) - completed)
        print(
            f"[unrolled-traj-das gpu={args.gpu}] completed={completed}/"
            f"{len(selected_indices)} elapsed={elapsed/3600:.2f}h "
            f"eta={eta/3600:.2f}h",
            flush=True,
        )
        del output_probe, feature_cache, residual_cache, gram, eye
        del gradients, features, noises, solved_queries, raw, per_query_probe
        del batched_gradient
        if DAS_USE_SM_DENOMINATOR:
            del solved_train, denominator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for lam, values in scores.items():
        atomic_numpy(root / f"lambda_{lambda_tag(lam)}.npy", values.cpu().numpy())
    root.mkdir(parents=True, exist_ok=True)
    with open(done_path, "w") as handle:
        json.dump(
            {
                "method": UNROLLED_TRAJ_DAS_METHOD,
                "query_ids": list(UNROLLED_TRAJ_DAS_QUERY_IDS),
                "selected_term_indices": selected_indices,
                "term_count": len(selected_indices),
                "global_term_count": total_global_terms,
                "das_timestamps": [int(value) for value in DAS_TIMESTEPS],
                "das_outer_mc": int(DAS_NUM_MC),
                "train_gradient_mc": train_mc,
                "trajectory_probe_count": int(UNROLLED_TRAJ_DAS_PROBES),
                "projection_dim": dimension,
                "global_projection": True,
                "projection_seed": list(UNROLLED_TRAJ_DAS_PROJECTION_SEED),
                "normalize_train_features": bool(DAS_NORMALIZE_PROJECTED_GRADS),
                "normalize_query_features": False,
                "batch_size": args.batch_size,
            },
            handle,
            indent=2,
        )
    print(f"[done] {root}", flush=True)


if __name__ == "__main__":
    main()
