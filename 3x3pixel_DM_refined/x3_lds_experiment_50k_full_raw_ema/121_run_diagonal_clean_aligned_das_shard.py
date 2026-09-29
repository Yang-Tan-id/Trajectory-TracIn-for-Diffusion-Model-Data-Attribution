"""Run a shard of timestamp-diagonal predicted-clean aligned DAS."""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

import x3pixel_DM_training as base
from attribution_one_query import _project_batched_grads, build_model, cond_for, model_paths, preload_dataset
from dataset_loader import ColorGridDataset
from diagonal_clean_das_config import *
from x3_endpoint_das_jax_logic_pytorch import (
    build_countsketch_specs,
    make_torch_generator,
    sample_noise,
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
    parser.add_argument("--family", choices=DIAGONAL_CLEAN_FAMILIES, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--timestamp-shard-index", type=int, required=True)
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=DAS_FEATURE_BATCH_SIZE)
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
    query_ids = diagonal_clean_query_ids(args.family)
    records = [by_id[query_id] for query_id in query_ids]
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    model, _, _ = build_model(
        model_paths(args.family)[-1], "ema", device
    )
    named = dict(model.named_parameters())
    names = tuple(named)
    active = tuple(named.values())
    schedule = base.make_linear_schedule(T, device=device)
    x_all, cond_all = preload_dataset(dataset, args.family, device)
    conditions = [cond_for(record, dataset, device) for record in records]
    if args.family == "unprompted":
        conditions = [torch.zeros_like(value) for value in conditions]
    condition_bank = torch.cat(conditions, dim=0)
    clean_np = np.load(DIAGONAL_CLEAN_CACHE_DIR / "predicted_clean.npy")
    timestamps = np.load(DIAGONAL_CLEAN_CACHE_DIR / "trajectory_t.npy")
    expected_shape = (100, len(DIAGONAL_CLEAN_QUERY_IDS), *x_all.shape[1:])
    if clean_np.shape != expected_shape or timestamps.shape != (100,):
        raise ValueError(
            f"cache mismatch: clean={clean_np.shape}, t={timestamps.shape}, "
            f"expected={expected_shape}/(100,)"
        )
    clean = torch.from_numpy(clean_np).to(device=device, dtype=torch.float32)
    selected = list(
        range(args.timestamp_shard_index, 100, args.timestamp_shard_count)
    )
    root = diagonal_clean_shard_root(
        args.family, args.timestamp_shard_index, args.timestamp_shard_count
    )
    done_path = root / "done.json"
    if done_path.is_file():
        print(f"[skip] {done_path}", flush=True)
        return
    scores = {
        float(lam): torch.zeros(
            (len(query_ids), N_TRAIN),
            device=device,
            dtype=torch.float64,
        )
        for lam in DAS_LAMBDAS
    }
    dimension = int(DAS_PROJ_DIM)
    normalize = bool(DAS_NORMALIZE_PROJECTED_GRADS)
    normalize_eps = 1e-8
    total_terms = len(selected) * int(DAS_NUM_MC)
    completed_terms = 0
    started = time.perf_counter()
    print(
        f"[diagonal-clean-das gpu={args.gpu}] family={args.family} "
        f"queries=q{query_ids[0]:02d}-q{query_ids[-1]:02d} timestamps="
        f"{len(selected)}/100 mc={DAS_NUM_MC} batch={args.batch_size} "
        f"projection={dimension} aligned=true one_endpoint_per_noise_level=true",
        flush=True,
    )

    for snapshot_index in selected:
        timestep = int(timestamps[snapshot_index])
        t_query = torch.tensor([timestep], device=device, dtype=torch.long)
        for mc_index in range(int(DAS_NUM_MC)):
            specs = build_countsketch_specs(
                list(active),
                dimension,
                device=device,
                seed_parts=(809, "diagonal_clean_gradient_projection", timestep, mc_index),
            )
            probe_generator = make_torch_generator(
                device, 809, "diagonal_clean_output_probe", timestep, mc_index
            )
            output_probe = sample_output_probe(
                tuple(clean[snapshot_index, query_ids[0]].unsqueeze(0).shape),
                device=device,
                rng=probe_generator,
            )
            noise_generator = make_torch_generator(
                device, 809, "diagonal_clean_aligned_noise", timestep, mc_index
            )
            aligned_noise = sample_noise(
                clean[snapshot_index, query_ids[0]].unsqueeze(0), rng=noise_generator
            )
            probe_single = output_probe[0]
            scalar_denominator = math.sqrt(float(probe_single.numel()))

            def single_scalar(parameter_dict, x0, condition):
                xt = base.q_sample(x0.unsqueeze(0), t_query, aligned_noise, schedule)
                prediction = functional_call(
                    model,
                    parameter_dict,
                    (xt, t_query, condition.unsqueeze(0)),
                )
                return (prediction * probe_single).sum() / scalar_denominator

            batched_gradient = vmap(grad(single_scalar), in_dims=(None, 0, 0))
            query_feature_parts = []
            for query_start in range(0, len(query_ids), 25):
                query_end = min(query_start + 25, len(query_ids))
                query_gradients = batched_gradient(
                    named,
                    clean[
                        snapshot_index,
                        list(query_ids[query_start:query_end]),
                    ],
                    condition_bank[query_start:query_end],
                )
                query_feature_parts.append(
                    _project_batched_grads(
                        query_gradients,
                        names,
                        specs,
                        dimension,
                        normalize,
                        normalize_eps,
                    )
                )
            query_features = torch.cat(query_feature_parts, dim=0)
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
                batch_noise = aligned_noise.expand(
                    end - start, *aligned_noise.shape[1:]
                )
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
                        f"[diagonal-clean-das gpu={args.gpu}] "
                        f"term={completed_terms+1}/{total_terms} "
                        f"snapshot={snapshot_index+1}/100 t={timestep} "
                        f"mc={mc_index+1}/{DAS_NUM_MC} "
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
                f"[diagonal-clean-das gpu={args.gpu}] "
                f"completed={completed_terms}/{total_terms} "
                f"elapsed={elapsed/3600:.2f}h eta={eta/3600:.2f}h",
                flush=True,
            )
            del specs, output_probe, aligned_noise, query_features
            del query_feature_parts, query_gradients
            del feature_cache, residual_cache, gram, eye, solved_queries, raw
            del gradients, features, batched_gradient
            if DAS_USE_SM_DENOMINATOR:
                del solved_train, denominator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for lam, values in scores.items():
        atomic_numpy(root / f"lambda_{lambda_tag(lam)}.npy", values.cpu().numpy())
    atomic_json(
        done_path,
        {
            "contract_version": CONTRACT_VERSION,
            "method": DIAGONAL_CLEAN_METHOD,
            "family": args.family,
            "query_ids": list(query_ids),
            "snapshot_indices": selected,
            "timestamps": [int(timestamps[index]) for index in selected],
            "num_mc_per_timestamp": int(DAS_NUM_MC),
            "term_count": total_terms,
            "pairing": "predicted clean from snapshot k scored only at noise level k",
            "parameter_source": "final EMA",
            "projection_dim": dimension,
            "normalize_projected_grads": normalize,
            "noise_alignment": (
                "same noise per (snapshot timestamp,MC) for query clean endpoint "
                "and all training-point feature/residual calculations"
            ),
            "batch_size": args.batch_size,
        },
    )
    print(f"[done] {root}", flush=True)


if __name__ == "__main__":
    main()
