"""One shard of trajectory-state relative-forward aligned DAS terms."""

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
    _project_batched_grads,
    build_model,
    cond_for,
    model_paths,
    preload_dataset,
)
from dataset_loader import ColorGridDataset
from multiclean_das_config import *
from x3_endpoint_das_jax_logic_pytorch import (
    build_countsketch_specs,
    make_torch_generator,
    sample_noise,
    sample_output_probe,
)


CONTRACT_VERSION = 2


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
    parser.add_argument("--family", choices=MULTICLEAN_FAMILIES, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--timestamp-shard-index", type=int, required=True)
    parser.add_argument("--timestamp-shard-count", type=int, default=2)
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
    query_ids = multiclean_query_ids(args.family)
    records = [by_id[query_id] for query_id in query_ids]
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    model, _, _ = build_model(model_paths(args.family)[-1], "ema", device)
    named = dict(model.named_parameters())
    names = tuple(named)
    active = tuple(named.values())
    schedule = base.make_linear_schedule(T, device=device)
    x_all, cond_all = preload_dataset(dataset, args.family, device)
    conditions = [cond_for(record, dataset, device) for record in records]
    if args.family == "unprompted":
        conditions = [torch.zeros_like(value) for value in conditions]
    condition_bank = torch.cat(conditions, dim=0)

    states_np = np.load(MULTICLEAN_CACHE_DIR / "trajectory_states.npy")
    expected_shape = (
        MULTICLEAN_ANCHOR_COUNT,
        len(MULTICLEAN_QUERY_IDS),
        *x_all.shape[1:],
    )
    if states_np.shape != expected_shape:
        raise ValueError(f"trajectory-state shape={states_np.shape}, expected={expected_shape}")
    states = torch.from_numpy(states_np).to(device=device, dtype=torch.float32)
    with open(MULTICLEAN_CACHE_DIR / "info.json") as handle:
        cache_info = json.load(handle)
    anchor_timesteps = tuple(int(value) for value in cache_info["anchor_timesteps"])
    if len(anchor_timesteps) != MULTICLEAN_ANCHOR_COUNT:
        raise ValueError("anchor timestep count differs")
    anchor_targets = tuple(
        multiclean_anchor_targets(anchor_timestep, target_count)
        for anchor_timestep, target_count in zip(
            anchor_timesteps, MULTICLEAN_ANCHOR_DAS_COUNTS
        )
    )
    all_pairs = tuple(
        (anchor_index, target_position, int(timestep))
        for anchor_index, targets in enumerate(anchor_targets)
        for target_position, timestep in enumerate(targets)
    )
    selected_pair_indices = list(
        range(
            args.timestamp_shard_index,
            len(all_pairs),
            args.timestamp_shard_count,
        )
    )
    root = multiclean_shard_root(
        args.family, args.timestamp_shard_index, args.timestamp_shard_count
    )
    done_path = root / "done.json"
    if done_path.is_file():
        print(f"[skip] {done_path}", flush=True)
        return

    scores = {
        float(lam): torch.zeros(
            (len(query_ids), N_TRAIN), device=device, dtype=torch.float64
        )
        for lam in DAS_LAMBDAS
    }
    dimension = int(DAS_PROJ_DIM)
    normalize = bool(DAS_NORMALIZE_PROJECTED_GRADS)
    normalize_eps = 1e-8
    total_terms = len(selected_pair_indices) * int(DAS_NUM_MC)
    completed_terms = 0
    started = time.perf_counter()
    print(
        f"[relative-forward-das gpu={args.gpu}] family={args.family} "
        f"queries=q{query_ids[0]:02d}-q{query_ids[-1]:02d} "
        f"anchors={MULTICLEAN_ANCHOR_COUNT} pairs={len(selected_pair_indices)}/"
        f"{len(all_pairs)} mc={DAS_NUM_MC} batch={args.batch_size} "
        f"projection={dimension} aligned=true initial_t999_skipped=true",
        flush=True,
    )

    for pair_index in selected_pair_indices:
        anchor_index, target_position, timestep = all_pairs[pair_index]
        anchor_timestep = anchor_timesteps[anchor_index]
        if timestep < anchor_timestep:
            raise ValueError((anchor_timestep, timestep))
        t_query = torch.tensor([timestep], device=device, dtype=torch.long)
        alpha_ratio = schedule.alpha_bars[timestep] / schedule.alpha_bars[
            anchor_timestep
        ]
        alpha_ratio = torch.clamp(alpha_ratio, min=0.0, max=1.0)
        query_scale = torch.sqrt(alpha_ratio)
        query_noise_scale = torch.sqrt(torch.clamp(1.0 - alpha_ratio, min=0.0))
        anchor_x = states[anchor_index, list(query_ids)]
        target_count = len(anchor_targets[anchor_index])

        for mc_index in range(int(DAS_NUM_MC)):
            seed_parts = (
                808,
                "relative_forward",
                anchor_index,
                target_position,
                anchor_timestep,
                timestep,
                mc_index,
            )
            specs = build_countsketch_specs(
                list(active), dimension, device=device, seed_parts=seed_parts
            )
            probe_generator = make_torch_generator(
                device, *seed_parts, "output_probe"
            )
            output_probe = sample_output_probe(
                tuple(anchor_x[0].unsqueeze(0).shape),
                device=device,
                rng=probe_generator,
            )
            noise_generator = make_torch_generator(device, *seed_parts, "noise")
            aligned_noise = sample_noise(
                anchor_x[0].unsqueeze(0), rng=noise_generator
            )
            query_xt = (
                query_scale * anchor_x
                + query_noise_scale * aligned_noise.expand_as(anchor_x)
            )
            probe_single = output_probe[0]
            scalar_denominator = math.sqrt(float(probe_single.numel()))

            def query_scalar(parameter_dict, xt, condition):
                prediction = functional_call(
                    model,
                    parameter_dict,
                    (xt.unsqueeze(0), t_query, condition.unsqueeze(0)),
                )
                return (prediction * probe_single).sum() / scalar_denominator

            def train_scalar(parameter_dict, x0, condition):
                xt = base.q_sample(
                    x0.unsqueeze(0), t_query, aligned_noise, schedule
                )
                prediction = functional_call(
                    model,
                    parameter_dict,
                    (xt, t_query, condition.unsqueeze(0)),
                )
                return (prediction * probe_single).sum() / scalar_denominator

            query_gradient = vmap(grad(query_scalar), in_dims=(None, 0, 0))
            train_gradient = vmap(grad(train_scalar), in_dims=(None, 0, 0))
            query_feature_parts = []
            for query_start in range(0, len(query_ids), 25):
                query_end = min(query_start + 25, len(query_ids))
                query_gradients = query_gradient(
                    named,
                    query_xt[query_start:query_end],
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
            residual_cache = torch.empty(N_TRAIN, device=device, dtype=torch.float32)
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
                gradients = train_gradient(named, xb, cb)
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
                        f"[relative-forward-das gpu={args.gpu}] "
                        f"term={completed_terms + 1}/{total_terms} "
                        f"anchor_t={anchor_timestep} target_t={timestep} "
                        f"target={target_position + 1}/{target_count} "
                        f"mc={mc_index + 1}/{DAS_NUM_MC} "
                        f"batch={batch_position}/{num_batches}",
                        flush=True,
                    )

            eye = torch.eye(dimension, device=device, dtype=torch.float32)
            term_weight = 1.0 / (float(target_count) * float(DAS_NUM_MC))
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
                scores[lam] += raw.T.square() * term_weight

            completed_terms += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / completed_terms * (total_terms - completed_terms)
            print(
                f"[relative-forward-das gpu={args.gpu}] "
                f"completed={completed_terms}/{total_terms} "
                f"elapsed={elapsed/3600:.2f}h eta={eta/3600:.2f}h",
                flush=True,
            )
            del specs, output_probe, aligned_noise, query_xt, query_features
            del query_feature_parts, query_gradients, feature_cache, residual_cache
            del gram, eye, solved_queries, raw, gradients, features
            del query_gradient, train_gradient
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
            "method": MULTICLEAN_METHOD,
            "family": args.family,
            "query_ids": list(query_ids),
            "anchor_indices": list(MULTICLEAN_ANCHOR_INDICES),
            "anchor_timesteps": list(anchor_timesteps),
            "anchor_target_timesteps": [list(values) for values in anchor_targets],
            "anchor_das_timestamp_counts": list(MULTICLEAN_ANCHOR_DAS_COUNTS),
            "anchor_count": MULTICLEAN_ANCHOR_COUNT,
            "initial_t999_anchor": "skipped",
            "anchor_reduction": (
                "sum of independently target-and-MC-averaged per-anchor "
                "squared DAS scores"
            ),
            "selected_pair_indices": selected_pair_indices,
            "pair_count": len(selected_pair_indices),
            "num_mc_per_target": int(DAS_NUM_MC),
            "term_count": total_terms,
            "parameter_source": "final EMA",
            "projection_dim": dimension,
            "normalize_projected_grads": normalize,
            "relative_forward_formula": (
                "x_s=sqrt(alpha_bar_s/alpha_bar_t)*x_t+"
                "sqrt(1-alpha_bar_s/alpha_bar_t)*noise"
            ),
            "noise_alignment": (
                "same noise for query relative-forward transition and training "
                "q_sample/loss at each (anchor,target,MC)"
            ),
            "batch_size": args.batch_size,
        },
    )
    print(f"[done] {root}", flush=True)


if __name__ == "__main__":
    main()
