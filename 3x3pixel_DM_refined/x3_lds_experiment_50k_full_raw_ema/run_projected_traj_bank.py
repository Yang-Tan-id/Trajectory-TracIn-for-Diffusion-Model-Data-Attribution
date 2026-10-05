"""Score a whole query family while computing each projected train gradient once."""

import argparse
import json
import math
import re
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
from exp_config import *
from x3_endpoint_das_jax_logic_pytorch import build_countsketch_specs, make_torch_generator


def output_methods(
    checkpoint_direction,
    first_order_only=False,
    output_suffix="",
    parameter_source="raw",
):
    prefix = (
        "traj_projected"
        if checkpoint_direction == "forward"
        else "traj_projected_backward"
    )
    orders = (
        ("first",)
        if first_order_only or checkpoint_direction == "backward"
        else ("first", "second")
    )
    suffix = f"_{output_suffix}" if output_suffix else ""
    return tuple(
        f"{prefix}_{order}_{parameter_source}_{contraction}{suffix}"
        for order in orders
        for contraction in TRACIN_CONTRACTIONS
    )


def family_complete(
    records,
    checkpoint_direction,
    first_order_only=False,
    output_suffix="",
    parameter_source="raw",
):
    return all(
        (ATTR_DIR / method / f"q{int(record['query_id']):02d}" / "scores.npy").is_file()
        for record in records
        for method in output_methods(
            checkpoint_direction, first_order_only, output_suffix, parameter_source
        )
    )


def parse_pair_indices(value, pair_count):
    if value is None:
        return list(range(pair_count))
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise ValueError("--checkpoint-pair-indices must contain at least one index")
    if len(indices) != len(set(indices)):
        raise ValueError("--checkpoint-pair-indices contains duplicates")
    if any(index < 0 or index >= pair_count for index in indices):
        raise ValueError(
            f"checkpoint pair indices must be in [0, {pair_count - 1}]"
        )
    return indices


def parse_query_ids(value):
    if value is None:
        return None
    query_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not query_ids:
        raise ValueError("--query-ids must contain at least one query ID")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("--query-ids contains duplicates")
    return query_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument(
        "--parameter-source", choices=("raw", "ema"), default="raw"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timestamp-shard-index", type=int, default=0)
    parser.add_argument("--timestamp-shard-count", type=int, default=1)
    parser.add_argument(
        "--checkpoint-direction",
        choices=("forward", "backward"),
        default="forward",
    )
    parser.add_argument(
        "--checkpoint-pair-indices",
        help="comma-separated positions in the direction-specific list of 49 pairs",
    )
    parser.add_argument("--first-order-only", action="store_true")
    parser.add_argument("--query-ids", help="comma-separated query IDs to score")
    parser.add_argument(
        "--exclude-endpoint",
        action="store_true",
        help="exclude trajectory index 99 (the final generated x0 state)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=TRACIN_PROJECTED_BATCH_SIZE
    )
    parser.add_argument(
        "--train-noise-sampling",
        choices=("independent", "antithetic"),
        default="independent",
        help="independent MC noise or paired +epsilon/-epsilon training noise",
    )
    parser.add_argument(
        "--train-mc-pairs",
        type=int,
        default=TRACIN_TRAIN_MC,
        help="number of base epsilon draws in antithetic mode (twice as many losses)",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="suffix added to shard namespaces and final attribution methods",
    )
    args = parser.parse_args()
    if args.output_suffix and not re.fullmatch(r"[A-Za-z0-9_]+", args.output_suffix):
        raise ValueError("--output-suffix may contain only letters, digits, and underscores")
    if args.timestamp_shard_count <= 0:
        raise ValueError("--timestamp-shard-count must be positive")
    if not 0 <= args.timestamp_shard_index < args.timestamp_shard_count:
        raise ValueError("timestamp shard index is outside the shard count")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.train_mc_pairs <= 0:
        raise ValueError("--train-mc-pairs must be positive")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    with open(QUERY_DIR / "manifest.json") as handle:
        records = [r for r in json.load(handle) if r["family"] == args.family]
    requested_query_ids = parse_query_ids(args.query_ids)
    if requested_query_ids is not None:
        by_id = {int(record["query_id"]): record for record in records}
        missing = [query_id for query_id in requested_query_ids if query_id not in by_id]
        if missing:
            raise ValueError(
                f"queries {missing} do not belong to family={args.family}"
            )
        records = [by_id[query_id] for query_id in requested_query_ids]
    if not records or (
        args.timestamp_shard_count == 1
        and family_complete(
            records,
            args.checkpoint_direction,
            args.first_order_only,
            args.output_suffix,
            args.parameter_source,
        )
    ):
        print(
            f"[skip] projected Traj {args.checkpoint_direction} bank complete "
            f"for {args.family}",
            flush=True,
        )
        return

    ds = ColorGridDataset(str(BASE_CSV), grid_size=3)
    paths = model_paths(args.family)
    sched = base.make_linear_schedule(T, device=device)
    x_all, cond_all = preload_dataset(ds, args.family, device)
    trajectories = [np.load(Path(r["dir"]) / "trajectory_xt.npy") for r in records]
    timestep_arrays = [np.load(Path(r["dir"]) / "trajectory_t.npy") for r in records]
    t_seq = timestep_arrays[0]
    if len(t_seq) != 100 or any(not np.array_equal(t_seq, values) for values in timestep_arrays):
        raise ValueError("all bank queries must share the same 100 trajectory timestamps")
    included_timestamp_indices = list(range(len(t_seq)))
    if args.exclude_endpoint:
        included_timestamp_indices.remove(len(t_seq) - 1)
    selected_timestamp_indices = included_timestamp_indices[
        args.timestamp_shard_index :: args.timestamp_shard_count
    ]
    shard_namespace = (
        "_projected_traj_shards"
        if args.checkpoint_direction == "forward"
        else "_projected_backward_traj_shards"
    )
    if args.output_suffix:
        shard_namespace += f"_{args.output_suffix}"
    if args.parameter_source != "raw":
        shard_namespace += f"_{args.parameter_source}"
    shard_root = (
        ATTR_DIR
        / shard_namespace
        / args.family
        / f"shard_{args.timestamp_shard_index:02d}_of_{args.timestamp_shard_count:02d}"
    )
    shard_done = shard_root / "done.json"
    if args.timestamp_shard_count > 1 and shard_done.is_file():
        print(f"[skip] projected Traj shard complete: {shard_done}", flush=True)
        return
    conditions = [cond_for(r, ds, device) for r in records]

    q_count = len(records)
    d = int(TRACIN_PROJ_DIM)
    batch_size = int(args.batch_size)
    train_mc_pairs = int(args.train_mc_pairs)
    noise_sample_count = (
        2 * train_mc_pairs
        if args.train_noise_sampling == "antithetic"
        else int(TRACIN_TRAIN_MC)
    )
    coefficient = float(TRACIN_SECOND_ORDER_COEFFICIENT)
    snap_weight = 1.0 / len(included_timestamp_indices)
    orders = (
        ("first",)
        if args.first_order_only or args.checkpoint_direction == "backward"
        else ("first", "second")
    )
    if args.checkpoint_direction == "forward":
        all_checkpoint_pairs = [(ci, ci + 1) for ci in range(len(paths) - 1)]
    else:
        all_checkpoint_pairs = [(ci, ci - 1) for ci in range(1, len(paths))]
    checkpoint_pair_indices = parse_pair_indices(
        args.checkpoint_pair_indices, len(all_checkpoint_pairs)
    )
    checkpoint_pairs = [
        all_checkpoint_pairs[index] for index in checkpoint_pair_indices
    ]
    linear = {o: torch.zeros((q_count, N_TRAIN), device=device, dtype=torch.float64) for o in orders}
    term_square = {o: torch.zeros_like(linear[o]) for o in linear}
    timestamp_square = {o: torch.zeros_like(linear[o]) for o in linear}
    started = time.perf_counter()
    print(
        f"[projected-bank {args.family}] start direction={args.checkpoint_direction} "
        f"queries={q_count} transitions={len(checkpoint_pairs)} "
        f"pair_indices={checkpoint_pair_indices} first_order_only={args.first_order_only} "
        f"timestamps={len(selected_timestamp_indices)}/{len(t_seq)} "
        f"shard={args.timestamp_shard_index}/{args.timestamp_shard_count} "
        f"train_points={N_TRAIN} train_noise={args.train_noise_sampling} "
        f"train_mc_pairs={train_mc_pairs if args.train_noise_sampling == 'antithetic' else 0} "
        f"train_loss_terms={noise_sample_count} batch={batch_size} dim={d}",
        flush=True,
    )

    # Timestamp-major traversal keeps only one QxN timestamp accumulator in memory.
    for shard_timestamp_i, si in enumerate(selected_timestamp_indices, start=1):
        tval_raw = t_seq[si]
        tval = int(tval_raw)
        timestamp_acc = {o: torch.zeros_like(linear[o]) for o in linear}
        print(
            f"[projected-bank {args.family}] timestamp {si+1}/{len(t_seq)} "
            f"shard_timestamp={shard_timestamp_i}/{len(selected_timestamp_indices)} "
            f"t={tval} start",
            flush=True,
        )

        for transition_i, (ci, target_ci) in enumerate(checkpoint_pairs, start=1):
            cur_path = paths[ci]
            checkpoint_started = time.perf_counter()
            print(
                f"[projected-bank {args.family}] timestamp {si+1}/{len(t_seq)} "
                f"direction={args.checkpoint_direction} transition "
                f"{transition_i}/{len(checkpoint_pairs)} "
                f"theta_{ci}->target_theta_{target_ci} start",
                flush=True,
            )
            model, _, ck = build_model(cur_path, args.parameter_source, device)
            target, _, target_ck = build_model(
                paths[target_ci], args.parameter_source, device
            )
            named = dict(model.named_parameters())
            names = tuple(named)
            params = tuple(named.values())
            params_dict = dict(named)
            target_named = dict(target.named_parameters())
            compute_second = "second" in orders
            delta = (
                tuple(target_named[n].detach() - named[n].detach() for n in names)
                if compute_second
                else None
            )
            specs = build_countsketch_specs(
                list(params), d, device=device,
                seed_parts=(TRAIN_SEED, "traj_tracin_projection", ci),
            )

            first_queries, second_queries = [], []
            t_q = torch.tensor([tval], device=device, dtype=torch.long)
            for trajectory, condition in zip(trajectories, conditions):
                xt_q = torch.from_numpy(trajectory[si]).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    eps_target = target(xt_q, t_q, condition).detach()
                query_loss = (model(xt_q, t_q, condition) - eps_target).pow(2).sum()
                query_grad = torch.autograd.grad(
                    query_loss,
                    params,
                    create_graph=compute_second,
                )
                first_queries.append(_project_gradient_tuple(query_grad, specs, d))
                if compute_second:
                    directional = sum(
                        (g * direction).sum()
                        for g, direction in zip(query_grad, delta)
                    )
                    query_hvp = torch.autograd.grad(directional, params)
                    second_queries.append(_project_gradient_tuple(
                        tuple(g + coefficient * h for g, h in zip(query_grad, query_hvp)),
                        specs,
                        d,
                    ))
            query_matrix = {"first": torch.stack(first_queries)}
            if compute_second:
                query_matrix["second"] = torch.stack(second_queries)
                del query_hvp, directional
            del query_grad, query_loss, eps_target

            t_mc = torch.full(
                (noise_sample_count,), tval, device=device, dtype=torch.long
            )

            def single_mean_loss(pdict, x0, cond, noises):
                x_mc = x0.unsqueeze(0).expand(noise_sample_count, *x0.shape)
                c_mc = cond.unsqueeze(0).expand(
                    noise_sample_count, cond.shape[-1]
                )
                xt = base.q_sample(x_mc, t_mc, noises, sched)
                pred = functional_call(model, pdict, (xt, t_mc, c_mc))
                return (
                    (pred - noises)
                    .pow(2)
                    .reshape(noise_sample_count, -1)
                    .mean(dim=1)
                    .mean()
                )

            batched_grad = vmap(grad(single_mean_loss), in_dims=(None, 0, 0, 0))
            # Backward mode scores gradients at theta_c against theta_{c-1}
            # and deliberately inherits eta_{c-1}, as requested.
            lr_ck = ck if args.checkpoint_direction == "forward" else target_ck
            term_weight = float(tracin_lr_weight(lr_ck)) * snap_weight
            num_batches = math.ceil(N_TRAIN / batch_size)
            progress_every = max(1, num_batches // 10)
            for batch_i, start in enumerate(range(0, N_TRAIN, batch_size), start=1):
                end = min(start + batch_size, N_TRAIN)
                xb, cb = x_all[start:end], cond_all[start:end]
                if args.train_noise_sampling == "antithetic":
                    generator = make_torch_generator(
                        device,
                        TRAIN_SEED,
                        "projected_traj_train_antithetic",
                        ci,
                        si,
                        start,
                        train_mc_pairs,
                    )
                    base_noises = torch.randn(
                        (end - start, train_mc_pairs, *xb.shape[1:]),
                        generator=generator,
                        device=device,
                        dtype=xb.dtype,
                    )
                    noises = torch.cat((base_noises, -base_noises), dim=1)
                else:
                    generator = make_torch_generator(
                        device,
                        TRAIN_SEED,
                        "projected_traj_train",
                        ci,
                        si,
                        start,
                        noise_sample_count,
                    )
                    noises = torch.randn(
                        (end - start, noise_sample_count, *xb.shape[1:]),
                        generator=generator,
                        device=device,
                        dtype=xb.dtype,
                    )
                grads_b = batched_grad(params_dict, xb, cb, noises)
                phi = _project_batched_grads(grads_b, names, specs, d, False, 1e-8)
                for order in orders:
                    dots = (phi @ query_matrix[order].T).T.to(torch.float64)
                    linear[order][:, start:end] += term_weight * dots
                    term_square[order][:, start:end] += term_weight * dots.square()
                    timestamp_acc[order][:, start:end] += term_weight * dots
                if batch_i == 1 or batch_i % progress_every == 0 or batch_i == num_batches:
                    print(
                        f"[projected-bank {args.family}] timestamp {si+1}/{len(t_seq)} "
                        f"transition {transition_i}/{len(checkpoint_pairs)} train_batch "
                        f"{batch_i}/{num_batches} points={end}/{N_TRAIN}",
                        flush=True,
                    )

            del model, target, query_matrix, first_queries, second_queries
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"[projected-bank {args.family}] timestamp {si+1}/{len(t_seq)} "
                f"transition {transition_i}/{len(checkpoint_pairs)} done "
                f"elapsed={(time.perf_counter()-checkpoint_started)/60:.1f}m",
                flush=True,
            )

        for order in orders:
            timestamp_square[order] += timestamp_acc[order].square()
        print(
            f"[projected-bank {args.family}] timestamp {si+1}/100 queries={q_count} "
            f"elapsed={(time.perf_counter()-started)/3600:.2f}h",
            flush=True,
        )

    result_map = {}
    for order in orders:
        result_map[(order, "linear")] = linear[order]
        result_map[(order, "timestamp_sum_squared")] = timestamp_square[order]
        result_map[(order, "termwise_squared")] = term_square[order]
    if args.timestamp_shard_count > 1:
        shard_root.mkdir(parents=True, exist_ok=True)
        for (order, contraction), values in result_map.items():
            np.save(
                shard_root / f"{order}_{contraction}.npy",
                values.cpu().numpy(),
            )
        with open(shard_done, "w") as handle:
            json.dump(
                {
                    "family": args.family,
                    "checkpoint_direction": args.checkpoint_direction,
                    "checkpoint_pair_indices": checkpoint_pair_indices,
                    "checkpoint_pairs": checkpoint_pairs,
                    "first_order_only": args.first_order_only,
                    "output_suffix": args.output_suffix,
                    "parameter_source": args.parameter_source,
                    "timestamp_shard_index": args.timestamp_shard_index,
                    "timestamp_shard_count": args.timestamp_shard_count,
                    "timestamp_indices": selected_timestamp_indices,
                    "included_timestamp_indices": included_timestamp_indices,
                    "exclude_endpoint": args.exclude_endpoint,
                    "timestamp_weight": snap_weight,
                    "query_ids": [int(record["query_id"]) for record in records],
                    "train_noise_sampling": args.train_noise_sampling,
                    "train_mc_pairs": (
                        train_mc_pairs
                        if args.train_noise_sampling == "antithetic"
                        else None
                    ),
                    "train_loss_terms": noise_sample_count,
                },
                handle,
                indent=2,
            )
        print(f"[done] projected Traj shard saved: {shard_root}", flush=True)
        return

    for qi, record in enumerate(records):
        for (order, contraction), values in result_map.items():
            prefix = (
                "traj_projected"
                if args.checkpoint_direction == "forward"
                else "traj_projected_backward"
            )
            suffix = f"_{args.output_suffix}" if args.output_suffix else ""
            method = (
                f"{prefix}_{order}_{args.parameter_source}_{contraction}{suffix}"
            )
            out = ATTR_DIR / method / f"q{int(record['query_id']):02d}"
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "scores.npy", values[qi].cpu().numpy())
            with open(out / "info.json", "w") as handle:
                json.dump({
                    "query": record, "order": order,
                    "target": (
                        "next" if args.checkpoint_direction == "forward" else "previous"
                    ),
                    "checkpoint_direction": args.checkpoint_direction,
                    "checkpoint_pair_indices": checkpoint_pair_indices,
                    "checkpoint_pairs": checkpoint_pairs,
                    "first_order_only": args.first_order_only,
                    "output_suffix": args.output_suffix,
                    "param_source": args.parameter_source,
                    "projection": "countsketch",
                    "proj_dim": d,
                    "num_snapshots": len(included_timestamp_indices),
                    "excluded_endpoint": args.exclude_endpoint,
                    "included_timestamp_indices": included_timestamp_indices,
                    "timestamp_weight": snap_weight,
                    "train_mc": noise_sample_count,
                    "train_noise_sampling": args.train_noise_sampling,
                    "train_mc_pairs": (
                        train_mc_pairs
                        if args.train_noise_sampling == "antithetic"
                        else None
                    ),
                    "train_loss_terms": noise_sample_count,
                    "contraction": contraction, "lr_weighted": TRACIN_USE_LR_WEIGHTS,
                    "second_order_coefficient": coefficient if order == "second" else 0.0,
                    "second_order_direction": "next_checkpoint_parameter_delta" if order == "second" else "disabled",
                    "learning_rate_source": (
                        "current_checkpoint"
                        if args.checkpoint_direction == "forward"
                        else "previous_checkpoint"
                    ),
                    "bank_scoring": True,
                }, handle, indent=2)


if __name__ == "__main__":
    main()
