"""100x10 endpoint-loss TracIn with saved-state full AdamW train features."""

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
from endpoint_tracin_adamw_mc10_config import *
from x3_endpoint_das_jax_logic_pytorch import (
    build_countsketch_specs,
    make_torch_generator,
)


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


def optimizer_state_by_name(checkpoint, names, device):
    optimizer_state = checkpoint["optimizer_state"]
    groups = optimizer_state["param_groups"]
    if len(groups) != 1:
        raise ValueError("expected one AdamW parameter group")
    parameter_ids = list(groups[0]["params"])
    if len(parameter_ids) != len(names):
        raise ValueError("optimizer/model parameter count mismatch")
    states = optimizer_state["state"]
    by_name = {}
    for name, parameter_id in zip(names, parameter_ids):
        state = states[parameter_id]
        by_name[name] = {
            "step": int(torch.as_tensor(state["step"]).item()),
            "exp_avg": state["exp_avg"].to(device=device, dtype=torch.float32),
            "exp_avg_sq": state["exp_avg_sq"].to(
                device=device, dtype=torch.float32
            ),
        }
        if "max_exp_avg_sq" in state:
            by_name[name]["max_exp_avg_sq"] = state["max_exp_avg_sq"].to(
                device=device, dtype=torch.float32
            )
    group = groups[0]
    return by_name, {
        "lr": float(group["lr"]),
        "beta1": float(group["betas"][0]),
        "beta2": float(group["betas"][1]),
        "eps": float(group["eps"]),
        "weight_decay": float(group.get("weight_decay", 0.0)),
        "amsgrad": bool(group.get("amsgrad", False)),
        "maximize": bool(group.get("maximize", False)),
    }


def adamw_full_batched(gradients, names, parameters, state, hyper):
    norm_sq = None
    for name in names:
        term = gradients[name].float().square().flatten(1).sum(dim=1)
        norm_sq = term if norm_sq is None else norm_sq + term
    scales = (float(GRAD_CLIP) / norm_sq.sqrt().clamp_min(ETA_EPS)).clamp(max=1.0)
    output = {}
    beta1, beta2 = hyper["beta1"], hyper["beta2"]
    sign = -1.0 if hyper["maximize"] else 1.0
    for name in names:
        shape = (len(scales),) + (1,) * (gradients[name].ndim - 1)
        gradient = sign * gradients[name].float() * scales.reshape(shape)
        item = state[name]
        step = item["step"] + 1
        moment = beta1 * item["exp_avg"].unsqueeze(0) + (1.0 - beta1) * gradient
        variance = (
            beta2 * item["exp_avg_sq"].unsqueeze(0)
            + (1.0 - beta2) * gradient.square()
        )
        if hyper["amsgrad"]:
            variance = torch.maximum(
                item["max_exp_avg_sq"].unsqueeze(0), variance
            )
        denominator = variance.sqrt() / math.sqrt(1.0 - beta2**step) + hyper["eps"]
        step_size = hyper["lr"] / (1.0 - beta1**step)
        decay = -hyper["lr"] * hyper["weight_decay"] * parameters[name]
        output[name] = decay.unsqueeze(0) - step_size * moment / denominator
    return output


def normalized_dots(query_matrix, train_matrix):
    raw = (train_matrix @ query_matrix.T).T
    query_norm = query_matrix.norm(dim=1).clamp_min(ETA_EPS)
    train_norm = train_matrix.norm(dim=1).clamp_min(ETA_EPS)
    return {
        "raw": raw,
        "query_l2": raw / query_norm[:, None],
        "train_l2": raw / train_norm[None, :],
        "query_train_l2": raw / (query_norm[:, None] * train_norm[None, :]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--timestamp-shard-index", type=int, required=True)
    parser.add_argument("--timestamp-shard-count", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=640)
    parser.add_argument("--grad-microbatch-size", type=int, default=32)
    parser.add_argument("--query-batch-size", type=int, default=8)
    args = parser.parse_args()
    if not 0 <= args.timestamp_shard_index < args.timestamp_shard_count:
        raise ValueError("invalid timestamp shard")
    if min(args.batch_size, args.grad_microbatch_size, args.query_batch_size) <= 0:
        raise ValueError("batch sizes must be positive")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    records = sorted(
        (record for record in manifest if record["family"] == args.family),
        key=lambda record: int(record["query_id"]),
    )
    query_ids = [int(record["query_id"]) for record in records]
    if not records:
        raise ValueError(f"no queries for family={args.family}")

    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    paths = model_paths(args.family)
    if len(paths) != 50:
        raise ValueError(f"expected 50 checkpoints, found {len(paths)}")
    schedule = base.make_linear_schedule(T, device=device)
    x_all, cond_all = preload_dataset(dataset, args.family, device)
    endpoints = torch.cat(
        [
            torch.from_numpy(np.load(Path(record["dir"]) / "final_state.npy"))
            for record in records
        ],
        dim=0,
    ).to(device=device, dtype=torch.float32)
    query_conditions = torch.cat(
        [cond_for(record, dataset, device) for record in records], dim=0
    )
    timestamps = tuple(int(value) for value in ETA_TIMESTAMPS)
    selected = list(
        range(args.timestamp_shard_index, len(timestamps), args.timestamp_shard_count)
    )
    root = eta_shard_root(
        args.family, args.timestamp_shard_index, args.timestamp_shard_count
    )
    done_path = root / "done.json"
    if done_path.is_file():
        print(f"[skip] {done_path}", flush=True)
        return

    expected_shape = (len(records), N_TRAIN)
    partial_paths = {
        variant: {
            contraction: root / f"partial_{variant}_{contraction}.npy"
            for contraction in ETA_CONTRACTIONS
        }
        for variant in ETA_VARIANTS
    }
    progress_path = root / "progress.json"
    completed = []
    flat_paths = [path for values in partial_paths.values() for path in values.values()]
    if all(path.is_file() for path in flat_paths) and progress_path.is_file():
        with open(progress_path) as handle:
            progress = json.load(handle)
        if int(progress["contract_version"]) != ETA_CONTRACT_VERSION:
            raise ValueError("partial contract changed")
        expected = {
            "batch_size": args.batch_size,
            "grad_microbatch_size": args.grad_microbatch_size,
            "query_batch_size": args.query_batch_size,
        }
        for key, value in expected.items():
            if int(progress[key]) != int(value):
                raise ValueError(f"partial {key} differs")
        if progress["query_ids"] != query_ids:
            raise ValueError("partial query IDs differ")
        completed = [int(value) for value in progress["completed_timestamps"]]
        scores = {
            variant: {
                contraction: torch.from_numpy(np.load(path)).to(
                    device=device, dtype=torch.float64
                )
                for contraction, path in values.items()
            }
            for variant, values in partial_paths.items()
        }
        print(f"[resume] timestamps={len(completed)}/{len(selected)}", flush=True)
    else:
        scores = {
            variant: {
                contraction: torch.zeros(
                    expected_shape, device=device, dtype=torch.float64
                )
                for contraction in ETA_CONTRACTIONS
            }
            for variant in ETA_VARIANTS
        }

    remaining = [value for value in selected if value not in set(completed)]
    total_terms = len(remaining) * 49
    completed_terms = 0
    started = time.perf_counter()
    print(
        f"[endpoint-adamw gpu={args.gpu}] family={args.family} "
        f"queries={query_ids[0]}..{query_ids[-1]} timestamps={len(selected)}/100 "
        f"query_mc=10 train_mc=10 independent_noise=true full_adamw=true "
        f"projection=4096 batch={args.batch_size} microbatch={args.grad_microbatch_size}",
        flush=True,
    )

    for local_timestamp, timestamp_index in enumerate(remaining, start=1):
        timestep = int(timestamps[timestamp_index])
        t_mc = torch.full(
            (ETA_QUERY_MC,), timestep, device=device, dtype=torch.long
        )
        timestamp_accumulators = {
            variant: torch.zeros(expected_shape, device=device, dtype=torch.float64)
            for variant in ETA_VARIANTS
        }

        for checkpoint_index in range(49):
            term_started = time.perf_counter()
            model, _, checkpoint = build_model(paths[checkpoint_index], "raw", device)
            named = dict(model.named_parameters())
            names = tuple(named)
            parameters = tuple(named.values())
            adam_state, adam_hyper = optimizer_state_by_name(
                checkpoint, names, device
            )
            specs = build_countsketch_specs(
                list(parameters),
                ETA_PROJECTION_DIM,
                device=device,
                seed_parts=(
                    TRAIN_SEED,
                    "endpoint_tracin_adamw_mc10_projection",
                    checkpoint_index,
                ),
            )

            def mean_mc_loss(parameter_dict, x0, condition, noises):
                x_bank = x0.unsqueeze(0).expand(ETA_QUERY_MC, *x0.shape)
                condition_bank = condition.unsqueeze(0).expand(
                    ETA_QUERY_MC, condition.shape[-1]
                )
                xt = base.q_sample(x_bank, t_mc, noises, schedule)
                prediction = functional_call(
                    model, parameter_dict, (xt, t_mc, condition_bank)
                )
                return (prediction - noises).square().mean()

            batched_gradient = vmap(
                grad(mean_mc_loss), in_dims=(None, 0, 0, 0)
            )
            query_generator = make_torch_generator(
                device,
                TRAIN_SEED,
                "endpoint_tracin_query_mc10",
                checkpoint_index,
                timestamp_index,
            )
            query_noises = torch.randn(
                (len(records), ETA_QUERY_MC, *endpoints.shape[1:]),
                generator=query_generator,
                device=device,
                dtype=endpoints.dtype,
            )
            query_chunks = []
            for start in range(0, len(records), args.query_batch_size):
                end = min(start + args.query_batch_size, len(records))
                gradients = batched_gradient(
                    named,
                    endpoints[start:end],
                    query_conditions[start:end],
                    query_noises[start:end],
                )
                query_chunks.append(
                    _project_batched_grads(
                        gradients,
                        names,
                        specs,
                        ETA_PROJECTION_DIM,
                        False,
                        1e-8,
                    )
                )
                del gradients
            query_matrix = torch.cat(query_chunks, dim=0).detach()

            num_batches = math.ceil(N_TRAIN / args.batch_size)
            progress_every = max(1, num_batches // 10)
            for batch_position, start in enumerate(
                range(0, N_TRAIN, args.batch_size), start=1
            ):
                end = min(start + args.batch_size, N_TRAIN)
                for micro_start in range(start, end, args.grad_microbatch_size):
                    micro_end = min(micro_start + args.grad_microbatch_size, end)
                    generator = make_torch_generator(
                        device,
                        TRAIN_SEED,
                        "endpoint_tracin_train_mc10",
                        checkpoint_index,
                        timestamp_index,
                        micro_start,
                    )
                    train_noises = torch.randn(
                        (
                            micro_end - micro_start,
                            ETA_TRAIN_MC,
                            *x_all.shape[1:],
                        ),
                        generator=generator,
                        device=device,
                        dtype=x_all.dtype,
                    )
                    gradients = batched_gradient(
                        named,
                        x_all[micro_start:micro_end],
                        cond_all[micro_start:micro_end],
                        train_noises,
                    )
                    updates = adamw_full_batched(
                        gradients, names, named, adam_state, adam_hyper
                    )
                    train_matrix = _project_batched_grads(
                        updates,
                        names,
                        specs,
                        ETA_PROJECTION_DIM,
                        False,
                        1e-8,
                    ).detach()
                    variants = normalized_dots(query_matrix, train_matrix)
                    for variant in ETA_VARIANTS:
                        dots = variants[variant].to(torch.float64)
                        scores[variant]["linear"][:, micro_start:micro_end] += (
                            dots / float(len(timestamps))
                        )
                        scores[variant]["termwise_squared"][
                            :, micro_start:micro_end
                        ] += dots.square() / float(len(timestamps))
                        timestamp_accumulators[variant][
                            :, micro_start:micro_end
                        ] += dots
                    del train_noises, gradients, updates, train_matrix, variants, dots
                if (
                    batch_position == 1
                    or batch_position % progress_every == 0
                    or batch_position == num_batches
                ):
                    print(
                        f"[endpoint-adamw gpu={args.gpu}] "
                        f"timestamp={local_timestamp}/{len(remaining)} global_t={timestamp_index+1}/100 "
                        f"pair={checkpoint_index+1}/49 batch={batch_position}/{num_batches}",
                        flush=True,
                    )

            completed_terms += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / completed_terms * (total_terms - completed_terms)
            print(
                f"[endpoint-adamw gpu={args.gpu}] term={completed_terms}/{total_terms} "
                f"elapsed={(time.perf_counter()-term_started)/60:.1f}m eta={eta/3600:.2f}h",
                flush=True,
            )
            del model, checkpoint, named, names, parameters, adam_state, adam_hyper
            del specs, batched_gradient, query_noises, query_chunks, query_matrix
            del mean_mc_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for variant in ETA_VARIANTS:
            scores[variant]["timestamp_sum_squared"] += (
                timestamp_accumulators[variant].square() / float(len(timestamps))
            )
        completed.append(timestamp_index)
        completed.sort()
        for variant in ETA_VARIANTS:
            for contraction, path in partial_paths[variant].items():
                atomic_numpy(path, scores[variant][contraction].cpu().numpy())
        atomic_json(
            progress_path,
            {
                "contract_version": ETA_CONTRACT_VERSION,
                "family": args.family,
                "query_ids": query_ids,
                "batch_size": args.batch_size,
                "grad_microbatch_size": args.grad_microbatch_size,
                "query_batch_size": args.query_batch_size,
                "completed_timestamps": completed,
            },
        )
        print(
            f"[checkpoint] timestamps={len(completed)}/{len(selected)}",
            flush=True,
        )

    for variant in ETA_VARIANTS:
        for contraction in ETA_CONTRACTIONS:
            atomic_numpy(
                root / f"{variant}_{contraction}.npy",
                scores[variant][contraction].cpu().numpy(),
            )
    atomic_json(
        done_path,
        {
            "contract_version": ETA_CONTRACT_VERSION,
            "family": args.family,
            "query_ids": query_ids,
            "timestamp_indices": selected,
            "query_mc": ETA_QUERY_MC,
            "train_mc": ETA_TRAIN_MC,
            "query_train_noise_alignment": False,
            "checkpoint_transitions": 49,
            "parameter_transform": "full saved-state AdamW",
            "projection_dim": ETA_PROJECTION_DIM,
            "normalizations": list(ETA_VARIANTS),
            "contractions": list(ETA_CONTRACTIONS),
        },
    )
    print(f"[done] {done_path}", flush=True)


if __name__ == "__main__":
    main()
