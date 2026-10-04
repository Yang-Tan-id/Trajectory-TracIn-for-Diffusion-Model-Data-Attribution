"""Compute checkpoint-sharded Diffusion-TracIn and Diffusion-ReTrac scores.

The query side is the diffusion-loss gradient on generated endpoint queries,
averaged over query noise and sparse timesteps.  The train side replays the
four exact (t_train, epsilon_train) events realized between saved checkpoints.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

import x3pixel_DM_training as base
from attribution_one_query import build_model, cond_for, model_paths, preload_dataset, tracin_lr_weight
from diffusion_retrac_config import *
from forward_loss_alignment_config import replay_noise_path, replay_t_path
from x3_endpoint_das_jax_logic_pytorch import build_countsketch_specs, make_torch_generator


CONTRACT_VERSION = 2


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def atomic_npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez(handle, **values)
    os.replace(temporary, path)


def project_gradient_batch(grads, names, specs, dimension, *, normalize):
    """CountSketch a batch of full gradients, optionally using exact L2 norms."""
    first = grads[names[0]]
    count = first.shape[0]
    output = torch.zeros(
        (count, dimension), device=first.device, dtype=torch.float32
    )
    denominator = None
    if normalize:
        norm_sq = torch.zeros(count, device=first.device, dtype=torch.float64)
        for name in names:
            flat = grads[name].reshape(count, -1)
            norm_sq += flat.double().square().sum(dim=1)
        denominator = norm_sq.sqrt().to(torch.float32).clamp_min(RETRAC_EPS)
    for name, (indices, signs) in zip(names, specs):
        flat = grads[name].reshape(count, -1).to(torch.float32)
        if denominator is not None:
            flat = flat / denominator[:, None]
        output.scatter_add_(
            1,
            indices.unsqueeze(0).expand(count, -1),
            flat * signs.unsqueeze(0),
        )
    return (output / math.sqrt(float(dimension))).detach()


def optimizer_state_by_name(checkpoint, names, device):
    """Map the saved AdamW state to model parameter names."""
    optimizer_state = checkpoint["optimizer_state"]
    groups = optimizer_state["param_groups"]
    if len(groups) != 1:
        raise ValueError("AdamW ReTrac expects exactly one parameter group")
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


def adamw_full_batched(gradients, names, parameters, optimizer_state, hyper):
    """One hypothetical saved-state AdamW update for every replayed event."""
    norm_sq = None
    for name in names:
        term = gradients[name].float().square().flatten(1).sum(dim=1)
        norm_sq = term if norm_sq is None else norm_sq + term
    scales = (float(GRAD_CLIP) / norm_sq.sqrt().clamp_min(RETRAC_EPS)).clamp(
        max=1.0
    )
    beta1, beta2 = hyper["beta1"], hyper["beta2"]
    sign = -1.0 if hyper["maximize"] else 1.0
    output = {}
    for name in names:
        shape = (len(scales),) + (1,) * (gradients[name].ndim - 1)
        gradient = sign * gradients[name].float() * scales.reshape(shape)
        state = optimizer_state[name]
        step = state["step"] + 1
        moment = beta1 * state["exp_avg"].unsqueeze(0) + (1.0 - beta1) * gradient
        variance = (
            beta2 * state["exp_avg_sq"].unsqueeze(0)
            + (1.0 - beta2) * gradient.square()
        )
        if hyper["amsgrad"]:
            variance = torch.maximum(
                state["max_exp_avg_sq"].unsqueeze(0), variance
            )
        denominator = (
            variance.sqrt() / math.sqrt(1.0 - beta2**step) + hyper["eps"]
        )
        step_size = hyper["lr"] / (1.0 - beta1**step)
        decay = -hyper["lr"] * hyper["weight_decay"] * parameters[name]
        output[name] = decay.unsqueeze(0) - step_size * moment / denominator
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=RETRAC_FAMILIES, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--checkpoint-shard-index", type=int, required=True)
    parser.add_argument("--checkpoint-shard-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.checkpoint_shard_index < args.checkpoint_shard_count:
        raise ValueError("invalid checkpoint shard")
    if args.batch_size <= 0 or args.query_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    family_query_ids = retrac_query_ids(args.family)
    checkpoint_paths = model_paths(args.family)
    selected_checkpoints = list(
        range(
            args.checkpoint_shard_index,
            len(checkpoint_paths),
            args.checkpoint_shard_count,
        )
    )
    root = retrac_shard_root(
        args.family, args.checkpoint_shard_index, args.checkpoint_shard_count
    )
    done_path = root / "done.json"
    if done_path.is_file():
        print(f"[skip] {done_path}", flush=True)
        return

    with open(QUERY_DIR / "manifest.json") as handle:
        by_id = {int(item["query_id"]): item for item in json.load(handle)}
    records = [by_id[query_id] for query_id in family_query_ids]
    endpoints = torch.cat(
        [
            torch.from_numpy(
                np.load(QUERY_DIR / f"q{query_id:02d}" / "final_state.npy")
            )
            for query_id in family_query_ids
        ],
        dim=0,
    ).to(device=device, dtype=torch.float32)

    # Dataset tensors and replay arrays are shared by every checkpoint.
    bootstrap_model, dataset, _ = build_model(
        checkpoint_paths[selected_checkpoints[0]], RETRAC_PARAM_SOURCE, device
    )
    x_all, cond_all = preload_dataset(dataset, args.family, device)
    query_conditions = torch.cat(
        [cond_for(record, dataset, device) for record in records], dim=0
    )
    if args.family == "unprompted":
        query_conditions.zero_()
    del bootstrap_model
    schedule = base.make_linear_schedule(T, device=device)
    replay_t = np.load(replay_t_path(), mmap_mode="r")
    replay_noise = np.load(replay_noise_path(), mmap_mode="r")

    query_count = len(family_query_ids)
    scores_tracin = torch.zeros(
        (query_count, N_TRAIN), device=device, dtype=torch.float64
    )
    scores_retrac = torch.zeros_like(scores_tracin)
    scores_retrac_adamw_full = torch.zeros_like(scores_tracin)
    completed = []
    partial_path = root / "partial_scores.npz"
    progress_path = root / "progress.json"
    if partial_path.is_file() and progress_path.is_file():
        with open(progress_path) as handle:
            progress = json.load(handle)
        if int(progress.get("contract_version", 0)) != CONTRACT_VERSION:
            raise ValueError("partial shard contract version mismatch")
        completed = [int(value) for value in progress["completed_checkpoints"]]
        partial = np.load(partial_path)
        scores_tracin.copy_(torch.from_numpy(partial["tracin"]).to(device))
        scores_retrac.copy_(torch.from_numpy(partial["retrac"]).to(device))
        scores_retrac_adamw_full.copy_(
            torch.from_numpy(partial["retrac_adamw_full"]).to(device)
        )
        print(f"[resume] completed checkpoints={completed}", flush=True)

    started = time.perf_counter()
    print(
        f"[retrac gpu={args.gpu}] family={args.family} "
        f"queries=q{family_query_ids[0]:02d}-q{family_query_ids[-1]:02d} "
        f"shard={args.checkpoint_shard_index}/"
        f"{args.checkpoint_shard_count} checkpoints={selected_checkpoints} "
        f"queries={query_count} timesteps={len(RETRAC_TIMESTEPS)} "
        f"query_mc={RETRAC_QUERY_MC} train_events=4 train_batch={args.batch_size} "
        f"query_batch={args.query_batch_size}",
        flush=True,
    )

    for local_position, checkpoint_index in enumerate(selected_checkpoints, start=1):
        if checkpoint_index in completed:
            continue
        checkpoint_started = time.perf_counter()
        model, _, checkpoint = build_model(
            checkpoint_paths[checkpoint_index], RETRAC_PARAM_SOURCE, device
        )
        named = dict(model.named_parameters())
        names = tuple(named)
        active = tuple(named.values())
        optimizer_state, optimizer_hyper = optimizer_state_by_name(
            checkpoint, names, device
        )
        specs = build_countsketch_specs(
            list(active),
            RETRAC_PROJ_DIM,
            device=device,
            seed_parts=(TRAIN_SEED, "diffusion_retrac_projection", checkpoint_index),
        )

        def single_loss(parameter_dict, x0, condition, timestep, noise):
            timestep = timestep.reshape(1)
            noise = noise.unsqueeze(0)
            xt = base.q_sample(x0.unsqueeze(0), timestep, noise, schedule)
            prediction = functional_call(
                model,
                parameter_dict,
                (xt, timestep, condition.unsqueeze(0)),
            )
            return F.mse_loss(prediction, noise)

        batched_gradient = vmap(
            grad(single_loss), in_dims=(None, 0, 0, 0, 0)
        )

        query_tracin = torch.zeros(
            (query_count, RETRAC_PROJ_DIM), device=device, dtype=torch.float32
        )
        query_retrac = torch.zeros_like(query_tracin)
        for timestamp_position, timestep_value in enumerate(RETRAC_TIMESTEPS, start=1):
            generator = make_torch_generator(
                device,
                TRAIN_SEED,
                "diffusion_retrac_query_noise",
                checkpoint_index,
                int(timestep_value),
            )
            all_noise = torch.randn(
                (query_count, RETRAC_QUERY_MC, *endpoints.shape[1:]),
                generator=generator,
                device=device,
                dtype=endpoints.dtype,
            )
            for query_start in range(0, query_count, args.query_batch_size):
                query_end = min(query_start + args.query_batch_size, query_count)
                size = query_end - query_start
                xq = endpoints[query_start:query_end, None].expand(
                    size, RETRAC_QUERY_MC, *endpoints.shape[1:]
                ).reshape(-1, *endpoints.shape[1:])
                cq = query_conditions[query_start:query_end, None].expand(
                    size, RETRAC_QUERY_MC, query_conditions.shape[-1]
                ).reshape(-1, query_conditions.shape[-1])
                tq = torch.full(
                    (size * RETRAC_QUERY_MC,),
                    int(timestep_value),
                    device=device,
                    dtype=torch.long,
                )
                nq = all_noise[query_start:query_end].reshape(
                    -1, *endpoints.shape[1:]
                )
                event_gradients = batched_gradient(named, xq, cq, tq, nq)
                mean_gradients = {
                    name: value.reshape(
                        size, RETRAC_QUERY_MC, *value.shape[1:]
                    ).mean(dim=1)
                    for name, value in event_gradients.items()
                }
                query_tracin[query_start:query_end] += project_gradient_batch(
                    mean_gradients,
                    names,
                    specs,
                    RETRAC_PROJ_DIM,
                    normalize=False,
                ) / float(len(RETRAC_TIMESTEPS))
                query_retrac[query_start:query_end] += project_gradient_batch(
                    mean_gradients,
                    names,
                    specs,
                    RETRAC_PROJ_DIM,
                    normalize=True,
                ) / float(len(RETRAC_TIMESTEPS))
                del event_gradients, mean_gradients
            if timestamp_position == 1 or timestamp_position % 10 == 0:
                print(
                    f"[retrac gpu={args.gpu}] checkpoint={checkpoint_index+1:02d}/50 "
                    f"query timestep={timestamp_position}/{len(RETRAC_TIMESTEPS)}",
                    flush=True,
                )

        checkpoint_lr = float(tracin_lr_weight(checkpoint))
        num_batches = math.ceil(N_TRAIN / args.batch_size)
        progress_every = max(1, num_batches // 10)
        for batch_position, start in enumerate(
            range(0, N_TRAIN, args.batch_size), start=1
        ):
            end = min(start + args.batch_size, N_TRAIN)
            count = end - start
            xb = x_all[start:end, None].expand(
                count, RETRAC_EVENTS_PER_CHECKPOINT, *x_all.shape[1:]
            ).reshape(-1, *x_all.shape[1:])
            cb = cond_all[start:end, None].expand(
                count, RETRAC_EVENTS_PER_CHECKPOINT, cond_all.shape[-1]
            ).reshape(-1, cond_all.shape[-1])
            tb = torch.from_numpy(
                np.asarray(replay_t[checkpoint_index, start:end], dtype=np.int64)
            ).to(device).reshape(-1)
            nb = torch.from_numpy(
                np.asarray(replay_noise[checkpoint_index, start:end])
            ).to(device=device, dtype=torch.float32).reshape(-1, *x_all.shape[1:])
            event_gradients = batched_gradient(named, xb, cb, tb, nb)
            train_event_tracin = project_gradient_batch(
                event_gradients,
                names,
                specs,
                RETRAC_PROJ_DIM,
                normalize=False,
            ).reshape(count, RETRAC_EVENTS_PER_CHECKPOINT, -1).mean(dim=1)
            train_event_retrac = project_gradient_batch(
                event_gradients,
                names,
                specs,
                RETRAC_PROJ_DIM,
                normalize=True,
            ).reshape(count, RETRAC_EVENTS_PER_CHECKPOINT, -1).mean(dim=1)
            adamw_updates = adamw_full_batched(
                event_gradients,
                names,
                named,
                optimizer_state,
                optimizer_hyper,
            )
            train_event_retrac_adamw_full = project_gradient_batch(
                adamw_updates,
                names,
                specs,
                RETRAC_PROJ_DIM,
                normalize=True,
            ).reshape(count, RETRAC_EVENTS_PER_CHECKPOINT, -1).mean(dim=1)
            scores_tracin[:, start:end] += (
                query_tracin @ train_event_tracin.T
            ).double() * checkpoint_lr
            scores_retrac[:, start:end] += (
                query_retrac @ train_event_retrac.T
            ).double() * checkpoint_lr
            scores_retrac_adamw_full[:, start:end] += (
                query_retrac @ train_event_retrac_adamw_full.T
            ).double() * checkpoint_lr
            del event_gradients, adamw_updates
            del train_event_tracin, train_event_retrac
            del train_event_retrac_adamw_full
            if (
                batch_position == 1
                or batch_position % progress_every == 0
                or batch_position == num_batches
            ):
                print(
                    f"[retrac gpu={args.gpu}] checkpoint={checkpoint_index+1:02d}/50 "
                    f"train batch={batch_position}/{num_batches}",
                    flush=True,
                )

        completed.append(checkpoint_index)
        atomic_npz(
            partial_path,
            tracin=scores_tracin.cpu().numpy(),
            retrac=scores_retrac.cpu().numpy(),
            retrac_adamw_full=scores_retrac_adamw_full.cpu().numpy(),
        )
        atomic_json(
            progress_path,
            {
                "contract_version": CONTRACT_VERSION,
                "family": args.family,
                "query_ids": list(family_query_ids),
                "checkpoint_shard_index": args.checkpoint_shard_index,
                "checkpoint_shard_count": args.checkpoint_shard_count,
                "selected_checkpoints": selected_checkpoints,
                "completed_checkpoints": completed,
            },
        )
        elapsed = time.perf_counter() - checkpoint_started
        total_elapsed = time.perf_counter() - started
        remaining = len(selected_checkpoints) - len(completed)
        print(
            f"[retrac gpu={args.gpu}] checkpoint={checkpoint_index+1:02d}/50 done "
            f"elapsed={elapsed/60:.1f}m shard_eta≈"
            f"{(total_elapsed/max(1,len(completed))*remaining)/3600:.2f}h",
            flush=True,
        )
        del model, named, active, query_tracin, query_retrac
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    atomic_json(
        done_path,
        {
            "contract_version": CONTRACT_VERSION,
            "family": args.family,
            "checkpoint_shard_index": args.checkpoint_shard_index,
            "checkpoint_shard_count": args.checkpoint_shard_count,
            "checkpoint_indices": selected_checkpoints,
            "query_ids": list(family_query_ids),
            "query_timesteps": list(RETRAC_TIMESTEPS),
            "query_mc": RETRAC_QUERY_MC,
            "train_events_per_checkpoint": RETRAC_EVENTS_PER_CHECKPOINT,
            "train_event_source": "exact replayed t_train and epsilon_train",
            "adamw_train_transform": (
                "saved-state full AdamW hypothetical update; full-space L2 "
                "normalization before CountSketch"
            ),
            "query_adamw_transform": False,
            "parameter_source": RETRAC_PARAM_SOURCE,
            "projection_dim": RETRAC_PROJ_DIM,
            "checkpoint_weight": "saved checkpoint learning rate",
        },
    )
    print(f"[done] {root}", flush=True)


if __name__ == "__main__":
    main()
