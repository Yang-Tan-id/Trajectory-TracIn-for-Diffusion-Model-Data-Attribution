"""Run a controlled aligned/permuted/independent TracIn-DAS ablation."""

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
from noise_pairing_ablation_config import *
from tracin_das_config import DAS_TIMESTEPS
from x3_endpoint_das_jax_logic_pytorch import (
    build_countsketch_specs,
    make_torch_generator,
)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def atomic_npz(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez(handle, **values)
    os.replace(temporary, path)


def optimizer_state_by_name(checkpoint, names, device):
    optimizer_state = checkpoint["optimizer_state"]
    groups = optimizer_state["param_groups"]
    if len(groups) != 1:
        raise ValueError("noise-pairing ablation expects one AdamW parameter group")
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
    scales = (float(GRAD_CLIP) / norm_sq.sqrt().clamp_min(NPA_EPS)).clamp(max=1.0)
    beta1, beta2 = hyper["beta1"], hyper["beta2"]
    sign = -1.0 if hyper["maximize"] else 1.0
    output = {}
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


def score_key(pairing, variant, contraction, group):
    return "__".join((pairing, variant, contraction, group))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--timestamp-shard-index", type=int, required=True)
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-microbatch-size", type=int, default=4)
    parser.add_argument("--query-term-batch-size", type=int, default=128)
    parser.add_argument("--family", choices=("prompted", "unprompted"), default="prompted")
    parser.add_argument("--query-scope", choices=("ten", "all"), default="ten")
    args = parser.parse_args()
    if not 0 <= args.timestamp_shard_index < args.timestamp_shard_count:
        raise ValueError("invalid timestamp shard")
    if min(args.batch_size, args.grad_microbatch_size, args.query_term_batch_size) <= 0:
        raise ValueError("batch sizes must be positive")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    if args.query_scope == "all":
        query_ids = npa100_query_ids(args.family)
        active_variants = ("raw",)
        active_contractions = ("timestamp_sum_squared",)
        contract_version = NPA100_CONTRACT_VERSION
        root = npa100_shard_root(
            args.family, args.timestamp_shard_index, args.timestamp_shard_count
        )
    else:
        if args.family != NPA_FAMILY:
            raise ValueError("the ten-query diagnostic only supports prompted")
        query_ids = NPA_QUERY_IDS
        active_variants = NPA_VARIANTS
        active_contractions = NPA_CONTRACTIONS
        contract_version = NPA_CONTRACT_VERSION
        root = npa_shard_root(args.timestamp_shard_index, args.timestamp_shard_count)
    done_path = root / "done.json"
    if done_path.is_file():
        print(f"[skip] {done_path}", flush=True)
        return

    with open(QUERY_DIR / "manifest.json") as handle:
        by_id = {int(item["query_id"]): item for item in json.load(handle)}
    records = [by_id[query_id] for query_id in query_ids]
    if any(record["family"] != args.family for record in records):
        raise ValueError(f"invalid query bank for family={args.family}")
    endpoints = torch.cat(
        [
            torch.from_numpy(np.load(QUERY_DIR / f"q{query_id:02d}" / "final_state.npy"))
            for query_id in query_ids
        ],
        dim=0,
    ).to(device=device, dtype=torch.float32)
    paths = model_paths(args.family)
    bootstrap, dataset, _ = build_model(paths[0], "raw", device)
    x_all, cond_all = preload_dataset(dataset, args.family, device)
    conditions = torch.cat(
        [cond_for(record, dataset, device) for record in records], dim=0
    )
    del bootstrap
    schedule = base.make_linear_schedule(T, device=device)

    selected = list(
        NPA_TIMESTAMP_INDICES[
            args.timestamp_shard_index :: args.timestamp_shard_count
        ]
    )
    expected_shape = (len(query_ids), N_TRAIN)
    scores = {
        score_key(pairing, variant, contraction, group): torch.zeros(
            expected_shape, device=device, dtype=torch.float64
        )
        for pairing in NPA_PAIRINGS
        for variant in active_variants
        for contraction in active_contractions
        for group in NPA_TIMESTAMP_GROUPS
    }
    partial_path = root / "partial_scores.npz"
    progress_path = root / "progress.json"
    completed = []
    if partial_path.is_file() and progress_path.is_file():
        with open(progress_path) as handle:
            progress = json.load(handle)
        if int(progress["contract_version"]) != contract_version:
            raise ValueError("partial contract version differs")
        if int(progress["grad_microbatch_size"]) != args.grad_microbatch_size:
            raise ValueError("partial grad microbatch differs")
        completed = [int(value) for value in progress["completed_timestamp_indices"]]
        with np.load(partial_path) as partial:
            for key in scores:
                scores[key].copy_(torch.from_numpy(partial[key]).to(device))
        print(f"[resume] timestamps={len(completed)}/{len(selected)}", flush=True)

    remaining = [index for index in selected if index not in set(completed)]
    started = time.perf_counter()
    total_terms = len(remaining) * len(NPA_CHECKPOINT_PAIRS)
    completed_terms = 0
    print(
        f"[pairing-ablation gpu={args.gpu}] family={args.family} "
        f"queries={query_ids[0]}..{query_ids[-1]} ({len(query_ids)}) directions=10 "
        f"timestamps={len(selected)}/20 checkpoint_pairs={NPA_CHECKPOINT_PAIRS} "
        f"pairings={NPA_PAIRINGS} full_adamw=true projection=4096 "
        f"batch={args.batch_size} microbatch={args.grad_microbatch_size}",
        flush=True,
    )

    for local_timestamp, timestamp_index in enumerate(remaining, start=1):
        timestep = int(DAS_TIMESTEPS[timestamp_index])
        groups = [
            group
            for group, indices in NPA_TIMESTAMP_GROUPS.items()
            if timestamp_index in indices
        ]
        timestamp_accumulators = {
            (pairing, variant): torch.zeros(
                (len(query_ids), NPA_DIRECTION_COUNT, N_TRAIN),
                device=device,
                dtype=torch.float32,
            )
            for pairing in NPA_PAIRINGS
            for variant in active_variants
        }

        for checkpoint_position, checkpoint_index in enumerate(
            NPA_CHECKPOINT_PAIRS, start=1
        ):
            term_started = time.perf_counter()
            model, _, checkpoint = build_model(paths[checkpoint_index], "raw", device)
            target, _, _ = build_model(paths[checkpoint_index + 1], "raw", device)
            named = dict(model.named_parameters())
            names = tuple(named)
            parameters = tuple(named.values())
            adam_state, adam_hyper = optimizer_state_by_name(
                checkpoint, names, device
            )
            specs = build_countsketch_specs(
                list(parameters),
                NPA_PROJECTION_DIM,
                device=device,
                seed_parts=(
                    TRAIN_SEED,
                    "noise_pairing_ablation_projection",
                    checkpoint_index,
                ),
            )
            base_generator = make_torch_generator(
                device,
                NPA_NOISE_SEED,
                "noise_pairing_base",
                checkpoint_index,
                timestamp_index,
            )
            independent_generator = make_torch_generator(
                device,
                NPA_NOISE_SEED,
                "noise_pairing_independent",
                checkpoint_index,
                timestamp_index,
            )
            base_noises = torch.randn(
                (NPA_DIRECTION_COUNT, *endpoints.shape[1:]),
                generator=base_generator,
                device=device,
                dtype=endpoints.dtype,
            )
            independent_noises = torch.randn(
                (NPA_DIRECTION_COUNT, *endpoints.shape[1:]),
                generator=independent_generator,
                device=device,
                dtype=endpoints.dtype,
            )
            permutation_generator = make_torch_generator(
                device,
                NPA_NOISE_SEED,
                "noise_pairing_permutation",
                checkpoint_index,
                timestamp_index,
            )
            random_permutation = torch.randperm(
                NPA_DIRECTION_COUNT,
                generator=permutation_generator,
                device=device,
            )
            cyclic_permutation = torch.roll(
                torch.arange(NPA_DIRECTION_COUNT, device=device), shifts=-1
            )

            query_count = len(query_ids)
            endpoint_bank = endpoints[:, None].expand(
                query_count, NPA_DIRECTION_COUNT, *endpoints.shape[1:]
            ).reshape(-1, *endpoints.shape[1:])
            query_noise_bank = base_noises[None].expand(
                query_count, NPA_DIRECTION_COUNT, *base_noises.shape[1:]
            ).reshape(-1, *base_noises.shape[1:])
            query_t = torch.full(
                (query_count * NPA_DIRECTION_COUNT,),
                timestep,
                device=device,
                dtype=torch.long,
            )
            query_condition_bank = conditions[:, None].expand(
                query_count, NPA_DIRECTION_COUNT, conditions.shape[-1]
            ).reshape(-1, conditions.shape[-1])
            query_xt = base.q_sample(endpoint_bank, query_t, query_noise_bank, schedule)
            with torch.no_grad():
                current = model(query_xt, query_t, query_condition_bank)
                following = target(query_xt, query_t, query_condition_bank)
                delta = following - current
                delta_norm = delta.flatten(1).norm(dim=1)
                output_direction = delta / delta_norm.clamp_min(NPA_EPS).reshape(
                    -1, *([1] * (delta.ndim - 1))
                )

            def query_scalar(parameter_dict, xt, t_value, condition, direction):
                prediction = functional_call(
                    model,
                    parameter_dict,
                    (xt.unsqueeze(0), t_value.reshape(1), condition.unsqueeze(0)),
                )
                return (prediction.squeeze(0) * direction).sum()

            query_grad_fn = vmap(
                grad(query_scalar), in_dims=(None, 0, 0, 0, 0)
            )
            query_chunks = []
            for start in range(0, len(query_xt), args.query_term_batch_size):
                end = min(start + args.query_term_batch_size, len(query_xt))
                gradients = query_grad_fn(
                    named,
                    query_xt[start:end],
                    query_t[start:end],
                    query_condition_bank[start:end],
                    output_direction[start:end],
                )
                query_chunks.append(
                    _project_batched_grads(
                        gradients,
                        names,
                        specs,
                        NPA_PROJECTION_DIM,
                        False,
                        NPA_EPS,
                    )
                )
                del gradients
            query_matrix = torch.cat(query_chunks).reshape(
                query_count, NPA_DIRECTION_COUNT, NPA_PROJECTION_DIM
            ).detach()
            query_unit = query_matrix / query_matrix.norm(
                dim=2, keepdim=True
            ).clamp_min(NPA_EPS)

            def train_loss(parameter_dict, x0, condition, t_value, noise):
                xt = base.q_sample(
                    x0.unsqueeze(0), t_value.reshape(1), noise.unsqueeze(0), schedule
                )
                prediction = functional_call(
                    model,
                    parameter_dict,
                    (xt, t_value.reshape(1), condition.unsqueeze(0)),
                )
                return (prediction - noise.unsqueeze(0)).square().mean()

            train_grad_fn = vmap(
                grad(train_loss), in_dims=(None, 0, 0, 0, 0)
            )
            combined_noises = torch.cat((base_noises, independent_noises), dim=0)
            num_batches = math.ceil(N_TRAIN / args.batch_size)
            progress_every = max(1, num_batches // 5)
            for batch_position, outer_start in enumerate(
                range(0, N_TRAIN, args.batch_size), start=1
            ):
                outer_end = min(outer_start + args.batch_size, N_TRAIN)
                for micro_start in range(
                    outer_start, outer_end, args.grad_microbatch_size
                ):
                    micro_end = min(
                        micro_start + args.grad_microbatch_size, outer_end
                    )
                    count = micro_end - micro_start
                    event_count = 2 * NPA_DIRECTION_COUNT
                    xb = x_all[micro_start:micro_end, None].expand(
                        count, event_count, *x_all.shape[1:]
                    ).reshape(-1, *x_all.shape[1:])
                    cb = cond_all[micro_start:micro_end, None].expand(
                        count, event_count, cond_all.shape[-1]
                    ).reshape(-1, cond_all.shape[-1])
                    tb = torch.full(
                        (count * event_count,),
                        timestep,
                        device=device,
                        dtype=torch.long,
                    )
                    nb = combined_noises[None].expand(
                        count, event_count, *combined_noises.shape[1:]
                    ).reshape(-1, *combined_noises.shape[1:])
                    gradients = train_grad_fn(named, xb, cb, tb, nb)
                    updates = adamw_full_batched(
                        gradients, names, named, adam_state, adam_hyper
                    )
                    train_matrix = _project_batched_grads(
                        updates,
                        names,
                        specs,
                        NPA_PROJECTION_DIM,
                        False,
                        NPA_EPS,
                    ).reshape(
                        count, 2, NPA_DIRECTION_COUNT, NPA_PROJECTION_DIM
                    ).detach()
                    base_train = train_matrix[:, 0]
                    independent_train = train_matrix[:, 1]
                    base_train_unit = base_train / base_train.norm(
                        dim=2, keepdim=True
                    ).clamp_min(NPA_EPS)
                    independent_train_unit = independent_train / independent_train.norm(
                        dim=2, keepdim=True
                    ).clamp_min(NPA_EPS)
                    train_by_pairing = {
                        "aligned": (base_train, base_train_unit),
                        "cyclic": (
                            base_train[:, cyclic_permutation],
                            base_train_unit[:, cyclic_permutation],
                        ),
                        "random_permutation": (
                            base_train[:, random_permutation],
                            base_train_unit[:, random_permutation],
                        ),
                        "independent": (independent_train, independent_train_unit),
                    }
                    for pairing, (train_raw, train_unit) in train_by_pairing.items():
                        dots_by_variant = {
                            "raw": torch.einsum(
                                "qmp,bmp->qmb", query_matrix, train_raw
                            )
                        }
                        if "query_train_l2" in active_variants:
                            dots_by_variant["query_train_l2"] = torch.einsum(
                                "qmp,bmp->qmb", query_unit, train_unit
                            )
                        for variant, dots in dots_by_variant.items():
                            timestamp_accumulators[(pairing, variant)][
                                :, :, micro_start:micro_end
                            ] += dots
                            if "linear" in active_contractions:
                                for group in groups:
                                    weight = 1.0 / len(NPA_TIMESTAMP_GROUPS[group])
                                    scores[
                                        score_key(pairing, variant, "linear", group)
                                    ][:, micro_start:micro_end] += (
                                        weight * dots.mean(dim=1).double()
                                    )
                            if "termwise_squared" in active_contractions:
                                for group in groups:
                                    weight = 1.0 / len(NPA_TIMESTAMP_GROUPS[group])
                                    scores[
                                        score_key(
                                            pairing,
                                            variant,
                                            "termwise_squared",
                                            group,
                                        )
                                    ][:, micro_start:micro_end] += (
                                        weight * dots.square().mean(dim=1).double()
                                    )
                    del gradients, updates, train_matrix
                    del base_train, independent_train
                    del base_train_unit, independent_train_unit
                    del train_by_pairing, dots_by_variant, dots
                if (
                    batch_position == 1
                    or batch_position % progress_every == 0
                    or batch_position == num_batches
                ):
                    print(
                        f"[pairing-ablation gpu={args.gpu}] "
                        f"timestamp={local_timestamp}/{len(remaining)} "
                        f"global_index={timestamp_index} pair={checkpoint_position}/10 "
                        f"batch={batch_position}/{num_batches}",
                        flush=True,
                    )

            completed_terms += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / completed_terms * (total_terms - completed_terms)
            print(
                f"[pairing-ablation gpu={args.gpu}] term={completed_terms}/{total_terms} "
                f"delta_norm=[{float(delta_norm.min()):.3e},{float(delta_norm.max()):.3e}] "
                f"term_elapsed={(time.perf_counter()-term_started)/60:.1f}m "
                f"eta={eta/3600:.2f}h",
                flush=True,
            )
            del model, target, checkpoint, named, names, parameters
            del adam_state, adam_hyper, specs
            del base_noises, independent_noises, combined_noises
            del random_permutation, cyclic_permutation
            del endpoint_bank, query_noise_bank, query_t, query_condition_bank
            del query_xt, current, following, delta, delta_norm, output_direction
            del query_chunks, query_matrix, query_unit
            del query_grad_fn, train_grad_fn, query_scalar, train_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for pairing in NPA_PAIRINGS:
            for variant in active_variants:
                timestamp_values = timestamp_accumulators[(pairing, variant)]
                reduced = timestamp_values.square().mean(dim=1).double()
                for group in groups:
                    weight = 1.0 / len(NPA_TIMESTAMP_GROUPS[group])
                    if "timestamp_sum_squared" in active_contractions:
                        scores[
                            score_key(
                                pairing, variant, "timestamp_sum_squared", group
                            )
                        ] += weight * reduced
        completed.append(timestamp_index)
        completed.sort()
        if len(completed) % 2 == 0 or len(completed) == len(selected):
            atomic_npz(
                partial_path,
                {key: value.cpu().numpy() for key, value in scores.items()},
            )
            atomic_json(
                progress_path,
                {
                    "contract_version": contract_version,
                    "timestamp_shard_index": args.timestamp_shard_index,
                    "timestamp_shard_count": args.timestamp_shard_count,
                    "selected_timestamp_indices": selected,
                    "completed_timestamp_indices": completed,
                    "grad_microbatch_size": args.grad_microbatch_size,
                },
            )
        del timestamp_accumulators
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    atomic_npz(partial_path, {key: value.cpu().numpy() for key, value in scores.items()})
    atomic_json(
        done_path,
        {
            "contract_version": contract_version,
            "query_ids": list(query_ids),
            "family": args.family,
            "checkpoint_pairs": list(NPA_CHECKPOINT_PAIRS),
            "timestamp_indices": selected,
            "direction_count": NPA_DIRECTION_COUNT,
            "pairings": list(NPA_PAIRINGS),
            "variants": list(active_variants),
            "contractions": list(active_contractions),
            "projection_dim": NPA_PROJECTION_DIM,
            "train_transform": "saved-state full AdamW hypothetical update",
            "query": "normalized next-checkpoint predicted-noise delta direction",
        },
    )
    print(f"[done] {root}", flush=True)


if __name__ == "__main__":
    main()
