#!/usr/bin/env python3
"""Endpoint-pollute pilot with multiple output-delta query VJP directions.

The train gradient/update is evaluated once per datapoint and timestamp.  Each
query uses several independently noised x_t states, hence several checkpoint
prediction deltas and parameter-space VJPs.  Scores are retained per direction
and as an average of the direction-level scores.  Train gradients are streamed
and never persisted.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


CELEBA_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = CELEBA_ROOT.parent
LEGACY_ROOT = REFINE_ROOT / "legacy_jax"
for path in (CELEBA_ROOT, REFINE_ROOT, LEGACY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp

from common.stage_artifact_runner import _write_score_outputs
from dataset_config_attr40 import ATTRIBUTION_INDICES_PATH, _prompt_path_tag
from dtrak.algorithm import build_countsketch_projector_jax
from traj_tracin.algorithm import (
    apply_checkpoint_config,
    array_to_device,
    checkpoint_timestamp_shared_noise_key,
    get_adapter,
    list_checkpoints_sorted,
    make_diffusion_schedule,
    make_train_batch,
    pad_indices_to_batch,
    q_sample,
    save_npz_compressed_atomic,
    schedule_to_device,
    select_state_params,
    tracin_checkpoint_lr_weight,
    tree_to_device,
)

from run_endpoint_timewise_1x50 import (
    EPS,
    VARIANTS,
    format_duration,
    load_queries,
    make_config,
    make_train_fn,
    parse_ints,
    progress_bar,
)


TRANSFORMS = ("raw_gradient", "adamw_full")


def make_query_fn(adapter, model, schedule, projector):
    def one_direction(params, target_params, endpoint, cond, noise, timestep):
        t = jnp.full((endpoint.shape[0],), timestep, dtype=jnp.int32)
        xt = q_sample(schedule, endpoint, t, noise)

        def pollute_loss(candidate):
            prediction = adapter.eps_apply(model, candidate, xt, t, cond)
            target = jax.lax.stop_gradient(
                adapter.eps_apply(model, target_params, xt, t, cond)
            )
            delta = jax.lax.stop_gradient(target - prediction)
            delta_norm = jnp.sqrt(jnp.sum(jnp.square(delta), dtype=jnp.float32))
            return jnp.mean(prediction * delta), delta_norm

        (_value, delta_norm), gradient = jax.value_and_grad(
            pollute_loss, has_aux=True
        )(params)
        return projector(gradient).astype(jnp.float32), delta_norm.astype(jnp.float32)

    def directions(params, target_params, endpoint, cond, noises, timestep):
        return jax.vmap(
            one_direction,
            in_axes=(None, None, None, None, 0, None),
        )(params, target_params, endpoint, cond, noises, timestep)

    return jax.jit(directions)


def direction_noises(
    train_seed: int,
    checkpoint_index: int,
    timestep: int,
    direction_count: int,
    endpoint_shape: tuple[int, ...],
) -> jax.Array:
    base = checkpoint_timestamp_shared_noise_key(
        train_seed, checkpoint_index, timestep
    )
    keys = [base]
    keys.extend(jax.random.fold_in(base, direction) for direction in range(1, direction_count))
    return jnp.stack(
        [jax.random.normal(key, endpoint_shape, dtype=jnp.float32) for key in keys],
        axis=0,
    )


def score_batch(
    query: jax.Array,
    query_normalized: jax.Array,
    raw_train: jax.Array,
    raw_train_normalized: jax.Array,
    adamw_train: jax.Array,
    adamw_train_normalized: jax.Array,
) -> np.ndarray:
    """Return [direction, transform, variant, query, batch]."""
    direction_count, query_count, proj_dim = query.shape
    query_flat = query.reshape(direction_count * query_count, proj_dim)
    query_norm_flat = query_normalized.reshape(direction_count * query_count, proj_dim)
    train_all = jnp.concatenate(
        (raw_train, raw_train_normalized, adamw_train, adamw_train_normalized),
        axis=0,
    )
    raw_dots = jnp.matmul(query_flat, train_all.T, precision=jax.lax.Precision.HIGHEST)
    normalized_dots = jnp.matmul(
        query_norm_flat, train_all.T, precision=jax.lax.Precision.HIGHEST
    )
    normalized_dots.block_until_ready()
    batch_size = raw_train.shape[0]
    raw = np.asarray(jax.device_get(raw_dots), dtype=np.float32).reshape(
        direction_count, query_count, 4, batch_size
    )
    normalized = np.asarray(jax.device_get(normalized_dots), dtype=np.float32).reshape(
        direction_count, query_count, 4, batch_size
    )
    result = np.empty(
        (direction_count, len(TRANSFORMS), len(VARIANTS), query_count, batch_size),
        dtype=np.float32,
    )
    for transform_slot, (raw_block, normalized_block) in enumerate(((0, 1), (2, 3))):
        result[:, transform_slot, 0] = raw[:, :, raw_block]
        result[:, transform_slot, 1] = normalized[:, :, raw_block]
        result[:, transform_slot, 2] = raw[:, :, normalized_block]
        result[:, transform_slot, 3] = normalized[:, :, normalized_block]
    return result


def namespace(transform: str, direction_label: str, direction_count: int) -> str:
    return (
        f"celeba_endpoint_pollute_{transform}_train1x10_"
        f"query{direction_count}directionx10_linear_deltaensemble_"
        f"shape3d_semantics_{direction_label}"
    )


def merge(args: argparse.Namespace, queries: list[dict[str, Any]], result_root: Path) -> None:
    shards = []
    for shard_index in range(args.shard_count):
        path = Path(f"{args.component_prefix}.shard_{shard_index:02d}.npz")
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            shards.append({key: np.asarray(payload[key]) for key in payload.files})
    score_indices = shards[0]["score_indices"]
    query_ids = shards[0]["query_ids"]
    for shard in shards:
        if not np.array_equal(shard["score_indices"], score_indices):
            raise ValueError("score indices differ between shards")
        if not np.array_equal(shard["query_ids"], query_ids):
            raise ValueError("query ids differ between shards")
        if len(shard["completed_checkpoints"]) != len(
            range(shard["shard_index"].item(), 49, args.shard_count)
        ):
            raise ValueError(f"incomplete shard {shard['shard_index'].item()}")
    scores = sum(
        (shard["scores"] for shard in shards),
        np.zeros_like(shards[0]["scores"], dtype=np.float64),
    )
    labels = [f"direction{index}" for index in range(args.direction_count)]
    arrays = [scores[index] for index in range(args.direction_count)]
    labels.append("direction_mean")
    arrays.append(np.mean(scores, axis=0, dtype=np.float64))
    record_by_id = {int(query["id"]): query for query in queries}
    namespaces = []
    for label, values in zip(labels, arrays, strict=True):
        for transform_slot, transform in enumerate(TRANSFORMS):
            score_namespace = namespace(transform, label, args.direction_count)
            namespaces.append(score_namespace)
            for query_slot, query_id in enumerate(query_ids):
                query = record_by_id[int(query_id)]
                root = (
                    result_root / "attribution_score" / "prompted_solo"
                    / f"train_seed_{args.train_seed}"
                    / f"query_{_prompt_path_tag(query['prompt'])}"
                    / f"initial_seed_{int(query['seed'])}"
                    / f"traj_tracin_{score_namespace}"
                )
                for variant_slot, (directory, variant) in enumerate(VARIANTS):
                    _write_score_outputs(
                        root / directory,
                        values[transform_slot, variant_slot, query_slot],
                        score_indices,
                        train_dir=args.component_prefix.parent,
                        query_dir=result_root / "sample_ddim_eta0_50",
                        algorithm="traj_tracin",
                        extra_manifest={
                            "query_objective": "endpoint_pollute",
                            "query_direction": label,
                            "query_direction_count": args.direction_count,
                            "query_timestamp_count": 10,
                            "train_timestamp_count": 10,
                            "train_mc_per_timestamp": 1,
                            "query_mc_per_timestamp": args.direction_count,
                            "score_variant": variant,
                            "score_contraction": "linear",
                            "train_feature": transform,
                            "normalization_order": (
                                "per-direction output delta L2; optional per-direction "
                                "projected query L2; score directions separately; average scores"
                            ),
                            "persistent_train_gradient_artifact": False,
                        },
                    )
        print(f"[write] {label} complete", flush=True)
    namespace_path = args.component_prefix.parent / "score_namespaces.txt"
    namespace_path.write_text("\n".join(namespaces) + "\n")
    print(f"[done] wrote {len(namespaces)} namespaces", flush=True)
    print(f"[namespaces] {namespace_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", default=",".join(str(i) for i in range(72)))
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--timestamp-count", type=int, default=10)
    parser.add_argument("--direction-count", type=int, default=4)
    parser.add_argument("--proj-dim", type=int, default=4096)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--progress-every-batches", type=int, default=25)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--component-prefix", type=Path, required=True)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.timestamp_count != 10:
        raise ValueError("this pilot requires exactly 10 timestamps")
    if args.direction_count < 2:
        raise ValueError("--direction-count must be at least 2")
    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError("invalid shard")

    query_ids = parse_ints(args.query_ids)
    os.environ.update(
        EXPERIMENT_TAG=args.experiment,
        TRAIN_SEED=str(args.train_seed),
        JAX_EPOCHS=str(args.epochs),
        TRAJ_PARAMETER_SOURCE="raw",
        TRAJ_TRACIN_PROJ_DIM=str(args.proj_dim),
    )
    cfg = make_config(args)
    result_root = CELEBA_ROOT / "result" / args.experiment
    sample_root = result_root / "sample_ddim_eta0_50"
    checkpoints = list_checkpoints_sorted(cfg.checkpoint_dir)
    if len(checkpoints) != 50:
        raise ValueError(f"expected 50 checkpoints, found {len(checkpoints)}")
    reference_checkpoint = Path(cfg.reference_ckpt or checkpoints[-1])
    apply_checkpoint_config(cfg, str(reference_checkpoint))
    adapter = get_adapter(cfg)
    dataset = adapter.iter_dataset(cfg)
    queries = load_queries(
        args.query_file, query_ids, sample_root, reference_checkpoint, adapter, dataset, cfg
    )
    if args.merge_only:
        merge(args, queries, result_root)
        return

    attribution_indices = np.asarray(
        np.load(ATTRIBUTION_INDICES_PATH), dtype=np.int64
    ).reshape(-1)
    device = adapter.choose_device(cfg.prefer_device)
    model = adapter.build_model(cfg)
    state_template = adapter.build_state_template(cfg, model, device)
    schedule = schedule_to_device(
        make_diffusion_schedule(cfg.timesteps, cfg.beta_start, cfg.beta_end), device
    )
    timesteps = np.linspace(999, 0, 10, dtype=np.int32)
    component_path = Path(
        f"{args.component_prefix}.shard_{args.shard_index:02d}.npz"
    )
    score_shape = (
        args.direction_count,
        len(TRANSFORMS),
        len(VARIANTS),
        len(queries),
        len(attribution_indices),
    )
    scores = np.zeros(score_shape, dtype=np.float64)
    completed: list[int] = []
    if component_path.is_file():
        with np.load(component_path, allow_pickle=False) as payload:
            if tuple(payload["scores"].shape) != score_shape:
                raise ValueError(f"resume shape mismatch: {payload['scores'].shape} != {score_shape}")
            scores[...] = payload["scores"]
            completed = [int(value) for value in payload["completed_checkpoints"]]
        print(f"[resume] completed checkpoints={completed}", flush=True)

    endpoint_shape = tuple(queries[0]["endpoint"].shape)
    total_batches = (len(attribution_indices) + args.train_batch_size - 1) // args.train_batch_size
    for checkpoint_index in range(49):
        if checkpoint_index % args.shard_count != args.shard_index:
            continue
        if checkpoint_index in completed:
            print(f"[skip] checkpoint {checkpoint_index + 1}/49", flush=True)
            continue
        state, _ = adapter.restore_state(checkpoints[checkpoint_index], state_template)
        target_state, _ = adapter.restore_state(checkpoints[checkpoint_index + 1], state_template)
        params = tree_to_device(select_state_params(state, "raw"), device)
        target_params = tree_to_device(select_state_params(target_state, "raw"), device)
        projector = build_countsketch_projector_jax(
            params,
            args.proj_dim,
            seed_parts=(args.train_seed, "traj_tracin_projection", checkpoint_index),
            device=device,
        )
        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
        history_update, _ = state.tx.update(zero_grads, state.opt_state, params)
        history_update = tree_to_device(history_update, device)
        history_feature = array_to_device(
            jnp.asarray(np.asarray(projector(history_update), dtype=np.float32)), device
        )
        query_fn = make_query_fn(adapter, model, schedule, projector)
        train_fn = make_train_fn(
            adapter, model, schedule, projector, state, params, history_update
        )
        checkpoint_weight = tracin_checkpoint_lr_weight(
            cfg, checkpoints[checkpoint_index], checkpoint_index, len(checkpoints), len(dataset)
        )

        for timestamp_slot, timestep in enumerate(timesteps):
            timestamp_started = time.monotonic()
            query_started = time.monotonic()
            query_features = np.empty(
                (args.direction_count, len(queries), args.proj_dim), dtype=np.float32
            )
            delta_norms = np.empty(
                (args.direction_count, len(queries)), dtype=np.float32
            )
            noises = array_to_device(
                direction_noises(
                    args.train_seed,
                    checkpoint_index,
                    int(timestep),
                    args.direction_count,
                    endpoint_shape,
                ),
                device,
            )
            for query_slot, query in enumerate(queries):
                endpoint = array_to_device(jnp.asarray(query["endpoint"]), device)
                cond = array_to_device(jnp.asarray(query["cond"]), device)
                features, norms = query_fn(
                    params,
                    target_params,
                    endpoint,
                    cond,
                    noises,
                    jnp.asarray(timestep),
                )
                features.block_until_ready()
                query_features[:, query_slot] = np.asarray(features, dtype=np.float32)
                delta_norms[:, query_slot] = np.asarray(norms, dtype=np.float32)
                query_done = query_slot + 1
                if query_done == 1 or query_done == len(queries) or query_done % 12 == 0:
                    query_elapsed = time.monotonic() - query_started
                    query_eta = query_elapsed / query_done * (len(queries) - query_done)
                    print(
                        f"[query directions] checkpoint={checkpoint_index + 1}/49 "
                        f"timestamp={timestamp_slot + 1}/10 "
                        f"queries={query_done}/{len(queries)} "
                        f"directions={args.direction_count} "
                        f"elapsed={format_duration(query_elapsed)} "
                        f"eta={format_duration(query_eta)}",
                        flush=True,
                    )
            query_features /= np.maximum(delta_norms[..., None], EPS)
            query_normalized = query_features / np.maximum(
                np.linalg.norm(query_features, axis=-1, keepdims=True), EPS
            )
            query_device = array_to_device(jnp.asarray(query_features), device)
            query_norm_device = array_to_device(jnp.asarray(query_normalized), device)

            train_noise = noises[0]
            print(
                f"[timestamp] checkpoint={checkpoint_index + 1}/49 "
                f"timestamp={timestamp_slot + 1}/10 t={int(timestep)} "
                f"directions={args.direction_count} queries={len(queries)}",
                flush=True,
            )
            for batch_index, start in enumerate(
                range(0, len(attribution_indices), args.train_batch_size), start=1
            ):
                end = min(len(attribution_indices), start + args.train_batch_size)
                padded = pad_indices_to_batch(
                    attribution_indices[start:end], args.train_batch_size
                )
                x_batch, cond_batch = make_train_batch(
                    adapter, dataset, padded, device, use_bfloat16=False
                )
                raw_train, _raw_norms, residual, _residual_norms = train_fn(
                    x_batch, cond_batch, train_noise, jnp.asarray(timestep)
                )
                raw_train.block_until_ready()
                count = end - start
                raw_train = raw_train[:count]
                adamw_train = residual[:count] + history_feature[None, :]
                adamw_direction = adamw_train / max(abs(float(checkpoint_weight)), 1e-20)
                raw_train_normalized = raw_train / jnp.maximum(
                    jnp.linalg.norm(raw_train, axis=-1, keepdims=True), EPS
                )
                adamw_train_normalized = adamw_direction / jnp.maximum(
                    jnp.linalg.norm(adamw_direction, axis=-1, keepdims=True), EPS
                )
                values = score_batch(
                    query_device,
                    query_norm_device,
                    raw_train,
                    raw_train_normalized,
                    adamw_direction,
                    adamw_train_normalized,
                )
                scores[..., start:end] += (
                    float(checkpoint_weight) / len(timesteps)
                ) * values
                if (
                    batch_index == 1
                    or batch_index == total_batches
                    or batch_index % args.progress_every_batches == 0
                ):
                    elapsed = time.monotonic() - timestamp_started
                    eta = elapsed / batch_index * (total_batches - batch_index)
                    print(
                        f"[timestamp progress] checkpoint={checkpoint_index + 1}/49 "
                        f"timestamp={timestamp_slot + 1}/10 "
                        f"{progress_bar(batch_index, total_batches)} "
                        f"batch={batch_index}/{total_batches} datapoints={end}/5000 "
                        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
                        flush=True,
                    )
        completed.append(checkpoint_index)
        save_npz_compressed_atomic(
            str(component_path),
            scores=scores,
            score_indices=attribution_indices,
            query_ids=np.asarray(query_ids, dtype=np.int32),
            completed_checkpoints=np.asarray(completed, dtype=np.int32),
            shard_index=np.asarray(args.shard_index, dtype=np.int32),
            direction_count=np.asarray(args.direction_count, dtype=np.int32),
            timesteps=timesteps,
        )
        print(f"[checkpoint] {checkpoint_index + 1}/49 saved", flush=True)
    print(f"[done] shard={args.shard_index}/{args.shard_count}", flush=True)


if __name__ == "__main__":
    main()
