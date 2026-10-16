#!/usr/bin/env python3
"""Endpoint-only CelebA TracIn for selected checkpoints and timesteps.

This is the checkpoint-level Shape3D-style endpoint estimator: form the
gradient of the mean selected-timestep diffusion loss on both query and train
sides, project both features, optionally normalize the projected vectors, and
take their checkpoint-weighted inner product.  AdamW-full is applied exactly
once to each mean train gradient.  Train features are streamed into scores and
are never persisted.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import jax
import jax.numpy as jnp

import run_endpoint_timewise_1x50 as base
from common.stage_artifact_runner import _write_score_outputs
from dtrak.algorithm import build_countsketch_projector_jax
from traj_tracin.algorithm import (
    array_to_device,
    apply_checkpoint_config,
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
    tree_l2_norm,
    tree_to_device,
)


TRANSFORMS = ("raw_gradient", "adamw_full")
VARIANTS = base.VARIANTS
EPS = base.EPS


def make_query_fn(adapter, model, schedule, projector):
    def one(params, endpoint, cond, noise, timestep):
        t = jnp.full((endpoint.shape[0],), timestep, dtype=jnp.int32)
        xt = q_sample(schedule, endpoint, t, noise)

        def loss_fn(candidate):
            prediction = adapter.eps_apply(model, candidate, xt, t, cond)
            return jnp.mean(jnp.square(prediction - noise))

        gradient = jax.grad(loss_fn)(params)
        return (
            projector(gradient).astype(jnp.float32),
            tree_l2_norm(gradient).astype(jnp.float32),
        )

    return jax.jit(one)


def namespace(
    transform: str,
    timestep_tag: str,
    checkpoint_count: int,
    timestamp_count: int,
    reduction: str = "linear",
    per_timestep_query: bool = False,
) -> str:
    if not per_timestep_query:
        return (
            f"celeba_endpoint_tracin_{transform}_"
            f"train_mean{timestamp_count}_query_mean{timestamp_count}_linear_"
            f"{timestep_tag}_checkpoint{checkpoint_count}_shape3d_semantics"
        )
    return (
        f"celeba_endpoint_tracin_{transform}_"
        f"train_mean{timestamp_count}_query_per_timestep_raw_or_l2_{reduction}_"
        f"{timestep_tag}_checkpoint{checkpoint_count}_shape3d_semantics"
    )


def merge(args: argparse.Namespace, queries: list[dict], result_root: Path) -> None:
    shards = []
    for shard_index in range(args.shard_count):
        path = Path(f"{args.component_prefix}.shard_{shard_index:02d}.npz")
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            shards.append({key: np.asarray(payload[key]) for key in payload.files})

    score_indices = shards[0]["score_indices"]
    query_ids = shards[0]["query_ids"]
    timesteps = shards[0]["timesteps"]
    checkpoint_indices = shards[0]["selected_checkpoint_indices"]
    for shard in shards:
        if not np.array_equal(shard["score_indices"], score_indices):
            raise ValueError("score indices differ between shards")
        if not np.array_equal(shard["query_ids"], query_ids):
            raise ValueError("query ids differ between shards")
        if not np.array_equal(shard["timesteps"], timesteps):
            raise ValueError("timesteps differ between shards")
        if not np.array_equal(shard["selected_checkpoint_indices"], checkpoint_indices):
            raise ValueError("checkpoint selections differ between shards")
        shard_index = int(shard["shard_index"])
        expected = checkpoint_indices[shard_index :: args.shard_count]
        completed = np.sort(shard["completed_checkpoints"].astype(np.int32))
        if not np.array_equal(completed, np.sort(expected)):
            raise ValueError(
                f"incomplete shard {shard_index}: {completed.tolist()} != {expected.tolist()}"
            )

    if args.per_timestep_query_reductions:
        linear = sum(
            (shard["linear_scores"] for shard in shards),
            np.zeros_like(shards[0]["linear_scores"], dtype=np.float64),
        )
        termwise_squared = sum(
            (shard["termwise_squared_scores"] for shard in shards),
            np.zeros_like(shards[0]["termwise_squared_scores"], dtype=np.float64),
        )
        timestamp_components = sum(
            (shard["timestamp_components"] for shard in shards),
            np.zeros_like(shards[0]["timestamp_components"], dtype=np.float64),
        )
        reductions = {
            "linear": linear,
            "termwise_squared": termwise_squared,
            "timestamp_sum_squared": np.sum(
                np.square(timestamp_components), axis=-2, dtype=np.float64
            ),
            "total_sum_squared": np.square(linear),
        }
    else:
        scores = sum(
            (shard["scores"] for shard in shards),
            np.zeros_like(shards[0]["scores"], dtype=np.float64),
        )
        reductions = {"linear": scores}
    record_by_id = {int(query["id"]): query for query in queries}
    namespaces = []
    for reduction, reduction_scores in reductions.items():
        for transform_slot, transform in enumerate(TRANSFORMS):
            score_namespace = namespace(
                transform,
                args.timestep_tag,
                len(checkpoint_indices),
                len(timesteps),
                reduction=reduction,
                per_timestep_query=args.per_timestep_query_reductions,
            )
            namespaces.append(score_namespace)
            for query_slot, query_id in enumerate(query_ids):
                query = record_by_id[int(query_id)]
                root = (
                    result_root
                    / "attribution_score"
                    / "prompted_solo"
                    / f"train_seed_{args.train_seed}"
                    / f"query_{base._prompt_path_tag(query['prompt'])}"
                    / f"initial_seed_{int(query['seed'])}"
                    / f"traj_tracin_{score_namespace}"
                )
                for variant_slot, (directory, variant) in enumerate(VARIANTS):
                    _write_score_outputs(
                        root / directory,
                        reduction_scores[transform_slot, variant_slot, query_slot],
                        score_indices,
                        train_dir=args.component_prefix.parent,
                        query_dir=result_root / "sample_ddim_eta0_50",
                        algorithm="traj_tracin",
                        extra_manifest={
                            "query_objective": "endpoint_tracin",
                            "train_feature": transform,
                            "score_variant": variant,
                            "score_contraction": reduction,
                            "timesteps": [int(value) for value in timesteps],
                            "timestep_tag": args.timestep_tag,
                            "selected_checkpoint_indices": [
                                int(value) for value in checkpoint_indices
                            ],
                            "train_timestamp_count": len(timesteps),
                            "query_timestamp_count": len(timesteps),
                            "train_mc_per_timestamp": 1,
                            "query_mc_per_timestamp": 1,
                            "normalization_order": (
                                "query raw variant: project each timestep gradient without "
                                "normalization; query_l2 variant: full-parameter L2-normalize "
                                "each timestep gradient before projection; train variants: "
                                "optionally projected-L2-normalize the gradient of the mean "
                                "selected-timestep train loss"
                                if args.per_timestep_query_reductions
                                else "mean selected-timestep projected gradients first; "
                                "then optional projected query/train L2 normalization"
                            ),
                            "reduction_semantics": (
                                "linear=sum_ckpt(weight*mean_t(dot)); "
                                "termwise_squared=sum_ckpt(weight*mean_t(dot^2)); "
                                "timestamp_sum_squared=sum_t((sum_ckpt "
                                "weight*dot/num_timestamps)^2); "
                                "total_sum_squared=(sum_ckpt(weight*mean_t(dot)))^2"
                            ),
                            "adamw_full_semantics": (
                                "apply restored checkpoint AdamW exactly once to the "
                                "gradient of the mean selected-timestep train loss"
                            ),
                            "persistent_train_gradient_artifact": False,
                        },
                    )
            print(f"[write] {transform} {reduction} complete", flush=True)

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
    parser.add_argument("--timesteps", type=base.parse_ints, required=True)
    parser.add_argument("--timestep-tag", required=True)
    parser.add_argument("--checkpoint-indices", type=base.parse_ints, required=True)
    parser.add_argument("--proj-dim", type=int, default=4096)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--progress-every-batches", type=int, default=250)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--component-prefix", type=Path, required=True)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument(
        "--per-timestep-query-reductions",
        action="store_true",
        help=(
            "Keep query timestep gradients separate and write linear, termwise-squared, "
            "timestamp-sum-squared, and total-sum-squared scores. Query-L2 variants "
            "normalize every full-space timestep gradient before projection."
        ),
    )
    args = parser.parse_args()

    if not args.timesteps or len(set(args.timesteps)) != len(args.timesteps):
        raise ValueError("--timesteps must be non-empty and unique")
    if min(args.timesteps) < 0 or max(args.timesteps) > 999:
        raise ValueError("--timesteps must lie in [0, 999]")
    if not args.checkpoint_indices or len(set(args.checkpoint_indices)) != len(
        args.checkpoint_indices
    ):
        raise ValueError("--checkpoint-indices must be non-empty and unique")
    if min(args.checkpoint_indices) < 0 or max(args.checkpoint_indices) > 48:
        raise ValueError("checkpoint source indices must lie in [0, 48]")
    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError("invalid checkpoint shard")
    if args.train_batch_size <= 0 or args.progress_every_batches <= 0:
        raise ValueError("batch size and progress interval must be positive")

    query_ids = base.parse_ints(args.query_ids)
    os.environ.update(
        EXPERIMENT_TAG=args.experiment,
        TRAIN_SEED=str(args.train_seed),
        JAX_EPOCHS=str(args.epochs),
        TRAJ_PARAMETER_SOURCE="raw",
        TRAJ_TRACIN_PROJ_DIM=str(args.proj_dim),
    )
    cfg = base.make_config(args)
    result_root = base.CELEBA_ROOT / "result" / args.experiment
    sample_root = result_root / "sample_ddim_eta0_50"
    checkpoints = list_checkpoints_sorted(cfg.checkpoint_dir)
    if len(checkpoints) != 50:
        raise ValueError(f"expected 50 checkpoints, found {len(checkpoints)}")
    reference_checkpoint = Path(cfg.reference_ckpt or checkpoints[-1])

    apply_checkpoint_config(cfg, str(reference_checkpoint))
    adapter = get_adapter(cfg)
    dataset = adapter.iter_dataset(cfg)
    queries = base.load_queries(
        args.query_file,
        query_ids,
        sample_root,
        reference_checkpoint,
        adapter,
        dataset,
        cfg,
    )

    if args.merge_only:
        merge(args, queries, result_root)
        return

    attribution_indices = np.asarray(
        np.load(base.ATTRIBUTION_INDICES_PATH), dtype=np.int64
    ).reshape(-1)
    if len(attribution_indices) != 5000 or len(np.unique(attribution_indices)) != 5000:
        raise ValueError("expected exactly 5000 unique attribution indices")

    device = adapter.choose_device(cfg.prefer_device)
    model = adapter.build_model(cfg)
    state_template = adapter.build_state_template(cfg, model, device)
    schedule = schedule_to_device(
        make_diffusion_schedule(cfg.timesteps, cfg.beta_start, cfg.beta_end), device
    )
    timesteps = np.asarray(args.timesteps, dtype=np.int32)
    checkpoint_indices = np.asarray(args.checkpoint_indices, dtype=np.int32)
    component_path = Path(
        f"{args.component_prefix}.shard_{args.shard_index:02d}.npz"
    )
    score_shape = (
        len(TRANSFORMS),
        len(VARIANTS),
        len(queries),
        len(attribution_indices),
    )
    scores = np.zeros(score_shape, dtype=np.float64)
    termwise_squared_scores = (
        np.zeros(score_shape, dtype=np.float64)
        if args.per_timestep_query_reductions
        else None
    )
    timestamp_components = (
        np.zeros(
            score_shape[:-1] + (len(timesteps), score_shape[-1]), dtype=np.float64
        )
        if args.per_timestep_query_reductions
        else None
    )
    completed: list[int] = []
    if component_path.is_file():
        with np.load(component_path, allow_pickle=False) as payload:
            if not np.array_equal(payload["timesteps"], timesteps):
                raise ValueError("resume timesteps do not match")
            if not np.array_equal(
                payload["selected_checkpoint_indices"], checkpoint_indices
            ):
                raise ValueError("resume checkpoint selection does not match")
            if args.per_timestep_query_reductions:
                linear_key = "linear_scores"
                if linear_key not in payload:
                    raise ValueError(
                        "resume component predates --per-timestep-query-reductions"
                    )
                scores[...] = payload[linear_key]
                assert termwise_squared_scores is not None
                assert timestamp_components is not None
                termwise_squared_scores[...] = payload["termwise_squared_scores"]
                timestamp_components[...] = payload["timestamp_components"]
            else:
                scores[...] = payload["scores"]
            completed = [int(value) for value in payload["completed_checkpoints"]]
        print(f"[resume] completed checkpoints={completed}", flush=True)

    endpoint_shape = tuple(queries[0]["endpoint"].shape)
    selected_for_shard = checkpoint_indices[
        args.shard_index :: args.shard_count
    ]
    for local_slot, checkpoint_index_value in enumerate(selected_for_shard, start=1):
        checkpoint_index = int(checkpoint_index_value)
        if checkpoint_index in completed:
            print(f"[skip] checkpoint source index={checkpoint_index}", flush=True)
            continue
        checkpoint_started = time.monotonic()
        state, _ = adapter.restore_state(checkpoints[checkpoint_index], state_template)
        params = tree_to_device(select_state_params(state, "raw"), device)
        projector = build_countsketch_projector_jax(
            params,
            args.proj_dim,
            seed_parts=(args.train_seed, "traj_tracin_projection", checkpoint_index),
            device=device,
        )
        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
        adamw_history_update, _ = state.tx.update(
            zero_grads, state.opt_state, params
        )
        adamw_history_update = tree_to_device(adamw_history_update, device)
        adamw_history_feature = array_to_device(
            jnp.asarray(np.asarray(projector(adamw_history_update), dtype=np.float32)),
            device,
        )

        query_fn = make_query_fn(adapter, model, schedule, projector)
        query_features = np.empty(
            (len(queries), len(timesteps), args.proj_dim), dtype=np.float32
        )
        query_full_norms = (
            np.empty((len(queries), len(timesteps)), dtype=np.float32)
            if args.per_timestep_query_reductions
            else None
        )
        for query_slot, query in enumerate(queries):
            endpoint = array_to_device(jnp.asarray(query["endpoint"]), device)
            cond = array_to_device(jnp.asarray(query["cond"]), device)
            noises = jax.random.normal(
                base.endpoint_noise_key(
                    args.train_seed,
                    int(query["id"]),
                    checkpoint_index,
                    100_001,
                ),
                (len(timesteps),) + endpoint_shape,
                dtype=jnp.float32,
            )
            for timestep_slot, timestep in enumerate(timesteps):
                feature, full_norm = query_fn(
                    params,
                    endpoint,
                    cond,
                    noises[timestep_slot],
                    jnp.asarray(timestep),
                )
                feature.block_until_ready()
                query_features[query_slot, timestep_slot] = np.asarray(
                    feature, dtype=np.float32
                )
                if query_full_norms is not None:
                    query_full_norms[query_slot, timestep_slot] = float(full_norm)
            done = query_slot + 1
            if done == 1 or done == len(queries) or done % 12 == 0:
                elapsed = time.monotonic() - checkpoint_started
                eta = elapsed / done * (len(queries) - done)
                print(
                    f"[query] checkpoint={checkpoint_index}->{checkpoint_index + 1} "
                    f"queries={done}/{len(queries)} elapsed={base.format_duration(elapsed)} "
                    f"eta={base.format_duration(eta)}",
                    flush=True,
                )

        query_mean = np.mean(query_features, axis=1, dtype=np.float64).astype(np.float32)
        query_mean_device = array_to_device(jnp.asarray(query_mean), device)
        query_mean_normalized = query_mean_device / jnp.maximum(
            jnp.linalg.norm(query_mean_device, axis=-1, keepdims=True), EPS
        )
        if args.per_timestep_query_reductions:
            assert query_full_norms is not None
            query_raw_terms = array_to_device(
                jnp.asarray(query_features.reshape(-1, args.proj_dim)), device
            )
            query_l2_terms = array_to_device(
                jnp.asarray(
                    (
                        query_features
                        / np.maximum(query_full_norms[..., None], EPS)
                    ).reshape(-1, args.proj_dim)
                ),
                device,
            )

        endpoint_train_fn = base.make_endpoint_train_mean_fn(
            adapter,
            model,
            schedule,
            projector,
            state,
            params,
            adamw_history_update,
        )
        endpoint_timesteps = array_to_device(jnp.asarray(timesteps), device)
        checkpoint_weight = tracin_checkpoint_lr_weight(
            cfg,
            checkpoints[checkpoint_index],
            checkpoint_index,
            len(checkpoints),
            len(dataset),
        )
        total_batches = (
            len(attribution_indices) + args.train_batch_size - 1
        ) // args.train_batch_size
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
            endpoint_rngs = array_to_device(
                jnp.stack(
                    [
                        jax.random.PRNGKey(
                            args.train_seed
                            + 700_000 * (checkpoint_index + 1)
                            + start
                            + offset
                        )
                        for offset in range(args.train_batch_size)
                    ],
                    axis=0,
                ),
                device,
            )
            raw_train, _raw_norms, adamw_residual, _residual_norms = endpoint_train_fn(
                x_batch, cond_batch, endpoint_rngs, endpoint_timesteps
            )
            raw_train.block_until_ready()
            count = end - start
            raw_train = raw_train[:count]
            adamw_residual = adamw_residual[:count]
            adamw_train = adamw_residual + adamw_history_feature[None, :]
            adamw_direction = adamw_train / max(
                abs(float(checkpoint_weight)), 1e-20
            )
            raw_train_normalized = raw_train / jnp.maximum(
                jnp.linalg.norm(raw_train, axis=-1, keepdims=True), EPS
            )
            adamw_train_normalized = adamw_direction / jnp.maximum(
                jnp.linalg.norm(adamw_direction, axis=-1, keepdims=True), EPS
            )

            train_all = jnp.concatenate(
                (
                    raw_train,
                    raw_train_normalized,
                    adamw_direction,
                    adamw_train_normalized,
                ),
                axis=0,
            )
            if args.per_timestep_query_reductions:
                raw_dots = jnp.matmul(
                    query_raw_terms,
                    train_all.T,
                    precision=jax.lax.Precision.HIGHEST,
                )
                normalized_dots = jnp.matmul(
                    query_l2_terms,
                    train_all.T,
                    precision=jax.lax.Precision.HIGHEST,
                )
            else:
                raw_dots = jnp.matmul(
                    query_mean_device,
                    train_all.T,
                    precision=jax.lax.Precision.HIGHEST,
                )
                normalized_dots = jnp.matmul(
                    query_mean_normalized,
                    train_all.T,
                    precision=jax.lax.Precision.HIGHEST,
                )
            normalized_dots.block_until_ready()
            host_shape = (
                (len(queries), len(timesteps), 4, count)
                if args.per_timestep_query_reductions
                else (len(queries), 4, count)
            )
            raw_host = np.asarray(jax.device_get(raw_dots), dtype=np.float32).reshape(
                host_shape
            )
            normalized_host = np.asarray(
                jax.device_get(normalized_dots), dtype=np.float32
            ).reshape(host_shape)
            for transform_slot, (raw_block, normalized_block) in enumerate(
                ((0, 1), (2, 3))
            ):
                if args.per_timestep_query_reductions:
                    values = (
                        raw_host[:, :, raw_block],
                        normalized_host[:, :, raw_block],
                        raw_host[:, :, normalized_block],
                        normalized_host[:, :, normalized_block],
                    )
                else:
                    values = (
                        raw_host[:, raw_block],
                        normalized_host[:, raw_block],
                        raw_host[:, normalized_block],
                        normalized_host[:, normalized_block],
                    )
                for variant_slot, value in enumerate(values):
                    if args.per_timestep_query_reductions:
                        assert termwise_squared_scores is not None
                        assert timestamp_components is not None
                        weighted = float(checkpoint_weight) * value
                        scores[transform_slot, variant_slot, :, start:end] += np.mean(
                            weighted, axis=1, dtype=np.float64
                        )
                        termwise_squared_scores[
                            transform_slot, variant_slot, :, start:end
                        ] += float(checkpoint_weight) * np.mean(
                            np.square(value), axis=1, dtype=np.float64
                        )
                        timestamp_components[
                            transform_slot, variant_slot, :, :, start:end
                        ] += weighted / len(timesteps)
                    else:
                        scores[transform_slot, variant_slot, :, start:end] += (
                            float(checkpoint_weight) * value
                        )

            if (
                batch_index == 1
                or batch_index == total_batches
                or batch_index % args.progress_every_batches == 0
            ):
                elapsed = time.monotonic() - checkpoint_started
                eta = elapsed / batch_index * (total_batches - batch_index)
                print(
                    f"[train+score] checkpoint={checkpoint_index}->{checkpoint_index + 1} "
                    f"{base.progress_bar(batch_index, total_batches)} "
                    f"batch={batch_index}/{total_batches} datapoints={end}/5000 "
                    f"elapsed={base.format_duration(elapsed)} "
                    f"batch_eta={base.format_duration(eta)}",
                    flush=True,
                )

        completed.append(checkpoint_index)
        component_payload = dict(
            score_indices=attribution_indices,
            query_ids=np.asarray(query_ids, dtype=np.int32),
            completed_checkpoints=np.asarray(completed, dtype=np.int32),
            selected_checkpoint_indices=checkpoint_indices,
            timesteps=timesteps,
            shard_index=np.asarray(args.shard_index, dtype=np.int32),
        )
        if args.per_timestep_query_reductions:
            assert termwise_squared_scores is not None
            assert timestamp_components is not None
            component_payload.update(
                linear_scores=scores,
                termwise_squared_scores=termwise_squared_scores,
                timestamp_components=timestamp_components,
            )
        else:
            component_payload["scores"] = scores
        save_npz_compressed_atomic(str(component_path), **component_payload)
        print(
            f"[checkpoint] {checkpoint_index}->{checkpoint_index + 1} saved "
            f"({local_slot}/{len(selected_for_shard)})",
            flush=True,
        )

    print(f"[done] shard={args.shard_index}/{args.shard_count}", flush=True)


if __name__ == "__main__":
    main()
