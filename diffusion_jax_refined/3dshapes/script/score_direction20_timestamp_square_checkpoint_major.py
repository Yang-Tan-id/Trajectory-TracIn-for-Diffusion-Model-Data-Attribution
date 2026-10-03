#!/usr/bin/env python3
"""Stream direction-aligned query gradients directly into final scores.

No query-gradient artifact is written. Each GPU owns a disjoint query shard,
walks the 49 current->next checkpoint pairs once, and immediately contracts
each query-gradient chunk with the matching AdamW-full train directions.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
LEGACY_ROOT = REFINE_ROOT / "legacy_jax"
for path in (SHAPES_ROOT, REFINE_ROOT, LEGACY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp

from common.stage_artifact_runner import _write_score_outputs
from dataset_config import _prompt_tag, attribution_config
from dtrak.algorithm import build_countsketch_projector_jax
from direction20_query_common import (
    direction_alignment_positions,
    load_queries,
    make_projected_query_batch_fn,
)
from traj_tracin.algorithm import (
    TrajAttributionConfig,
    apply_checkpoint_config,
    array_to_device,
    checkpoint_direction_shared_noise_key,
    get_adapter,
    list_checkpoints_sorted,
    make_diffusion_schedule,
    schedule_to_device,
    select_state_params,
    tree_to_device,
)


VARIANTS = (
    ("score", "raw"),
    ("score_query_normalized", "query_l2_normalized"),
    ("score_train_l2_normalized", "train_l2_normalized"),
    ("score_query_train_l2_normalized", "query_train_l2_normalized"),
)


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def score_root(result_root, train_seed, prompt, seed, namespace) -> Path:
    return (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
        / f"query_{_prompt_tag(prompt)}"
        / f"initial_seed_{seed}"
        / f"traj_tracin_{namespace}"
    )


def all_outputs_exist(root: Path) -> bool:
    return all((root / directory / "scores.npy").is_file() for directory, _ in VARIANTS)


def load_train_part(path: Path, direction_count: int):
    with np.load(path, allow_pickle=False) as payload:
        residual = np.asarray(payload["train_features"], dtype=np.float32)
        history = np.asarray(payload["optimizer_history_features"], dtype=np.float32)
        indices = np.asarray(payload["score_indices"], dtype=np.int64)
        directions = np.asarray(payload["direction_indices"], dtype=np.int32)
        train_timestamps = np.asarray(payload["train_timesteps_used"], dtype=np.int32)
        noise_rule = str(np.asarray(payload["train_noise_seed_rule"]).item())
    if residual.ndim != 3 or residual.shape[0] != direction_count:
        raise ValueError(f"{path}: unexpected train shape {residual.shape}")
    if history.shape != (direction_count, residual.shape[-1]):
        raise ValueError(f"{path}: unexpected history shape {history.shape}")
    if not np.array_equal(directions, np.arange(direction_count)):
        raise ValueError(f"{path}: direction order is not 0..D-1")
    expected_rule = (
        "checkpoint_direction_shared_noise_key("
        "train_seed,checkpoint_index,direction_index)"
    )
    if noise_rule != expected_rule:
        raise ValueError(f"{path}: train noise rule mismatch: {noise_rule}")
    return residual + history[:, None, :], indices, train_timestamps


def make_chunk_dot_fn(eps: float = 1e-8):
    @jax.jit
    def chunk_dots(train_direction, query):
        # train_direction: [point, projection]
        # query: [query, timestamp, projection]; result: [query, timestamp, point]
        raw = jnp.einsum(
            "kp,qtp->qtk",
            train_direction,
            query,
            precision=jax.lax.Precision.HIGHEST,
        )
        query_norm = query / jnp.maximum(
            jnp.linalg.norm(query, axis=-1, keepdims=True), eps
        )
        query_l2 = jnp.einsum(
            "kp,qtp->qtk",
            train_direction,
            query_norm,
            precision=jax.lax.Precision.HIGHEST,
        )
        train_norm = jnp.maximum(
            jnp.linalg.norm(train_direction, axis=-1), eps
        )[None, None, :]
        return jnp.stack(
            (raw, query_l2, raw / train_norm, query_l2 / train_norm), axis=1
        )

    return chunk_dots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", required=True)
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train-artifact", type=Path, required=True)
    parser.add_argument("--score-namespace", required=True)
    parser.add_argument("--direction-count", type=int, default=20)
    parser.add_argument("--timestamp-count", type=int, default=100)
    parser.add_argument("--query-batch-size", type=int, default=2)
    parser.add_argument("--term-chunk-size", type=int, default=40)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    all_query_ids = parse_ints(args.query_ids)
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    query_ids = all_query_ids[args.shard_index :: args.shard_count]
    if not query_ids:
        print(f"[skip] empty query shard {args.shard_index}/{args.shard_count}")
        return

    os.environ.update(
        EXPERIMENT_TAG=args.experiment,
        TRAIN_SEED=str(args.train_seed),
        JAX_EPOCHS=str(args.epochs),
        TRAJ_PARAMETER_SOURCE="raw",
        TRAJ_TRACIN_PROJ_DIM="4096",
    )
    cfg_values = attribution_config("traj_tracin")
    result_root = SHAPES_ROOT / "result" / args.experiment
    checkpoint_root = result_root / "model" / "prompted_jax"
    cfg_values.update(
        checkpoint_dir=str(checkpoint_root),
        reference_ckpt=str(
            checkpoint_root / f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt"
        ),
        seed=args.train_seed,
        epochs=args.epochs,
        parameter_source="raw",
        num_traj_snapshots=args.timestamp_count,
        proj_dim=4096,
        out_dir=str(result_root / "tmp"),
    )
    cfg = TrajAttributionConfig(**cfg_values)
    checkpoints = list_checkpoints_sorted(cfg.checkpoint_dir)
    if len(checkpoints) != 50:
        raise ValueError(f"expected 50 checkpoints, found {len(checkpoints)}")
    reference_checkpoint = Path(cfg.reference_ckpt or checkpoints[-1])
    sample_root = result_root / "sample_ddim_eta0_1000"

    apply_checkpoint_config(cfg, str(reference_checkpoint))
    adapter = get_adapter(cfg)
    device = adapter.choose_device(cfg.prefer_device)
    dataset = adapter.iter_dataset(cfg)
    model = adapter.build_model(cfg)
    state_template = adapter.build_state_template(cfg, model, device)
    schedule = schedule_to_device(
        make_diffusion_schedule(cfg.timesteps, cfg.beta_start, cfg.beta_end), device
    )
    queries = load_queries(
        args.query_file, query_ids, sample_root, reference_checkpoint,
        adapter, dataset, cfg,
    )
    for query in queries:
        query["output_root"] = score_root(
            result_root, args.train_seed, query["prompt"], query["seed"],
            args.score_namespace,
        )
    queries = [q for q in queries if not all_outputs_exist(q["output_root"])]
    if not queries:
        print("[done] every query in this shard already has four score variants")
        return

    positions_one = direction_alignment_positions(args.timestamp_count)
    timesteps_one = (999 - positions_one).astype(np.int32)
    term_count = args.direction_count * args.timestamp_count
    alpha_bars = np.asarray(jax.device_get(schedule.alphas_cumprod), dtype=np.float32)
    train_part_dir = Path(str(args.train_artifact) + ".parts")
    dot_fn = make_chunk_dot_fn()
    accumulators = None
    score_indices = None

    print(
        f"[stream] shard={args.shard_index}/{args.shard_count} "
        f"queries={len(queries)} terms/query/checkpoint={term_count}; "
        "query gradients written to disk=0",
        flush=True,
    )
    for ckpt_i in range(49):
        train_part = train_part_dir / f"ckpt_{ckpt_i:04d}.npz"
        if not train_part.is_file():
            raise FileNotFoundError(train_part)
        train_full, indices, train_timesteps = load_train_part(
            train_part, args.direction_count
        )
        if not np.array_equal(train_timesteps, timesteps_one):
            raise ValueError(f"{train_part}: train/query timestep grid mismatch")
        if score_indices is None:
            score_indices = indices
            # [query, variant, direction*timestamp, train point]. For 50
            # queries this is ~8 GB host RAM and replaces ~75 GB of query
            # gradients per GPU shard.
            accumulators = np.zeros(
                (len(queries), len(VARIANTS), term_count, len(indices)),
                dtype=np.float32,
            )
        elif not np.array_equal(score_indices, indices):
            raise ValueError(f"{train_part}: score indices changed")

        current_state, _ = adapter.restore_state(checkpoints[ckpt_i], state_template)
        target_state, _ = adapter.restore_state(checkpoints[ckpt_i + 1], state_template)
        params = tree_to_device(select_state_params(current_state, "raw"), device)
        target_params = tree_to_device(select_state_params(target_state, "raw"), device)
        projector = build_countsketch_projector_jax(
            params, 4096,
            seed_parts=(args.train_seed, "traj_tracin_projection", ckpt_i),
            device=device,
        )
        query_fn = make_projected_query_batch_fn(adapter, model, projector)
        train_device = jax.device_put(jnp.asarray(train_full))
        endpoint_shape = queries[0]["endpoint"].shape
        noise_bank = np.stack(
            [
                np.asarray(
                    jax.device_get(
                        jax.random.normal(
                            checkpoint_direction_shared_noise_key(
                                args.train_seed, ckpt_i, direction_i
                            ),
                            endpoint_shape,
                            dtype=jnp.float32,
                        )
                    ),
                    dtype=np.float32,
                )
                for direction_i in range(args.direction_count)
            ],
            axis=0,
        )

        for query_start in range(0, len(queries), args.query_batch_size):
            real = queries[query_start : query_start + args.query_batch_size]
            padded = list(real)
            while len(padded) < args.query_batch_size:
                padded.append(padded[-1])
            conds = np.stack([item["cond"] for item in padded], axis=0)

            for direction_i in range(args.direction_count):
                direction_offset = direction_i * args.timestamp_count
                for timestamp_start in range(
                    0, args.timestamp_count, args.term_chunk_size
                ):
                    timestamp_end = min(
                        args.timestamp_count,
                        timestamp_start + args.term_chunk_size,
                    )
                    real_indices = np.arange(
                        timestamp_start, timestamp_end, dtype=np.int32
                    )
                    chunk_indices = real_indices
                    if len(chunk_indices) < args.term_chunk_size:
                        chunk_indices = np.pad(
                            chunk_indices,
                            (0, args.term_chunk_size - len(chunk_indices)),
                            mode="edge",
                        )
                    chunk_t = timesteps_one[chunk_indices]
                    xt_by_query = []
                    for item in padded:
                        endpoint = item["endpoint"]
                        xt_by_query.append(
                            np.stack(
                                [
                                    np.sqrt(float(alpha_bars[int(t)])) * endpoint
                                    + np.sqrt(
                                        max(0.0, 1.0 - float(alpha_bars[int(t)]))
                                    )
                                    * noise_bank[direction_i]
                                    for t in chunk_t
                                ],
                                axis=0,
                            )
                        )
                    query_features, _delta_norms = query_fn(
                        params,
                        target_params,
                        array_to_device(
                            jnp.asarray(np.stack(xt_by_query), dtype=jnp.float32),
                            device,
                        ),
                        array_to_device(
                            jnp.asarray(chunk_t, dtype=jnp.int32), device
                        ),
                        array_to_device(jnp.asarray(conds), device),
                    )
                    dots = dot_fn(train_device[direction_i], query_features)
                    dots.block_until_ready()
                    real_query_count = len(real)
                    real_term_count = len(real_indices)
                    term_start = direction_offset + timestamp_start
                    term_end = direction_offset + timestamp_end
                    accumulators[
                        query_start : query_start + real_query_count,
                        :,
                        term_start:term_end,
                        :,
                    ] += np.asarray(
                        dots[:real_query_count, :, :real_term_count, :],
                        dtype=np.float32,
                    )

        del train_device, train_full, current_state, target_state
        print(
            f"[stream] checkpoint={ckpt_i + 1}/49 complete | "
            f"queries={len(queries)} | query artifacts written=0",
            flush=True,
        )

    assert accumulators is not None and score_indices is not None
    for query_i, query in enumerate(queries):
        # Query-at-a-time float64 reduction avoids duplicating the full shard
        # accumulator during the final square.
        final_scores = np.square(accumulators[query_i].astype(np.float64)).sum(axis=1)
        for variant_i, (directory, variant_name) in enumerate(VARIANTS):
            output_dir = query["output_root"] / directory
            if (output_dir / "scores.npy").is_file():
                continue
            _write_score_outputs(
                output_dir,
                final_scores[variant_i],
                score_indices,
                train_dir=train_part_dir,
                query_dir=sample_root,
                algorithm="traj_tracin",
                extra_manifest={
                    "score_variant": variant_name,
                    "score_contraction": "timestamp_sum_squared",
                    "checkpoint_reduction": "sum_then_square",
                    "direction_timestamp_reduction": "sum_after_square",
                    "direction_count": args.direction_count,
                    "timestamp_count": args.timestamp_count,
                    "train_feature": "adamw_full_residual_plus_history",
                    "query_objective": (
                        "normalized_polluted_endpoint_next_checkpoint_delta_projection"
                    ),
                    "checkpoint_weighting": "inside_adamw_update_only",
                    "strict_noise_direction_alignment": True,
                    "checkpoint_major_streaming_query": True,
                    "query_gradient_artifact_written": False,
                },
            )
        print(f"[score] query {query['id']} complete", flush=True)


if __name__ == "__main__":
    main()
