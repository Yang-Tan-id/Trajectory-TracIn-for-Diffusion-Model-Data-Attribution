#!/usr/bin/env python3
"""CelebA endpoint-pollute timewise 1x50 and endpoint-TracIn mean-50.

Endpoint-pollute keeps aligned per-timestamp contractions.  Endpoint-TracIn does
not: its raw query is the mean of 50 simple-loss gradients, while its normalized
query is the ReTrac definition (normalize every full-parameter timestamp loss
gradient, then project and average).  Its train feature is the gradient of the
mean 50-timestamp loss; AdamW is applied once to that mean gradient.

Each train pass produces both the raw-gradient feature and the full hypothetical
AdamW update from the restored checkpoint optimizer state.  Train features are
streamed directly into scores and never persisted.  Raw per-loss queries and
per-loss full-L2-normalized queries are persisted for later ReTrac reuse.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
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
from dataset_config_attr40 import (
    ATTRIBUTION_INDICES_PATH,
    CLASS_NAMES,
    COMMON_CIFAR,
    DATA_ROOT,
    TRAINING_MODULE_NAME,
    _prompt_path_tag,
)
from dtrak.algorithm import build_countsketch_projector_jax
from traj_tracin.algorithm import (
    TrajAttributionConfig,
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
    tree_l2_norm,
    tree_to_device,
)


OBJECTIVES = ("endpoint_pollute", "endpoint_tracin")
TRANSFORMS = ("raw_gradient", "adamw_full")
REDUCTIONS = ("linear", "termwise_squared", "timestamp_sum_squared")
VARIANTS = (
    ("score", "raw"),
    ("score_query_normalized", "query_l2_normalized"),
    ("score_train_l2_normalized", "train_l2_normalized"),
    ("score_query_train_l2_normalized", "query_train_l2_normalized"),
)
QUERY_KEYS = (
    ("endpoint_pollute_raw", "celeba_endpoint_pollute_raw_1x50"),
    (
        "endpoint_pollute_normalized",
        "celeba_endpoint_pollute_full_l2_normalized_1x50",
    ),
    ("endpoint_tracin_raw", "celeba_endpoint_tracin_simple_loss_raw_1x50"),
    (
        "endpoint_tracin_normalized",
        "celeba_endpoint_tracin_simple_loss_full_l2_normalized_1x50",
    ),
)
EPS = 1e-12


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def make_config(args: argparse.Namespace) -> TrajAttributionConfig:
    values = dict(COMMON_CIFAR)
    values.update(
        module_name=TRAINING_MODULE_NAME,
        checkpoint_dir=str(
            CELEBA_ROOT / "result" / args.experiment / "model" / "prompted_jax"
        ),
        reference_ckpt=str(
            CELEBA_ROOT
            / "result"
            / args.experiment
            / "model"
            / "prompted_jax"
            / f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt"
        ),
        seed=args.train_seed,
        epochs=args.epochs,
        query="Smiling,Young",
        parameter_source="raw",
        ddim_steps=50,
        num_traj_snapshots=50,
        train_mc_samples=1,
        proj_dim=args.proj_dim,
        max_train_points=5000,
        random_subset=False,
        score_batch_size=args.train_batch_size,
        out_dir=str(CELEBA_ROOT / "result" / args.experiment / "tmp"),
        class_names=CLASS_NAMES,
        data_root=DATA_ROOT,
    )
    return TrajAttributionConfig(**values)


def load_queries(
    query_file: Path,
    query_ids: list[int],
    sample_root: Path,
    checkpoint: Path,
    adapter: Any,
    dataset: Any,
    cfg: TrajAttributionConfig,
) -> list[dict[str, Any]]:
    records = json.loads(query_file.read_text())["queries"]
    queries = []
    for query_id in query_ids:
        record = records[query_id]
        prompt = str(record["prompt"])
        seed = int(record["initial_seed"])
        endpoint_path = (
            sample_root
            / "cifar"
            / f"prompt_{_prompt_path_tag(prompt)}"
            / f"model_prompted_solo__ckpt_{checkpoint.stem}"
            / f"seed_{seed:06d}"
            / "final_state.npy"
        )
        if not endpoint_path.is_file():
            raise FileNotFoundError(endpoint_path)
        endpoint = np.load(endpoint_path)
        if endpoint.ndim != 4 or endpoint.shape[0] < 1:
            raise ValueError(f"invalid endpoint {endpoint_path}: {endpoint.shape}")
        queries.append(
            {
                "id": query_id,
                "prompt": prompt,
                "seed": seed,
                "endpoint": np.asarray(endpoint[:1], dtype=np.float32),
                "cond": np.asarray(adapter.make_query_cond(dataset, prompt, cfg)),
            }
        )
    return queries


def query_part_path(query_part_root: Path, query_id: int, checkpoint: int) -> Path:
    return query_part_root / f"q{query_id:03d}" / f"ckpt_{checkpoint:04d}.npz"


def query_artifact_path(
    sample_root: Path,
    reference_checkpoint: Path,
    query: dict[str, Any],
    namespace: str,
) -> Path:
    return (
        sample_root
        / "cifar"
        / f"prompt_{_prompt_path_tag(query['prompt'])}"
        / f"model_prompted_solo__ckpt_{reference_checkpoint.stem}"
        / f"seed_{int(query['seed']):06d}_query_gradient_{namespace}"
        / "traj_tracin"
        / "query_gradient_artifact.npz"
    )


def make_query_fn(adapter, model, schedule, projector):
    def one(params, target_params, endpoint, cond, noise, timestep):
        t = jnp.full((endpoint.shape[0],), timestep, dtype=jnp.int32)
        xt = q_sample(schedule, endpoint, t, noise)

        def simple_loss(candidate):
            pred = adapter.eps_apply(model, candidate, xt, t, cond)
            return jnp.mean(jnp.square(pred - noise))

        def pollute_loss(candidate):
            pred = adapter.eps_apply(model, candidate, xt, t, cond)
            target = jax.lax.stop_gradient(
                adapter.eps_apply(model, target_params, xt, t, cond)
            )
            delta = jax.lax.stop_gradient(target - pred)
            return jnp.mean(pred * delta)

        pollute_grad = jax.grad(pollute_loss)(params)
        simple_grad = jax.grad(simple_loss)(params)
        pollute_norm = tree_l2_norm(pollute_grad)
        simple_norm = tree_l2_norm(simple_grad)
        return (
            projector(pollute_grad).astype(jnp.float32),
            pollute_norm.astype(jnp.float32),
            projector(simple_grad).astype(jnp.float32),
            simple_norm.astype(jnp.float32),
        )

    return jax.jit(one)


def make_train_fn(adapter, model, schedule, projector, state, params):
    def one(x0, cond, noise, timestep):
        x0 = x0[None, ...]
        cond = cond[None, ...] if cond.ndim else cond[None]
        t = jnp.full((1,), timestep, dtype=jnp.int32)

        def loss_fn(candidate):
            xt = q_sample(schedule, x0, t, noise)
            pred = adapter.eps_apply(model, candidate, xt, t, cond)
            return jnp.mean(jnp.square(pred - noise))

        grad = jax.grad(loss_fn)(params)
        update, _ = state.tx.update(grad, state.opt_state, params)
        return (
            projector(grad).astype(jnp.float32),
            tree_l2_norm(grad).astype(jnp.float32),
            projector(update).astype(jnp.float32),
            tree_l2_norm(update).astype(jnp.float32),
        )

    return jax.jit(jax.vmap(one, in_axes=(0, 0, None, None)))


def make_endpoint_train_mean_fn(adapter, model, schedule, projector, state, params):
    """Gradient of mean-50 loss, followed by exactly one AdamW update."""

    def one(x0, cond, noises, timesteps):
        x0 = x0[None, ...]
        cond = cond[None, ...] if cond.ndim else cond[None]
        initial = jax.tree_util.tree_map(jnp.zeros_like, params)

        def body(total, inputs):
            noise, timestep = inputs
            t = jnp.full((1,), timestep, dtype=jnp.int32)

            def loss_fn(candidate):
                xt = q_sample(schedule, x0, t, noise)
                pred = adapter.eps_apply(model, candidate, xt, t, cond)
                return jnp.mean(jnp.square(pred - noise))

            grad = jax.grad(loss_fn)(params)
            total = jax.tree_util.tree_map(jnp.add, total, grad)
            return total, None

        total, _ = jax.lax.scan(body, initial, (noises, timesteps))
        denominator = jnp.asarray(timesteps.shape[0], dtype=jnp.float32)
        mean_grad = jax.tree_util.tree_map(lambda value: value / denominator, total)
        update, _ = state.tx.update(mean_grad, state.opt_state, params)
        return (
            projector(mean_grad).astype(jnp.float32),
            tree_l2_norm(mean_grad).astype(jnp.float32),
            projector(update).astype(jnp.float32),
            tree_l2_norm(update).astype(jnp.float32),
        )

    return jax.jit(jax.vmap(one, in_axes=(0, 0, None, None)))


def load_or_compute_query_checkpoint(
    *,
    checkpoint_index: int,
    query_part_root: Path,
    queries: list[dict[str, Any]],
    timesteps: np.ndarray,
    params: Any,
    target_params: Any,
    query_fn: Any,
    train_seed: int,
    device: Any,
    proj_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    paths = [query_part_path(query_part_root, int(q["id"]), checkpoint_index) for q in queries]
    if all(path.is_file() for path in paths):
        raw = np.empty((2, len(queries), len(timesteps), proj_dim), dtype=np.float32)
        norms = np.empty((2, len(queries), len(timesteps)), dtype=np.float32)
        for query_slot, path in enumerate(paths):
            with np.load(path, allow_pickle=False) as payload:
                raw[:, query_slot] = payload["raw_features"]
                norms[:, query_slot] = payload["full_l2_norms"]
        print(f"[query] checkpoint {checkpoint_index + 1}: loaded cached parts", flush=True)
        return raw, norms

    raw = np.empty((2, len(queries), len(timesteps), proj_dim), dtype=np.float32)
    norms = np.empty((2, len(queries), len(timesteps)), dtype=np.float32)
    endpoint_shape = queries[0]["endpoint"].shape
    for query_slot, query in enumerate(queries):
        endpoint = array_to_device(jnp.asarray(query["endpoint"]), device)
        cond = array_to_device(jnp.asarray(query["cond"]), device)
        for timestamp_slot, timestep in enumerate(timesteps):
            noise = jax.random.normal(
                checkpoint_timestamp_shared_noise_key(
                    train_seed, checkpoint_index, int(timestep)
                ),
                endpoint_shape,
                dtype=jnp.float32,
            )
            pollute, pollute_norm, simple, simple_norm = query_fn(
                params, target_params, endpoint, cond, noise, jnp.asarray(timestep)
            )
            simple.block_until_ready()
            raw[0, query_slot, timestamp_slot] = np.asarray(pollute, dtype=np.float32)
            raw[1, query_slot, timestamp_slot] = np.asarray(simple, dtype=np.float32)
            norms[0, query_slot, timestamp_slot] = float(pollute_norm)
            norms[1, query_slot, timestamp_slot] = float(simple_norm)
        path = paths[query_slot]
        if not path.is_file():
            save_npz_compressed_atomic(
                str(path),
                raw_features=raw[:, query_slot],
                full_l2_norms=norms[:, query_slot],
                objective_names=np.asarray(OBJECTIVES),
                checkpoint_index=np.asarray(checkpoint_index, dtype=np.int32),
                timesteps=timesteps,
                normalization=np.asarray(
                    "full_parameter_l2_per_checkpoint_timestamp_loss_before_projection"
                ),
            )
        print(
            f"[query] checkpoint {checkpoint_index + 1} "
            f"query {query_slot + 1}/{len(queries)} Q{query['id']}",
            flush=True,
        )
    return raw, norms


def update_scores(
    linear_scores: np.ndarray,
    termwise_squared_scores: np.ndarray,
    timestamp_components: np.ndarray,
    *,
    query_raw: jax.Array,
    query_normalized: jax.Array,
    raw_train: jax.Array,
    raw_train_normalized: jax.Array,
    adamw_train: jax.Array,
    adamw_train_normalized: jax.Array,
    destination: slice,
    timestamp_slot: int,
    raw_weight: float,
    adamw_weight: float,
    objective_slice: slice = slice(None),
) -> None:
    objective_count, query_count, proj_dim = query_raw.shape
    batch_size = raw_train.shape[0]
    query_raw_flat = query_raw.reshape(objective_count * query_count, proj_dim)
    query_norm_flat = query_normalized.reshape(objective_count * query_count, proj_dim)
    train_all = jnp.concatenate(
        (raw_train, raw_train_normalized, adamw_train, adamw_train_normalized),
        axis=0,
    )
    raw_dots = jnp.matmul(
        query_raw_flat, train_all.T, precision=jax.lax.Precision.HIGHEST
    )
    norm_dots = jnp.matmul(
        query_norm_flat, train_all.T, precision=jax.lax.Precision.HIGHEST
    )
    norm_dots.block_until_ready()
    raw_host = np.asarray(jax.device_get(raw_dots), dtype=np.float32).reshape(
        objective_count, query_count, 4, batch_size
    )
    norm_host = np.asarray(jax.device_get(norm_dots), dtype=np.float32).reshape(
        objective_count, query_count, 4, batch_size
    )
    # train_all block order: raw, raw-normalized, AdamW-full,
    # AdamW-full-normalized.  Query normalization changes only which of the
    # two dot products is used, so all 16 requested score families come from
    # two GEMMs instead of sixteen.
    for transform_slot, (raw_block, normalized_block, weight) in enumerate(
        ((0, 1, raw_weight), (2, 3, adamw_weight))
    ):
        values = (
            raw_host[:, :, raw_block],
            norm_host[:, :, raw_block],
            raw_host[:, :, normalized_block],
            norm_host[:, :, normalized_block],
        )
        for variant_slot, value in enumerate(values):
            contribution = float(weight) * value
            key = (objective_slice, transform_slot, variant_slot, slice(None), destination)
            linear_scores[key] += contribution
            # Match the established endpoint-pollute convention: apply the
            # checkpoint/timestamp weight after squaring the individual dot.
            termwise_squared_scores[key] += float(weight) * np.square(value)
            timestamp_components[
                objective_slice,
                transform_slot,
                variant_slot,
                :,
                timestamp_slot,
                destination,
            ] += contribution.astype(np.float32)


def update_endpoint_scores(
    linear_scores: np.ndarray,
    *,
    query_raw: jax.Array,
    query_retrac: jax.Array,
    raw_train: jax.Array,
    raw_train_normalized: jax.Array,
    adamw_train: jax.Array,
    adamw_train_normalized: jax.Array,
    destination: slice,
    raw_weight: float,
) -> None:
    """Checkpoint-level Endpoint-TracIn; there is no timestamp pairing here."""
    batch_size = raw_train.shape[0]
    train_all = jnp.concatenate(
        (raw_train, raw_train_normalized, adamw_train, adamw_train_normalized),
        axis=0,
    )
    raw_dots = jnp.matmul(query_raw, train_all.T, precision=jax.lax.Precision.HIGHEST)
    retrac_dots = jnp.matmul(
        query_retrac, train_all.T, precision=jax.lax.Precision.HIGHEST
    )
    retrac_dots.block_until_ready()
    raw_host = np.asarray(jax.device_get(raw_dots), dtype=np.float32).reshape(
        query_raw.shape[0], 4, batch_size
    )
    retrac_host = np.asarray(jax.device_get(retrac_dots), dtype=np.float32).reshape(
        query_raw.shape[0], 4, batch_size
    )
    for transform_slot, (raw_block, normalized_block, weight) in enumerate(
        ((0, 1, raw_weight), (2, 3, 1.0))
    ):
        values = (
            raw_host[:, raw_block],
            retrac_host[:, raw_block],
            raw_host[:, normalized_block],
            retrac_host[:, normalized_block],
        )
        for variant_slot, value in enumerate(values):
            linear_scores[
                1, transform_slot, variant_slot, :, destination
            ] += float(weight) * value


def merge_query_bank(
    *,
    queries: list[dict[str, Any]],
    query_part_root: Path,
    sample_root: Path,
    reference_checkpoint: Path,
    checkpoint_count: int,
    timesteps: np.ndarray,
    proj_dim: int,
    remove_parts: bool,
) -> list[Path]:
    artifacts = []
    for query in queries:
        query_id = int(query["id"])
        paths = [query_part_path(query_part_root, query_id, ckpt) for ckpt in range(checkpoint_count)]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Q{query_id}: missing {len(missing)} query checkpoint parts; first={missing[0]}"
            )
        features_by_key: dict[str, list[np.ndarray]] = {key: [] for key, _ in QUERY_KEYS}
        norms_by_objective: list[list[np.ndarray]] = [[], []]
        for path in paths:
            with np.load(path, allow_pickle=False) as payload:
                raw = np.asarray(payload["raw_features"], dtype=np.float32)
                norms = np.asarray(payload["full_l2_norms"], dtype=np.float32)
            for objective_slot, prefix in enumerate(("endpoint_pollute", "endpoint_tracin")):
                features_by_key[f"{prefix}_raw"].append(raw[objective_slot])
                features_by_key[f"{prefix}_normalized"].append(
                    raw[objective_slot] / np.maximum(norms[objective_slot, :, None], EPS)
                )
                norms_by_objective[objective_slot].append(norms[objective_slot])

        ckpt_indices = np.repeat(np.arange(checkpoint_count, dtype=np.int32), len(timesteps))
        tiled_timesteps = np.tile(timesteps, checkpoint_count)
        term_weights = np.full(
            checkpoint_count * len(timesteps), 1.0 / len(timesteps), dtype=np.float32
        )
        for key, namespace in QUERY_KEYS:
            artifact = query_artifact_path(
                sample_root, reference_checkpoint, query, namespace
            )
            if not artifact.is_file():
                objective_slot = 0 if key.startswith("endpoint_pollute") else 1
                save_npz_compressed_atomic(
                    str(artifact),
                    query_features=np.concatenate(features_by_key[key], axis=0),
                    query_full_l2_norms=np.concatenate(norms_by_objective[objective_slot]),
                    ckpt_indices=ckpt_indices,
                    timesteps=tiled_timesteps,
                    term_weights=term_weights,
                    proj_dim=np.asarray(proj_dim, dtype=np.int32),
                    query_objective=np.asarray(OBJECTIVES[objective_slot]),
                    query_normalized=np.asarray(key.endswith("normalized")),
                    normalization=np.asarray(
                        "full_parameter_l2_per_checkpoint_timestamp_loss_before_projection"
                    ),
                    query_timestamp_count=np.asarray(len(timesteps), dtype=np.int32),
                    query_mc_per_timestamp=np.asarray(1, dtype=np.int32),
                    projection_seed_rule=np.asarray(
                        "(train_seed,'traj_tracin_projection',checkpoint_index)"
                    ),
                )
            artifacts.append(artifact)
        if remove_parts:
            shutil.rmtree(query_part_root / f"q{query_id:03d}")
        print(f"[query merge] Q{query_id} complete", flush=True)
    return artifacts


def merge_scores(
    *,
    args: argparse.Namespace,
    queries: list[dict[str, Any]],
    component_prefix: Path,
    result_root: Path,
) -> None:
    shards = []
    for shard in range(args.shard_count):
        path = Path(f"{component_prefix}.shard_{shard:02d}.npz")
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            shards.append({key: np.asarray(payload[key]) for key in payload.files})
    score_indices = shards[0]["score_indices"]
    query_ids = shards[0]["query_ids"]
    for shard in shards[1:]:
        if not np.array_equal(score_indices, shard["score_indices"]):
            raise ValueError("score_indices differ between checkpoint shards")
        if not np.array_equal(query_ids, shard["query_ids"]):
            raise ValueError("query_ids differ between checkpoint shards")
    linear = sum(
        (shard["linear"] for shard in shards), np.zeros_like(shards[0]["linear"])
    )
    termwise_squared = sum(
        (shard["termwise_squared"] for shard in shards),
        np.zeros_like(shards[0]["termwise_squared"]),
    )
    timestamp_shards = []
    for shard in range(args.shard_count):
        path = Path(f"{component_prefix}.shard_{shard:02d}.timestamp_components.npy")
        if not path.is_file():
            raise FileNotFoundError(path)
        timestamp_shards.append(np.load(path, mmap_mode="r"))
    timestamp_sum_squared = np.zeros_like(linear, dtype=np.float64)
    for timestamp_slot in range(timestamp_shards[0].shape[4]):
        combined = np.zeros_like(linear, dtype=np.float64)
        for component in timestamp_shards:
            combined += component[:, :, :, :, timestamp_slot, :]
        timestamp_sum_squared += np.square(combined)
    reductions = {
        "linear": linear,
        "termwise_squared": termwise_squared,
        "timestamp_sum_squared": timestamp_sum_squared,
    }
    record_by_id = {int(query["id"]): query for query in queries}
    for query_slot, query_id_value in enumerate(query_ids):
        query_id = int(query_id_value)
        query = record_by_id[query_id]
        for objective_slot, objective in enumerate(OBJECTIVES):
            for transform_slot, transform in enumerate(TRANSFORMS):
                for reduction, reduction_scores in reductions.items():
                    if objective == "endpoint_tracin" and reduction != "linear":
                        continue
                    if objective == "endpoint_tracin":
                        namespace = (
                            f"celeba_endpoint_tracin_{transform}_"
                            "train_mean50_query_retrac50_linear"
                        )
                    else:
                        namespace = (
                            f"celeba_{objective}_{transform}_train1x50_query1x50_"
                            f"{reduction}"
                        )
                    root = (
                        result_root
                        / "attribution_score"
                        / "prompted_solo"
                        / f"train_seed_{args.train_seed}"
                        / f"query_{_prompt_path_tag(query['prompt'])}"
                        / f"initial_seed_{int(query['seed'])}"
                        / f"traj_tracin_{namespace}"
                    )
                    for variant_slot, (directory, variant) in enumerate(VARIANTS):
                        _write_score_outputs(
                            root / directory,
                            reduction_scores[
                                objective_slot,
                                transform_slot,
                                variant_slot,
                                query_slot,
                            ],
                            score_indices,
                            train_dir=component_prefix.parent,
                            query_dir=result_root / "sample_ddim_eta0_50",
                            algorithm="traj_tracin",
                            extra_manifest={
                                "score_variant": variant,
                                "score_contraction": reduction,
                                "query_objective": objective,
                                "train_feature": transform,
                                "train_timestamp_count": 50,
                                "query_timestamp_count": 50,
                                "train_mc_per_timestamp": 1,
                                "query_mc_per_timestamp": 1,
                                "normalization_order": (
                                    "endpoint_pollute: normalize each full-parameter event "
                                    "before projection; endpoint_tracin query_l2: normalize "
                                    "each query loss before projection then average; "
                                    "endpoint_tracin train_l2: normalize the mean-50 train "
                                    "gradient/update before projection"
                                ),
                                "reduction_semantics": (
                                    "linear=sum(weight*dot); "
                                    "termwise_squared=sum(weight*dot^2); "
                                    "timestamp_sum_squared=sum_t((sum_ckpt weight*dot)^2)"
                                ),
                                "persistent_train_gradient_artifact": False,
                                "adamw_full_semantics": (
                                    "endpoint_pollute: update each timestamp gradient; "
                                    "endpoint_tracin: update the mean-50 loss gradient once; "
                                    "both use restored optimizer state and include moments, "
                                    "checkpoint learning rate, clipping and weight decay"
                                ),
                            },
                        )
        print(f"[score merge] Q{query_id} complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", default=",".join(str(i) for i in range(100)))
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--timestamp-count", type=int, default=50)
    parser.add_argument("--proj-dim", type=int, default=4096)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--endpoint-train-batch-size", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--component-prefix", type=Path, required=True)
    parser.add_argument("--query-part-root", type=Path, required=True)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--keep-query-parts", action="store_true")
    args = parser.parse_args()
    if args.timestamp_count != 50:
        raise ValueError("this experiment contract requires exactly 50 timestamps")
    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError("invalid checkpoint shard")

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

    # The adapter is needed in merge mode only to build query records, so avoid
    # importing/restoring JAX model state until after the merge-only branch.
    apply_checkpoint_config(cfg, str(reference_checkpoint))
    adapter = get_adapter(cfg)
    dataset = adapter.iter_dataset(cfg)
    queries = load_queries(
        args.query_file,
        query_ids,
        sample_root,
        reference_checkpoint,
        adapter,
        dataset,
        cfg,
    )

    if args.merge_only:
        merge_query_bank(
            queries=queries,
            query_part_root=args.query_part_root,
            sample_root=sample_root,
            reference_checkpoint=reference_checkpoint,
            checkpoint_count=49,
            timesteps=np.linspace(999, 0, 50, dtype=np.int32),
            proj_dim=args.proj_dim,
            remove_parts=not args.keep_query_parts,
        )
        merge_scores(
            args=args,
            queries=queries,
            component_prefix=args.component_prefix,
            result_root=result_root,
        )
        return

    attribution_indices = np.asarray(
        np.load(ATTRIBUTION_INDICES_PATH), dtype=np.int64
    ).reshape(-1)
    if len(attribution_indices) != 5000 or len(np.unique(attribution_indices)) != 5000:
        raise ValueError(
            f"expected 5000 unique attribution indices: {ATTRIBUTION_INDICES_PATH}"
        )
    device = adapter.choose_device(cfg.prefer_device)
    model = adapter.build_model(cfg)
    state_template = adapter.build_state_template(cfg, model, device)
    schedule = schedule_to_device(
        make_diffusion_schedule(cfg.timesteps, cfg.beta_start, cfg.beta_end), device
    )
    timesteps = np.linspace(999, 0, 50, dtype=np.int32)
    component_path = Path(
        f"{args.component_prefix}.shard_{args.shard_index:02d}.npz"
    )
    active_checkpoint_path = Path(
        f"{args.component_prefix}.shard_{args.shard_index:02d}.active_checkpoint"
    )
    score_shape = (
        len(OBJECTIVES),
        len(TRANSFORMS),
        len(VARIANTS),
        len(queries),
        len(attribution_indices),
    )
    linear_scores = np.zeros(score_shape, dtype=np.float64)
    termwise_squared_scores = np.zeros(score_shape, dtype=np.float64)
    timestamp_component_path = Path(
        f"{args.component_prefix}.shard_{args.shard_index:02d}.timestamp_components.npy"
    )
    # The timestamp memmap is updated in place.  If a worker died midway
    # through a checkpoint, its partial contribution cannot be distinguished
    # from a complete one.  Restart that checkpoint shard from zero rather
    # than silently double-counting it.
    if active_checkpoint_path.is_file():
        print(
            f"[recovery] incomplete checkpoint marker found at "
            f"{active_checkpoint_path}; resetting shard accumulators",
            flush=True,
        )
        component_path.unlink(missing_ok=True)
        timestamp_component_path.unlink(missing_ok=True)
        active_checkpoint_path.unlink(missing_ok=True)
    completed: list[int] = []
    if component_path.is_file():
        with np.load(component_path, allow_pickle=False) as payload:
            linear_scores[...] = payload["linear"]
            termwise_squared_scores[...] = payload["termwise_squared"]
            completed = list(np.asarray(payload["completed_checkpoints"], dtype=np.int32))
        if not timestamp_component_path.is_file():
            raise FileNotFoundError(
                f"resume component exists but timestamp memmap is missing: "
                f"{timestamp_component_path}"
            )
        print(f"[resume] completed checkpoints={completed}", flush=True)
    timestamp_shape = score_shape[:-1] + (len(timesteps), score_shape[-1])
    if timestamp_component_path.is_file():
        timestamp_components = np.load(timestamp_component_path, mmap_mode="r+")
        if timestamp_components.shape != timestamp_shape:
            raise ValueError(
                f"timestamp component shape mismatch: {timestamp_components.shape} "
                f"!= {timestamp_shape}"
            )
    else:
        timestamp_component_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp_components = np.lib.format.open_memmap(
            timestamp_component_path,
            mode="w+",
            dtype=np.float32,
            shape=timestamp_shape,
        )
        timestamp_components[...] = 0
        timestamp_components.flush()

    for checkpoint_index in range(49):
        if checkpoint_index % args.shard_count != args.shard_index:
            continue
        if checkpoint_index in completed:
            print(f"[skip] checkpoint {checkpoint_index + 1}/49", flush=True)
            continue
        state, _ = adapter.restore_state(checkpoints[checkpoint_index], state_template)
        target_state, _ = adapter.restore_state(
            checkpoints[checkpoint_index + 1], state_template
        )
        params = tree_to_device(select_state_params(state, "raw"), device)
        target_params = tree_to_device(select_state_params(target_state, "raw"), device)
        projector = build_countsketch_projector_jax(
            params,
            args.proj_dim,
            seed_parts=(args.train_seed, "traj_tracin_projection", checkpoint_index),
            device=device,
        )
        query_fn = make_query_fn(adapter, model, schedule, projector)
        query_raw, query_norms = load_or_compute_query_checkpoint(
            checkpoint_index=checkpoint_index,
            query_part_root=args.query_part_root,
            queries=queries,
            timesteps=timesteps,
            params=params,
            target_params=target_params,
            query_fn=query_fn,
            train_seed=args.train_seed,
            device=device,
            proj_dim=args.proj_dim,
        )
        query_normalized = query_raw / np.maximum(query_norms[..., None], EPS)
        query_raw_device = array_to_device(jnp.asarray(query_raw), device)
        query_norm_device = array_to_device(jnp.asarray(query_normalized), device)
        train_fn = make_train_fn(adapter, model, schedule, projector, state, params)
        raw_checkpoint_weight = tracin_checkpoint_lr_weight(
            cfg, checkpoints[checkpoint_index], checkpoint_index, len(checkpoints), len(dataset)
        )
        active_checkpoint_path.write_text(f"{checkpoint_index}\n")

        for timestamp_slot, timestep in enumerate(timesteps):
            noise = jax.random.normal(
                checkpoint_timestamp_shared_noise_key(
                    args.train_seed, checkpoint_index, int(timestep)
                ),
                queries[0]["endpoint"].shape,
                dtype=jnp.float32,
            )
            for start in range(0, len(attribution_indices), args.train_batch_size):
                end = min(len(attribution_indices), start + args.train_batch_size)
                real_indices = attribution_indices[start:end]
                padded = pad_indices_to_batch(real_indices, args.train_batch_size)
                x_batch, cond_batch = make_train_batch(
                    adapter, dataset, padded, device, use_bfloat16=False
                )
                raw_train, raw_norms, adamw_train, adamw_norms = train_fn(
                    x_batch, cond_batch, noise, jnp.asarray(timestep)
                )
                raw_train.block_until_ready()
                raw_train = raw_train[: end - start]
                adamw_train = adamw_train[: end - start]
                raw_train_normalized = raw_train / jnp.maximum(
                    raw_norms[: end - start, None], EPS
                )
                adamw_train_normalized = adamw_train / jnp.maximum(
                    adamw_norms[: end - start, None], EPS
                )
                destination = slice(start, end)
                update_scores(
                    linear_scores,
                    termwise_squared_scores,
                    timestamp_components,
                    query_raw=query_raw_device[0:1, :, timestamp_slot],
                    query_normalized=query_norm_device[0:1, :, timestamp_slot],
                    raw_train=raw_train,
                    raw_train_normalized=raw_train_normalized,
                    adamw_train=adamw_train,
                    adamw_train_normalized=adamw_train_normalized,
                    destination=destination,
                    timestamp_slot=timestamp_slot,
                    raw_weight=float(raw_checkpoint_weight) / len(timesteps),
                    adamw_weight=1.0 / len(timesteps),
                    objective_slice=slice(0, 1),
                )
            print(
                f"[train+score] checkpoint {checkpoint_index + 1}/49 "
                f"timestamp {timestamp_slot + 1}/50 t={int(timestep)}",
                flush=True,
            )

        # Endpoint-TracIn has one feature per checkpoint, not one contraction
        # per timestamp.  Query-L2 follows paper ReTrac: normalize every query
        # loss gradient in full parameter space, then project and average.
        endpoint_query_raw = array_to_device(
            jnp.asarray(
                np.mean(query_raw[1], axis=1, dtype=np.float64).astype(np.float32)
            ),
            device,
        )
        endpoint_query_retrac = array_to_device(
            jnp.asarray(
                np.mean(query_normalized[1], axis=1, dtype=np.float64).astype(np.float32)
            ),
            device,
        )
        endpoint_noises = array_to_device(
            jnp.stack(
                [
                    jax.random.normal(
                        checkpoint_timestamp_shared_noise_key(
                            args.train_seed, checkpoint_index, int(timestep)
                        ),
                        queries[0]["endpoint"].shape,
                        dtype=jnp.float32,
                    )
                    for timestep in timesteps
                ],
                axis=0,
            ),
            device,
        )
        endpoint_timesteps = array_to_device(jnp.asarray(timesteps), device)
        endpoint_train_fn = make_endpoint_train_mean_fn(
            adapter, model, schedule, projector, state, params
        )
        endpoint_batch_size = args.endpoint_train_batch_size
        for start in range(0, len(attribution_indices), endpoint_batch_size):
            end = min(len(attribution_indices), start + endpoint_batch_size)
            real_indices = attribution_indices[start:end]
            padded = pad_indices_to_batch(real_indices, endpoint_batch_size)
            x_batch, cond_batch = make_train_batch(
                adapter, dataset, padded, device, use_bfloat16=False
            )
            raw_train, raw_norms, adamw_train, adamw_norms = endpoint_train_fn(
                x_batch, cond_batch, endpoint_noises, endpoint_timesteps
            )
            raw_train.block_until_ready()
            real_count = end - start
            raw_train = raw_train[:real_count]
            adamw_train = adamw_train[:real_count]
            update_endpoint_scores(
                linear_scores,
                query_raw=endpoint_query_raw,
                query_retrac=endpoint_query_retrac,
                raw_train=raw_train,
                raw_train_normalized=raw_train
                / jnp.maximum(raw_norms[:real_count, None], EPS),
                adamw_train=adamw_train,
                adamw_train_normalized=adamw_train
                / jnp.maximum(adamw_norms[:real_count, None], EPS),
                destination=slice(start, end),
                raw_weight=float(raw_checkpoint_weight),
            )
            if start == 0 or end == len(attribution_indices) or end % 500 == 0:
                print(
                    f"[endpoint train+score] checkpoint {checkpoint_index + 1}/49 "
                    f"datapoints={end}/{len(attribution_indices)}",
                    flush=True,
                )

        completed.append(checkpoint_index)
        timestamp_components.flush()
        save_npz_compressed_atomic(
            str(component_path),
            linear=linear_scores,
            termwise_squared=termwise_squared_scores,
            score_indices=attribution_indices,
            query_ids=np.asarray(query_ids, dtype=np.int32),
            completed_checkpoints=np.asarray(completed, dtype=np.int32),
            objective_names=np.asarray(OBJECTIVES),
            train_transforms=np.asarray(TRANSFORMS),
            variants=np.asarray([variant for _, variant in VARIANTS]),
            normalization=np.asarray(
                "full_parameter_l2_per_checkpoint_timestamp_loss_before_projection"
            ),
        )
        active_checkpoint_path.unlink(missing_ok=True)
        print(f"[checkpoint] {checkpoint_index + 1}/49 saved", flush=True)

    print(f"[done] shard {args.shard_index}/{args.shard_count}", flush=True)


if __name__ == "__main__":
    main()
