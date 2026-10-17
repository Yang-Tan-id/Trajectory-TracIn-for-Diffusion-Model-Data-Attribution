#!/usr/bin/env python3
"""CelebA ReTrac with per-loss full-space L2 normalization.

The query side reuses the 49-checkpoint bank produced by
run_endpoint_timewise_1x50.py.  Each stored checkpoint feature is the mean of
50 independently normalized query-loss gradients.  The train side follows the
working endpoint-pollute batch-one gradient path while replaying each event's
sample, noise, timestep, checkpoint, and learning-rate weight.  Each gradient
is normalized in full parameter space and contracted with the query feature.
Train gradients are never persisted.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
CELEBA_ROOT = SCRIPT_DIR.parent
REFINED_ROOT = CELEBA_ROOT.parent
LEGACY_ROOT = REFINED_ROOT / "legacy_jax"
for path in (SCRIPT_DIR, CELEBA_ROOT, REFINED_ROOT, LEGACY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import numpy as np

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
from run_endpoint_timewise_1x50 import (
    format_duration,
    load_queries,
    progress_bar,
    query_artifact_path,
)
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


EPS = 1e-12
QUERY_NAMESPACE = (
    "celeba_paper_retrac_endpoint_simple_loss_mean50_of_full_l2_1x50"
)
TRANSFORMS = ("raw_gradient", "adamw_full")
SCORE_DIRECTORY = "score_query_train_l2_normalized"
SCORE_VARIANT = "query_train_l2_normalized"


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


def make_event_train_fn(
    *,
    adapter: Any,
    model: Any,
    schedule: Any,
    projector: Any,
    state: Any,
    params: Any,
    cfg: Any,
):
    """Use the working endpoint-pollute batch-one gradient structure.

    The original event sample, diffusion noise, timestep, checkpoint, and
    learning-rate weight are retained.  We intentionally avoid the expensive
    full-batch train-mode/dropout VJP graph that stalls XLA compilation.
    """

    opt_state = state.opt_state

    def one(x0, cond, noise, timestep):
        x0 = x0[None, ...]
        cond = cond[None, ...] if cond.ndim else cond[None]
        noise = noise[None, ...]
        t = timestep[None].astype(jnp.int32)

        def loss_fn(candidate):
            xt = q_sample(schedule, x0, t, noise)
            target = x0 if cfg.predict_x0 else noise
            pred = adapter.eps_apply(model, candidate, xt, t, cond)
            return jnp.mean(jnp.square(pred - target))

        grad = jax.grad(loss_fn)(params)
        update, _ = state.tx.update(grad, opt_state, params)

        def normalized_project(tree):
            norm = tree_l2_norm(tree)
            normalized = jax.tree_util.tree_map(
                lambda leaf: leaf / jnp.maximum(norm, EPS).astype(leaf.dtype), tree
            )
            return projector(normalized).astype(jnp.float32)

        return normalized_project(grad), normalized_project(update)

    return jax.jit(jax.vmap(one, in_axes=(0, 0, 0, 0)))


def load_query_features(
    *,
    queries: list[dict[str, Any]],
    sample_root: Path,
    reference_checkpoint: Path,
    checkpoint_count: int,
    proj_dim: int,
) -> np.ndarray:
    features = np.empty(
        (checkpoint_count, len(queries), proj_dim), dtype=np.float32
    )
    for query_slot, query in enumerate(queries):
        path = query_artifact_path(
            sample_root, reference_checkpoint, query, QUERY_NAMESPACE
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"missing ReTrac query bank for Q{query['id']}: {path}; "
                "run endpoint_timewise_1x50 first"
            )
        with np.load(path, allow_pickle=False) as payload:
            current = np.asarray(payload["query_features"], dtype=np.float32)
            normalization = str(np.asarray(payload["normalization"]).item())
        if current.shape != (checkpoint_count, proj_dim):
            raise ValueError(f"{path}: {current.shape} != {(checkpoint_count, proj_dim)}")
        if normalization != "mean_t(project(full_parameter_l2_normalize(grad_loss_t)))":
            raise ValueError(f"unexpected query normalization in {path}: {normalization}")
        features[:, query_slot] = current
    return features


def score_shard(
    *,
    args: argparse.Namespace,
    cfg: TrajAttributionConfig,
    adapter: Any,
    dataset: Any,
    queries: list[dict[str, Any]],
    query_features: np.ndarray,
    checkpoints: list[Path],
    attribution_indices: np.ndarray,
    component_path: Path,
) -> None:
    device = adapter.choose_device(cfg.prefer_device)
    model = adapter.build_model(cfg)
    state_template = adapter.build_state_template(cfg, model, device)
    schedule = schedule_to_device(
        make_diffusion_schedule(cfg.timesteps, cfg.beta_start, cfg.beta_end), device
    )
    module = importlib.import_module("DM__training_CIFAR5_MULTI_pixel")
    steps_per_epoch = len(dataset) // int(cfg.batch_size)
    total_steps = steps_per_epoch * int(cfg.epochs)
    lr_cfg = SimpleNamespace(**vars(cfg))
    lr_cfg.lr_schedule = cfg.tracin_lr_schedule
    lr_cfg.lr_warmup_ratio = cfg.tracin_warmup_ratio
    lr_schedule = module.make_learning_rate_schedule(lr_cfg, total_steps)
    score_row = np.full((len(dataset),), -1, dtype=np.int64)
    score_row[attribution_indices] = np.arange(len(attribution_indices), dtype=np.int64)
    scores = np.zeros(
        (len(TRANSFORMS), len(queries), len(attribution_indices)), dtype=np.float64
    )
    completed: list[int] = []
    first_kernel = True
    active_path = component_path.with_suffix(".active_checkpoint")
    if active_path.is_file():
        active_path.unlink(missing_ok=True)
        print(
            "[recovery] ignored the interrupted in-memory checkpoint and "
            "kept the last atomic shard accumulator",
            flush=True,
        )
    if component_path.is_file():
        with np.load(component_path, allow_pickle=False) as payload:
            scores[...] = payload["scores"]
            completed = list(
                np.asarray(payload["completed_checkpoints"], dtype=np.int32)
            )
        print(f"[resume] completed checkpoints={completed}", flush=True)

    for checkpoint_index in range(49):
        if checkpoint_index % args.shard_count != args.shard_index:
            continue
        if checkpoint_index in completed:
            print(f"[skip] checkpoint {checkpoint_index + 1}/49", flush=True)
            continue
        active_path.write_text(f"{checkpoint_index}\n")
        state, _ = adapter.restore_state(checkpoints[checkpoint_index], state_template)
        params = tree_to_device(select_state_params(state, "raw"), device)
        projector = build_countsketch_projector_jax(
            params,
            args.proj_dim,
            seed_parts=(args.train_seed, "traj_tracin_projection", checkpoint_index),
            device=device,
        )
        event_train_fn = make_event_train_fn(
            adapter=adapter,
            model=model,
            schedule=schedule,
            projector=projector,
            state=state,
            params=params,
            cfg=cfg,
        )
        query_device = array_to_device(
            jnp.asarray(query_features[checkpoint_index]), device
        )
        start_epoch = 4 * (checkpoint_index + 1)
        rng = state.rng
        completed_events = 0
        checkpoint_started = time.monotonic()
        for event_slot, epoch in enumerate(range(start_epoch + 1, start_epoch + 5)):
            indices = np.arange(len(dataset), dtype=np.int64)
            np.random.default_rng(int(cfg.seed) + epoch).shuffle(indices)
            for batch_no, start in enumerate(
                range(0, steps_per_epoch * int(cfg.batch_size), int(cfg.batch_size))
            ):
                selected = indices[start : start + int(cfg.batch_size)]
                # Preserve the original four-way RNG advance even though the
                # pollute-style batch-one path does not replay training dropout.
                next_rng, noise_rng, t_rng, _dropout_rng = jax.random.split(rng, 4)
                rng = next_rng
                rows = score_row[selected]
                positions = np.flatnonzero(rows >= 0)
                if not len(positions):
                    continue
                x = jax.device_put(
                    module.maybe_to_dtype(
                        jnp.asarray(dataset.images[selected]), cfg.use_bfloat16
                    ),
                    device,
                )
                y = jax.device_put(jnp.asarray(dataset.labels[selected]), device)
                noise = jax.random.normal(noise_rng, x.shape, dtype=x.dtype)
                timesteps = jax.random.randint(
                    t_rng, (int(cfg.batch_size),), 0, int(cfg.timesteps)
                )
                event_step = (epoch - 1) * steps_per_epoch + batch_no
                event_lr = float(
                    np.asarray(jax.device_get(lr_schedule(jnp.asarray(event_step))))
                )
                for offset in range(0, len(positions), args.event_chunk_size):
                    real_positions = positions[offset : offset + args.event_chunk_size]
                    real_count = len(real_positions)
                    padded_positions = np.pad(
                        real_positions,
                        (0, args.event_chunk_size - real_count),
                        mode="edge",
                    )
                    compile_started = time.monotonic()
                    if first_kernel:
                        print(
                            "[compile start] pollute-style batch-one grad + "
                            "AdamW + full-L2 + projection",
                            flush=True,
                        )
                    raw, adamw = event_train_fn(
                        x[padded_positions],
                        y[padded_positions],
                        noise[padded_positions],
                        timesteps[padded_positions],
                    )
                    raw.block_until_ready()
                    if first_kernel:
                        print(
                            "[compile done] train features ready "
                            f"elapsed={format_duration(time.monotonic() - compile_started)}",
                            flush=True,
                        )
                    train = jnp.concatenate(
                        (raw[:real_count], adamw[:real_count]), axis=0
                    )
                    dots = jnp.matmul(
                        query_device,
                        train.T,
                        precision=jax.lax.Precision.HIGHEST,
                    )
                    dots.block_until_ready()
                    if first_kernel:
                        print("[dot ready] query contraction complete", flush=True)
                        first_kernel = False
                        if args.compile_only:
                            active_path.unlink(missing_ok=True)
                            print("[compile only] warm-up complete", flush=True)
                            return
                    host = np.asarray(jax.device_get(dots), dtype=np.float32).reshape(
                        len(queries), len(TRANSFORMS), real_count
                    )
                    destination = rows[real_positions]
                    scores[0, :, destination] += event_lr * host[:, 0]
                    scores[1, :, destination] += event_lr * host[:, 1]
                    completed_events += real_count
                if (
                    batch_no == 0
                    or batch_no + 1 == steps_per_epoch
                    or (batch_no + 1) % args.progress_every_batches == 0
                ):
                    elapsed = time.monotonic() - checkpoint_started
                    expected_events = 4 * len(attribution_indices)
                    eta = (
                        elapsed / completed_events * (expected_events - completed_events)
                        if completed_events
                        else 0.0
                    )
                    print(
                        f"[progress] checkpoint={checkpoint_index + 1}/49 "
                        f"event={event_slot + 1}/4 epoch={epoch} "
                        f"batch={batch_no + 1}/{steps_per_epoch} "
                        f"attribution_events={completed_events}/{expected_events} "
                        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
                        flush=True,
                    )
            rng, _, _ = jax.random.split(rng, 3)

        completed.append(checkpoint_index)
        save_npz_compressed_atomic(
            str(component_path),
            scores=scores,
            completed_checkpoints=np.asarray(completed, dtype=np.int32),
            score_indices=attribution_indices,
            query_ids=np.asarray([query["id"] for query in queries], dtype=np.int32),
        )
        active_path.unlink(missing_ok=True)
        print(f"[checkpoint] {checkpoint_index + 1}/49 complete", flush=True)


def merge(
    *,
    args: argparse.Namespace,
    queries: list[dict[str, Any]],
    attribution_indices: np.ndarray,
) -> None:
    shards = []
    for shard in range(args.shard_count):
        path = Path(f"{args.component_prefix}.shard_{shard:02d}.npz")
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            shards.append(np.asarray(payload["scores"], dtype=np.float64))
    scores = sum(shards, np.zeros_like(shards[0]))
    result_root = CELEBA_ROOT / "result" / args.experiment
    for query_slot, query in enumerate(queries):
        for transform_slot, transform in enumerate(TRANSFORMS):
            namespace = (
                f"celeba_paper_retrac_{transform}_train_replay4_batch1_"
                "query_mean50_of_full_l2_linear_shape3d_semantics"
            )
            root = (
                result_root
                / "attribution_score"
                / "prompted_solo"
                / f"train_seed_{args.train_seed}"
                / f"query_{_prompt_path_tag(query['prompt'])}"
                / f"initial_seed_{int(query['seed'])}"
                / f"traj_tracin_{namespace}"
                / SCORE_DIRECTORY
            )
            _write_score_outputs(
                root,
                scores[transform_slot, query_slot],
                attribution_indices,
                train_dir=args.component_prefix.parent,
                query_dir=result_root / "sample_ddim_eta0_50",
                algorithm="paper_retrac",
                extra_manifest={
                    "score_variant": SCORE_VARIANT,
                    "score_contraction": "linear",
                    "train_feature": transform,
                    "query_objective": (
                        "mean_t(project(full_parameter_l2_normalize(query_grad_t)))"
                    ),
                    "train_normalization": (
                        "full_parameter_l2_normalize(each_train_loss_gradient)"
                    ),
                    "train_event_count": 4,
                    "train_replay_semantics": (
                        "pollute_style_independent_batch1_with_replayed_noise_and_timestep"
                    ),
                    "training_dropout_replayed": False,
                    "query_timestamp_count": 50,
                    "query_mc_per_timestamp": 1,
                    "checkpoint_count": 49,
                    "persistent_train_gradient_artifact": False,
                    "prediction_sign_for_loss_utility": -1,
                },
            )
        print(f"[merge] Q{query['id']} complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", default=",".join(map(str, range(72))))
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--proj-dim", type=int, default=4096)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--event-chunk-size", type=int, default=4)
    parser.add_argument("--progress-every-batches", type=int, default=25)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--component-prefix", type=Path, required=True)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")
    if args.merge_only and args.compile_only:
        raise ValueError("--merge-only and --compile-only are mutually exclusive")
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
    configured_devices = getattr(cfg, "num_devices", None)
    requested_devices = int(configured_devices) if configured_devices is not None else 1
    use_data_parallel = bool(getattr(cfg, "use_data_parallel", False))
    if use_data_parallel and requested_devices > 1:
        raise RuntimeError(
            "exact event replay currently requires checkpoints trained with a "
            "single-device RNG topology; checkpoint config reports "
            f"use_data_parallel={use_data_parallel}, "
            f"num_devices={configured_devices}"
        )
    adapter = get_adapter(cfg)
    dataset = adapter.iter_dataset(cfg)
    query_ids = parse_ints(args.query_ids)
    queries = load_queries(
        args.query_file,
        query_ids,
        sample_root,
        reference_checkpoint,
        adapter,
        dataset,
        cfg,
    )
    attribution_indices = np.asarray(
        np.load(ATTRIBUTION_INDICES_PATH), dtype=np.int64
    ).reshape(-1)
    if len(attribution_indices) != 5000:
        raise ValueError(f"expected 5000 attribution points, got {len(attribution_indices)}")
    if args.merge_only:
        merge(
            args=args,
            queries=queries,
            attribution_indices=attribution_indices,
        )
        return
    query_features = load_query_features(
        queries=queries,
        sample_root=sample_root,
        reference_checkpoint=reference_checkpoint,
        checkpoint_count=49,
        proj_dim=args.proj_dim,
    )
    component_path = Path(
        f"{args.component_prefix}.shard_{args.shard_index:02d}.npz"
    )
    component_path.parent.mkdir(parents=True, exist_ok=True)
    score_shard(
        args=args,
        cfg=cfg,
        adapter=adapter,
        dataset=dataset,
        queries=queries,
        query_features=query_features,
        checkpoints=checkpoints,
        attribution_indices=attribution_indices,
        component_path=component_path,
    )


if __name__ == "__main__":
    main()
