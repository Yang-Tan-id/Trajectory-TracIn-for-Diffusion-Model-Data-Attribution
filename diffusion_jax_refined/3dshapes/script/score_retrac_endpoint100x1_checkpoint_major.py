#!/usr/bin/env python3
"""Stream exact-event ReTrac and endpoint-TracIn 100x1 scores.

The query at checkpoint c is the projected gradient of the mean denoising
loss of the saved query endpoint over 100 explicit timesteps and one noise
draw per timestep.  ReTrac contracts it with the four exact stochastic
training events between checkpoints c and c+1.  Endpoint-TracIn contracts it
with the checkpoint-level 100-timestep/MC1 mean-loss train gradient.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

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
from direction20_query_common import load_queries
from dtrak.algorithm import build_countsketch_projector_jax
from traj_tracin.algorithm import (
    TrajAttributionConfig,
    apply_checkpoint_config,
    array_to_device,
    get_adapter,
    list_checkpoints_sorted,
    make_diffusion_schedule,
    q_sample,
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
EPS = 1e-8


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def output_root(result_root: Path, train_seed: int, query: dict[str, Any], namespace: str) -> Path:
    return (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
        / f"query_{_prompt_tag(query['prompt'])}"
        / f"initial_seed_{query['seed']}"
        / f"traj_tracin_{namespace}"
    )


def outputs_exist(root: Path) -> bool:
    return all((root / directory / "scores.npy").is_file() for directory, _ in VARIANTS)


def endpoint_noise_key(train_seed: int, query_id: int, checkpoint: int, timestep: int):
    key = jax.random.PRNGKey(int(train_seed) + 487_301)
    for value in (query_id, checkpoint, timestep):
        key = jax.random.fold_in(key, int(value))
    return key


def make_endpoint_query_fn(adapter, model, schedule, projector):
    def one(params, endpoint, cond, noise, timesteps):
        # endpoint/noise: [1,H,W,C] and [T,1,H,W,C].  Keeping the sample
        # dimension explicit makes this also work for non-image adapters.
        count = timesteps.shape[0]
        x0 = jnp.broadcast_to(endpoint[None, ...], (count,) + endpoint.shape)
        x0 = x0.reshape((count * endpoint.shape[0],) + endpoint.shape[1:])
        noise_flat = noise.reshape(x0.shape)
        t_flat = jnp.repeat(timesteps.astype(jnp.int32), endpoint.shape[0])
        cond_rep = jnp.broadcast_to(cond[None, ...], (count,) + cond.shape)
        cond_flat = cond_rep.reshape((count * cond.shape[0],) + cond.shape[1:])

        def loss_fn(candidate):
            xt = q_sample(schedule, x0, t_flat, noise_flat)
            pred = adapter.eps_apply(model, candidate, xt, t_flat, cond_flat)
            return jnp.mean(jnp.square(pred - noise_flat))

        grad = jax.grad(loss_fn)(params)
        return projector(grad).astype(jnp.float32)

    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0, None)))


@jax.jit
def endpoint_contractions(train: jax.Array, query: jax.Array) -> jax.Array:
    # train [K,P], query [Q,P] -> [Q,4,K]
    raw = jnp.einsum("kp,qp->qk", train, query, precision=jax.lax.Precision.HIGHEST)
    qnorm = jnp.maximum(jnp.linalg.norm(query, axis=-1), EPS)[:, None]
    tnorm = jnp.maximum(jnp.linalg.norm(train, axis=-1), EPS)[None, :]
    return jnp.stack((raw, raw / qnorm, raw / tnorm, raw / (qnorm * tnorm)), axis=1)


@jax.jit
def retrac_contractions(events: jax.Array, query: jax.Array) -> jax.Array:
    # events [E,K,P], query [Q,P] -> [Q,4,E,K]
    raw = jnp.einsum("ekp,qp->qek", events, query, precision=jax.lax.Precision.HIGHEST)
    qnorm = jnp.maximum(jnp.linalg.norm(query, axis=-1), EPS)[:, None, None]
    tnorm = jnp.maximum(jnp.linalg.norm(events, axis=-1), EPS)[None, :, :]
    return jnp.stack((raw, raw / qnorm, raw / tnorm, raw / (qnorm * tnorm)), axis=1)


def load_endpoint_train_part(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        features = np.asarray(payload["train_features"], dtype=np.float32)
        indices = np.asarray(payload["score_indices"], dtype=np.int64)
        timesteps = np.asarray(payload["train_timesteps_used"], dtype=np.int32)
    if features.shape != (1, len(indices), 4096):
        raise ValueError(f"{path}: expected (1,K,4096), got {features.shape}")
    if timesteps.shape != (100,):
        raise ValueError(f"{path}: expected 100 train timesteps, got {timesteps.shape}")
    return features[0], indices, timesteps


def load_event_shard(path: Path, learning_rate_schedule, steps_per_epoch: int):
    with np.load(path, allow_pickle=False) as payload:
        features = np.asarray(payload["train_features"], dtype=np.float32)
        indices = np.asarray(payload["dataset_indices"], dtype=np.int64)
        batches = np.asarray(payload["batch_indices"], dtype=np.int64)
        epoch = int(np.asarray(payload["epoch"]).item())
        definition = str(np.asarray(payload["event_feature"]).item())
    if definition != "raw_gradient":
        raise ValueError(f"{path}: expected raw_gradient, got {definition}")
    steps = (epoch - 1) * steps_per_epoch + batches
    lrs = np.asarray(jax.device_get(learning_rate_schedule(jnp.asarray(steps))), dtype=np.float64)
    return features, indices, lrs


def load_retrac_events(
    root: Path,
    checkpoint: int,
    expected_indices: np.ndarray,
    learning_rate_schedule,
    steps_per_epoch: int,
    shards: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    start_epoch = 4 * (checkpoint + 1)
    interval = root / f"epoch_{start_epoch}_{start_epoch + 4}"
    event_features = []
    event_lrs = []
    for epoch in range(start_epoch + 1, start_epoch + 5):
        feature_parts = []
        index_parts = []
        lr_parts = []
        for shard in range(shards):
            path = interval / (
                f"event_gradient_epoch_{epoch:04d}_shard_{shard:02d}_of_{shards:02d}.npz"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            features, indices, lrs = load_event_shard(
                path, learning_rate_schedule, steps_per_epoch
            )
            feature_parts.append(features)
            index_parts.append(indices)
            lr_parts.append(lrs)
        features = np.concatenate(feature_parts, axis=0)
        indices = np.concatenate(index_parts)
        lrs = np.concatenate(lr_parts)
        lookup = {int(index): row for row, index in enumerate(indices)}
        try:
            order = np.asarray([lookup[int(index)] for index in expected_indices], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"{interval}: ReTrac indices do not cover endpoint train bank") from exc
        event_features.append(features[order])
        event_lrs.append(lrs[order])
    return np.stack(event_features), np.stack(event_lrs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", required=True)
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--endpoint-train-artifact", type=Path, required=True)
    parser.add_argument("--retrac-event-root", type=Path, required=True)
    parser.add_argument("--retrac-namespace", default="retrac_exact4_endpoint100x1_q0_99")
    parser.add_argument("--endpoint-namespace", default="endpoint_tracin_train100x1_query100x1_q0_99")
    parser.add_argument("--query-batch-size", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    all_query_ids = parse_ints(args.query_ids)
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
    values = attribution_config("traj_tracin")
    result_root = SHAPES_ROOT / "result" / args.experiment
    checkpoint_root = result_root / "model" / "prompted_jax"
    values.update(
        checkpoint_dir=str(checkpoint_root),
        reference_ckpt=str(checkpoint_root / f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt"),
        seed=args.train_seed,
        epochs=args.epochs,
        parameter_source="raw",
        num_traj_snapshots=100,
        proj_dim=4096,
        out_dir=str(result_root / "tmp"),
    )
    cfg = TrajAttributionConfig(**values)
    checkpoints = list_checkpoints_sorted(cfg.checkpoint_dir)
    if len(checkpoints) != 50:
        raise ValueError(f"expected 50 checkpoints, found {len(checkpoints)}")
    reference_checkpoint = Path(cfg.reference_ckpt or checkpoints[-1])
    apply_checkpoint_config(cfg, str(reference_checkpoint))
    adapter = get_adapter(cfg)
    device = adapter.choose_device(cfg.prefer_device)
    dataset = adapter.iter_dataset(cfg)
    model = adapter.build_model(cfg)
    state_template = adapter.build_state_template(cfg, model, device)
    schedule = schedule_to_device(
        make_diffusion_schedule(cfg.timesteps, cfg.beta_start, cfg.beta_end), device
    )
    sample_root = result_root / "sample_ddim_eta0_1000"
    queries = load_queries(
        args.query_file, query_ids, sample_root, reference_checkpoint,
        adapter, dataset, cfg,
    )
    for query in queries:
        query["retrac_root"] = output_root(
            result_root, args.train_seed, query, args.retrac_namespace
        )
        query["endpoint_root"] = output_root(
            result_root, args.train_seed, query, args.endpoint_namespace
        )
    queries = [
        query for query in queries
        if not (
            outputs_exist(query["retrac_root"])
            and outputs_exist(query["endpoint_root"])
        )
    ]
    if not queries:
        print("[done] every query in this shard already has both methods and four variants")
        return

    module = importlib.import_module("DM__training_CIFAR5_MULTI_pixel")
    steps_per_epoch = len(dataset) // int(cfg.batch_size)
    total_steps = steps_per_epoch * int(cfg.epochs)
    # TrajAttributionConfig stores the synchronized training schedule under
    # tracin_* names.  The original trainer expects lr_* names.
    lr_cfg = SimpleNamespace(**vars(cfg))
    lr_cfg.lr_schedule = cfg.tracin_lr_schedule
    lr_cfg.lr_warmup_ratio = cfg.tracin_warmup_ratio
    lr_schedule = module.make_learning_rate_schedule(lr_cfg, total_steps)
    endpoint_parts = Path(str(args.endpoint_train_artifact) + ".parts")
    endpoint_scores = np.zeros((len(queries), len(VARIANTS), len(dataset)), dtype=np.float64)
    retrac_scores = np.zeros_like(endpoint_scores)
    score_indices = None

    print(
        f"[stream] shard={args.shard_index}/{args.shard_count} queries={len(queries)} "
        "checkpoint pairs=49; query=mean(100 timestamps x MC1); query artifacts=0",
        flush=True,
    )
    for checkpoint in range(49):
        train, indices, timesteps = load_endpoint_train_part(
            endpoint_parts / f"ckpt_{checkpoint:04d}.npz"
        )
        if score_indices is None:
            score_indices = indices
            endpoint_scores = np.zeros(
                (len(queries), len(VARIANTS), len(indices)), dtype=np.float64
            )
            retrac_scores = np.zeros_like(endpoint_scores)
        elif not np.array_equal(score_indices, indices):
            raise ValueError(f"checkpoint {checkpoint}: endpoint train indices changed")
        events, event_lrs = load_retrac_events(
            args.retrac_event_root,
            checkpoint,
            indices,
            lr_schedule,
            steps_per_epoch,
        )
        state, _ = adapter.restore_state(checkpoints[checkpoint], state_template)
        params = tree_to_device(select_state_params(state, "raw"), device)
        projector = build_countsketch_projector_jax(
            params,
            4096,
            seed_parts=(args.train_seed, "traj_tracin_projection", checkpoint),
            device=device,
        )
        query_fn = make_endpoint_query_fn(adapter, model, schedule, projector)
        train_device = array_to_device(jnp.asarray(train), device)
        events_device = array_to_device(jnp.asarray(events), device)
        checkpoint_lr = float(
            module.learning_rate_at_step(
                cfg, 4 * (checkpoint + 1) * steps_per_epoch, total_steps
            )
        )

        for start in range(0, len(queries), args.query_batch_size):
            real = queries[start : start + args.query_batch_size]
            padded = list(real)
            while len(padded) < args.query_batch_size:
                padded.append(padded[-1])
            endpoints = np.stack([query["endpoint"] for query in padded])
            conds = np.stack([query["cond"] for query in padded])
            noises = np.stack(
                [
                    np.asarray(
                        jax.device_get(
                            # One vectorized draw produces the 100 independent
                            # MC1 noises; do not dispatch 100 tiny RNG kernels.
                            jax.random.normal(
                                endpoint_noise_key(
                                    args.train_seed,
                                    int(query["id"]),
                                    checkpoint,
                                    100_001,
                                ),
                                (len(timesteps),) + query["endpoint"].shape,
                                dtype=jnp.float32,
                            )
                        ),
                        dtype=np.float32,
                    )
                    for query in padded
                ],
                axis=0,
            )
            query_features = query_fn(
                params,
                array_to_device(jnp.asarray(endpoints), device),
                array_to_device(jnp.asarray(conds), device),
                array_to_device(jnp.asarray(noises), device),
                array_to_device(jnp.asarray(timesteps), device),
            )
            endpoint_values = endpoint_contractions(train_device, query_features)
            retrac_values = retrac_contractions(events_device, query_features)
            endpoint_values.block_until_ready()
            retrac_values.block_until_ready()
            count = len(real)
            endpoint_scores[start : start + count] += (
                checkpoint_lr
                * np.asarray(endpoint_values[:count], dtype=np.float64)
            )
            retrac_scores[start : start + count] += np.sum(
                np.asarray(retrac_values[:count], dtype=np.float64)
                * event_lrs[None, None, :, :],
                axis=2,
            )

        print(f"[checkpoint] {checkpoint + 1}/49 complete", flush=True)

    assert score_indices is not None
    for query_index, query in enumerate(queries):
        for variant_index, (directory, variant) in enumerate(VARIANTS):
            for method, root_key, scores in (
                ("retrac_exact_training_events", "retrac_root", retrac_scores),
                ("endpoint_tracin", "endpoint_root", endpoint_scores),
            ):
                output = query[root_key] / directory
                if (output / "scores.npy").is_file():
                    continue
                _write_score_outputs(
                    output,
                    scores[query_index, variant_index],
                    score_indices,
                    train_dir=(args.retrac_event_root if root_key == "retrac_root" else endpoint_parts),
                    query_dir=sample_root,
                    algorithm=method,
                    extra_manifest={
                        "score_variant": variant,
                        "query_objective": "endpoint_denoising_mean_loss_100_timestamps_mc1",
                        "query_gradient_artifact_written": False,
                        "checkpoint_count": 49,
                        "prediction_sign_for_loss_utility": -1,
                        "train_definition": (
                            "four exact training events with saved t/noise/dropout and event LR"
                            if root_key == "retrac_root"
                            else "mean denoising loss over 100 timestamps x MC1 and checkpoint LR"
                        ),
                    },
                )
        print(f"[score] Q{query['id']} complete", flush=True)


if __name__ == "__main__":
    main()
