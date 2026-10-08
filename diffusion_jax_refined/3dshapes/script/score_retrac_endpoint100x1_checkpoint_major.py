#!/usr/bin/env python3
"""Stream exact-event ReTrac and endpoint-TracIn scores.

The query at checkpoint c is the projected gradient of the mean denoising
loss of the saved query endpoint over explicit timesteps and one noise
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
    select_snapshot_positions,
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


def outputs_exist(root: Path, paper_retrac: bool = False) -> bool:
    variants = VARIANTS[-1:] if paper_retrac else VARIANTS
    return all((root / directory / "scores.npy").is_file() for directory, _ in variants)


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


def make_paper_retrac_endpoint_query_fn(adapter, model, schedule, projector):
    """Normalize each full-space timestep gradient, then average its sketch."""
    def one(params, endpoint, cond, noises, timesteps):
        def loss_one(candidate, noise, timestep):
            t = jnp.full((endpoint.shape[0],), timestep, dtype=jnp.int32)
            xt = q_sample(schedule, endpoint, t, noise)
            pred = adapter.eps_apply(model, candidate, xt, t, cond)
            return jnp.mean(jnp.square(pred - noise))

        def body(total, inputs):
            noise, timestep = inputs
            grad = jax.grad(loss_one)(params, noise, timestep)
            norm_sq = sum(
                jnp.vdot(leaf.astype(jnp.float32), leaf.astype(jnp.float32)).real
                for leaf in jax.tree_util.tree_leaves(grad)
            )
            denominator = jnp.sqrt(jnp.maximum(norm_sq, 1e-16))
            normalized = jax.tree_util.tree_map(
                lambda leaf: leaf / denominator.astype(leaf.dtype), grad
            )
            return total + projector(normalized).astype(jnp.float32), None

        initial = jnp.zeros((4096,), dtype=jnp.float32)
        total, _ = jax.lax.scan(body, initial, (noises, timesteps))
        return total / jnp.asarray(timesteps.shape[0], dtype=jnp.float32)

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


def load_adamw_aligned10x10_train_part(
    path: Path, *, add_optimizer_history: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse cached AdamW 10x10 terms to one stored-LR feature per checkpoint."""
    with np.load(path, allow_pickle=False) as payload:
        features = np.asarray(payload["train_features"], dtype=np.float32)
        indices = np.asarray(payload["score_indices"], dtype=np.int64)
        weights = np.asarray(payload["term_weights"], dtype=np.float64).reshape(-1)
        ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32).reshape(-1)
        if features.ndim != 3 or features.shape[1:] != (len(indices), 4096):
            raise ValueError(f"{path}: unexpected AdamW feature shape {features.shape}")
        if weights.shape[0] != features.shape[0] or ckpts.shape[0] != features.shape[0]:
            raise ValueError(f"{path}: AdamW term metadata does not match {features.shape[0]} terms")
        if add_optimizer_history:
            if "optimizer_history_features" not in payload:
                raise ValueError(f"{path}: missing optimizer_history_features for AdamW full")
            history = np.asarray(payload["optimizer_history_features"], dtype=np.float32)
            if history.shape != (features.shape[0], features.shape[2]):
                raise ValueError(f"{path}: optimizer history shape {history.shape} is incompatible")
            features = features + history[:, None, :]
    # Existing term_weights carry the checkpoint LR and timestamp averaging.
    collapsed = np.einsum("t,tkp->kp", weights, features, optimize=True)
    return collapsed.astype(np.float32), indices


def load_event_shard(
    path: Path, learning_rate_schedule, steps_per_epoch: int, *,
    paper_retrac: bool = False, paper_train_transform: str = "raw",
):
    with np.load(path, allow_pickle=False) as payload:
        feature_key = (
            "train_features_adamw_full"
            if paper_retrac and paper_train_transform == "adamw_full"
            else "train_features"
        )
        features = np.asarray(payload[feature_key], dtype=np.float32)
        indices = np.asarray(payload["dataset_indices"], dtype=np.int64)
        batches = np.asarray(payload["batch_indices"], dtype=np.int64)
        timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        epoch = int(np.asarray(payload["epoch"]).item())
        definition = str(np.asarray(payload["event_feature"]).item())
    if paper_retrac:
        allowed = {
            "raw_gradient_full_l2_normalized",
            "raw_and_adamw_full_l2_normalized",
        }
        if paper_train_transform == "adamw_full":
            allowed = {"raw_and_adamw_full_l2_normalized"}
    else:
        allowed = {"raw_gradient"}
    if definition not in allowed:
        raise ValueError(f"{path}: expected one of {sorted(allowed)}, got {definition}")
    present = batches >= 0
    steps = (epoch - 1) * steps_per_epoch + np.maximum(batches, 0)
    lrs = np.asarray(jax.device_get(learning_rate_schedule(jnp.asarray(steps))), dtype=np.float64)
    lrs[~present] = 0.0
    if timesteps.shape != indices.shape:
        raise ValueError(
            f"{path}: timestep shape {timesteps.shape} does not match indices {indices.shape}"
        )
    return features, indices, lrs, timesteps


def load_retrac_events(
    root: Path,
    checkpoint: int,
    expected_indices: np.ndarray | None,
    learning_rate_schedule,
    steps_per_epoch: int,
    shards: int = 2,
    paper_retrac: bool = False,
    paper_train_transform: str = "raw",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start_epoch = 4 * (checkpoint + 1)
    interval = root / f"epoch_{start_epoch}_{start_epoch + 4}"
    event_features = []
    event_lrs = []
    event_timesteps = []
    for epoch in range(start_epoch + 1, start_epoch + 5):
        feature_parts = []
        index_parts = []
        lr_parts = []
        timestep_parts = []
        for shard in range(shards):
            path = interval / (
                f"event_gradient_epoch_{epoch:04d}_shard_{shard:02d}_of_{shards:02d}.npz"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            features, indices, lrs, timesteps = load_event_shard(
                path, learning_rate_schedule, steps_per_epoch,
                paper_retrac=paper_retrac,
                paper_train_transform=paper_train_transform,
            )
            feature_parts.append(features)
            index_parts.append(indices)
            lr_parts.append(lrs)
            timestep_parts.append(timesteps)
        features = np.concatenate(feature_parts, axis=0)
        indices = np.concatenate(index_parts)
        lrs = np.concatenate(lr_parts)
        timesteps = np.concatenate(timestep_parts)
        if expected_indices is None:
            expected_indices = np.sort(np.unique(indices))
            if len(expected_indices) != len(indices):
                raise ValueError(f"{interval}: duplicate ReTrac dataset indices")
        lookup = {int(index): row for row, index in enumerate(indices)}
        try:
            order = np.asarray([lookup[int(index)] for index in expected_indices], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"{interval}: ReTrac indices do not cover the score index bank") from exc
        event_features.append(features[order])
        event_lrs.append(lrs[order])
        event_timesteps.append(timesteps[order])
    assert expected_indices is not None
    return (
        np.stack(event_features),
        np.stack(event_lrs),
        np.stack(event_timesteps),
        expected_indices,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", required=True)
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--endpoint-train-artifact", type=Path)
    parser.add_argument(
        "--endpoint-adamw-aligned10x10", action="store_true",
        help="Use cached AdamW aligned10x10 train terms, collapsed with stored term weights.",
    )
    parser.add_argument("--endpoint-adamw-full", action="store_true")
    parser.add_argument("--retrac-event-root", type=Path, required=True)
    parser.add_argument("--methods", choices=("retrac", "endpoint", "both"), default="both")
    parser.add_argument(
        "--paper-retrac", action="store_true",
        help="Use full-space per-timestep/per-event L2 normalization before projection.",
    )
    parser.add_argument(
        "--paper-train-transform", choices=("raw", "adamw_full"), default="raw",
        help="Select the pre-projection normalized train feature in paper-ReTrac mode.",
    )
    parser.add_argument("--retrac-namespace", default="retrac_exact4_endpoint100x1_q0_99")
    parser.add_argument("--query-timestamp-count", type=int, default=100)
    parser.add_argument(
        "--retrac-reduction",
        choices=("linear", "timestamp_sum_squared"),
        default="linear",
        help=(
            "Reduce event terms linearly, or first sum terms sharing the exact "
            "replayed training timestep across checkpoints/events and then square."
        ),
    )
    parser.add_argument("--endpoint-namespace", default="endpoint_tracin_train100x1_query100x1_q0_99")
    parser.add_argument("--query-batch-size", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.query_timestamp_count <= 0:
        parser.error("--query-timestamp-count must be positive")
    run_retrac = args.methods in ("retrac", "both")
    run_endpoint = args.methods in ("endpoint", "both")
    if run_endpoint and args.endpoint_train_artifact is None:
        parser.error("--endpoint-train-artifact is required for --methods endpoint/both")

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
        num_traj_snapshots=args.query_timestamp_count,
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
    queries = [query for query in queries if not (
        (not run_retrac or outputs_exist(query["retrac_root"], args.paper_retrac))
        and (not run_endpoint or outputs_exist(query["endpoint_root"]))
    )]
    if not queries:
        print(f"[done] every query in this shard already has {args.methods} and four variants")
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
    endpoint_parts = (
        Path(str(args.endpoint_train_artifact) + ".parts") if run_endpoint else None
    )
    endpoint_scores = None
    retrac_scores = None
    retrac_timestamp_groups = None
    score_indices = None
    ddim_ts = np.linspace(
        int(cfg.timesteps) - 1, 0, int(cfg.ddim_steps), dtype=np.int32
    )
    query_positions = select_snapshot_positions(
        int(cfg.ddim_steps), args.query_timestamp_count, cfg.traj_snapshot_positions
    )
    query_timesteps = np.asarray(
        [int(ddim_ts[int(position)]) for position in query_positions], dtype=np.int32
    )
    # Draw the historical 100-timestep noise bank and select matching entries.
    # This keeps a 10-timestep run directly comparable with the existing 100t
    # run: shared timesteps use exactly the same MC1 noise.
    noise_reference_positions = select_snapshot_positions(
        int(cfg.ddim_steps), 100, None
    )
    noise_reference_timesteps = np.asarray(
        [int(ddim_ts[int(position)]) for position in noise_reference_positions],
        dtype=np.int32,
    )
    noise_reference_lookup = {
        int(timestep): index
        for index, timestep in enumerate(noise_reference_timesteps)
    }
    try:
        query_noise_indices = np.asarray(
            [noise_reference_lookup[int(timestep)] for timestep in query_timesteps],
            dtype=np.int32,
        )
    except KeyError as error:
        raise ValueError(
            "query timestep schedule must be a subset of the 100-timestep "
            f"reference schedule; missing timestep={error.args[0]}"
        ) from error

    print(
        f"[stream] shard={args.shard_index}/{args.shard_count} queries={len(queries)} "
        f"checkpoint pairs=49; query=mean({args.query_timestamp_count} timestamps x MC1); "
        "query artifacts=0",
        flush=True,
    )
    for checkpoint in range(49):
        train = None
        events = None
        event_lrs = None
        event_timesteps = None
        timesteps = query_timesteps
        if run_endpoint:
            assert endpoint_parts is not None
            part_path = endpoint_parts / f"ckpt_{checkpoint:04d}.npz"
            if args.endpoint_adamw_aligned10x10:
                train, indices = load_adamw_aligned10x10_train_part(
                    part_path, add_optimizer_history=args.endpoint_adamw_full
                )
            else:
                train, indices, stored_timesteps = load_endpoint_train_part(part_path)
                if not np.array_equal(stored_timesteps, query_timesteps):
                    raise ValueError(
                        f"checkpoint {checkpoint}: endpoint/query timestep schedule mismatch"
                    )
            if score_indices is None:
                score_indices = indices
            elif not np.array_equal(score_indices, indices):
                raise ValueError(f"checkpoint {checkpoint}: endpoint train indices changed")
        if run_retrac:
            events, event_lrs, event_timesteps, retrac_indices = load_retrac_events(
                args.retrac_event_root,
                checkpoint,
                score_indices,
                lr_schedule,
                steps_per_epoch,
                paper_retrac=args.paper_retrac,
                paper_train_transform=args.paper_train_transform,
            )
            if score_indices is None:
                score_indices = retrac_indices
            elif not np.array_equal(score_indices, retrac_indices):
                raise ValueError(f"checkpoint {checkpoint}: ReTrac indices changed")
        if endpoint_scores is None and run_endpoint:
            endpoint_scores = np.zeros(
                (len(queries), len(VARIANTS), len(score_indices)), dtype=np.float64
            )
        if retrac_scores is None and run_retrac:
            retrac_scores = np.zeros(
                (len(queries), len(VARIANTS), len(score_indices)), dtype=np.float64
            )
            if args.retrac_reduction == "timestamp_sum_squared":
                retrac_timestamp_groups = np.zeros(
                    (
                        len(queries), len(VARIANTS), int(cfg.timesteps),
                        len(score_indices),
                    ),
                    dtype=np.float32,
                )
        state, _ = adapter.restore_state(checkpoints[checkpoint], state_template)
        params = tree_to_device(select_state_params(state, "raw"), device)
        projector = build_countsketch_projector_jax(
            params,
            4096,
            seed_parts=(args.train_seed, "traj_tracin_projection", checkpoint),
            device=device,
        )
        query_fn = (
            make_paper_retrac_endpoint_query_fn(adapter, model, schedule, projector)
            if args.paper_retrac
            else make_endpoint_query_fn(adapter, model, schedule, projector)
        )
        train_device = (
            array_to_device(jnp.asarray(train), device) if run_endpoint else None
        )
        events_device = (
            array_to_device(jnp.asarray(events), device) if run_retrac else None
        )
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
                            # Reproduce the existing 100t noise bank, then use
                            # the entries belonging to this timestamp subset.
                            jax.random.normal(
                                endpoint_noise_key(
                                    args.train_seed,
                                    int(query["id"]),
                                    checkpoint,
                                    100_001,
                                ),
                                (100,) + query["endpoint"].shape,
                                dtype=jnp.float32,
                            )[query_noise_indices]
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
            count = len(real)
            if run_endpoint:
                assert train_device is not None and endpoint_scores is not None
                endpoint_values = endpoint_contractions(train_device, query_features)
                endpoint_values.block_until_ready()
                endpoint_scores[start : start + count] += (
                    (1.0 if args.endpoint_adamw_aligned10x10 else checkpoint_lr)
                    * np.asarray(endpoint_values[:count], dtype=np.float64)
                )
            if run_retrac:
                assert events_device is not None and event_lrs is not None
                assert retrac_scores is not None
                if args.paper_retrac:
                    raw = jnp.einsum(
                        "ekp,qp->qek", events_device, query_features,
                        precision=jax.lax.Precision.HIGHEST,
                    )
                    retrac_values = raw[:, None, :, :]
                else:
                    retrac_values = retrac_contractions(events_device, query_features)
                retrac_values.block_until_ready()
                weighted_events = (
                    np.asarray(retrac_values[:count], dtype=np.float32)
                    * event_lrs[None, None, :, :].astype(np.float32)
                )
                if args.retrac_reduction == "linear":
                    retrac_scores[start : start + count] += np.sum(
                        weighted_events, axis=2, dtype=np.float64
                    )
                else:
                    assert retrac_timestamp_groups is not None
                    assert event_timesteps is not None
                    point_indices = np.arange(len(score_indices), dtype=np.int64)
                    for query_offset in range(count):
                        for variant_index in range(len(VARIANTS)):
                            grouped = retrac_timestamp_groups[
                                start + query_offset, variant_index
                            ]
                            for event_index in range(event_timesteps.shape[0]):
                                np.add.at(
                                    grouped,
                                    (event_timesteps[event_index], point_indices),
                                    weighted_events[
                                        query_offset, variant_index, event_index
                                    ],
                                )

        print(f"[checkpoint] {checkpoint + 1}/49 complete", flush=True)

    if run_retrac and args.retrac_reduction == "timestamp_sum_squared":
        assert retrac_scores is not None and retrac_timestamp_groups is not None
        retrac_scores.fill(0.0)
        for timestep in range(retrac_timestamp_groups.shape[2]):
            values = retrac_timestamp_groups[:, :, timestep, :].astype(np.float64)
            retrac_scores += np.square(values)
        del retrac_timestamp_groups

    assert score_indices is not None
    output_variants = VARIANTS[-1:] if args.paper_retrac else VARIANTS
    for query_index, query in enumerate(queries):
        for output_index, (directory, variant) in enumerate(output_variants):
            variant_index = 0 if args.paper_retrac else output_index
            methods = []
            if run_retrac:
                methods.append(("retrac_exact_training_events", "retrac_root", retrac_scores))
            if run_endpoint:
                methods.append(("endpoint_tracin", "endpoint_root", endpoint_scores))
            for method, root_key, scores in methods:
                assert scores is not None
                output = query[root_key] / directory
                if (output / "scores.npy").is_file():
                    continue
                _write_score_outputs(
                    output,
                    scores[query_index, variant_index],
                    score_indices,
                    train_dir=(
                        args.retrac_event_root
                        if root_key == "retrac_root"
                        else endpoint_parts
                    ),
                    query_dir=sample_root,
                    algorithm=method,
                    extra_manifest={
                        "score_variant": variant,
                        "query_objective": (
                            "mean_of_full_l2_normalized_endpoint_denoising_gradients_"
                            f"{args.query_timestamp_count}_timestamps_mc1"
                            if args.paper_retrac
                            else "endpoint_denoising_mean_loss_"
                            f"{args.query_timestamp_count}_timestamps_mc1"
                        ),
                        "query_gradient_artifact_written": False,
                        "checkpoint_count": 49,
                        "score_reduction": (
                            args.retrac_reduction
                            if root_key == "retrac_root"
                            else "linear"
                        ),
                        "prediction_sign_for_loss_utility": -1,
                        "train_definition": (
                            "four saved training events, each full-space-L2-normalized before projection, with event LR"
                            if args.paper_retrac and args.paper_train_transform == "raw" and root_key == "retrac_root"
                            else "four saved training events, each transformed by AdamW full, full-space-L2-normalized, then projected, with event LR"
                            if args.paper_retrac and root_key == "retrac_root"
                            else "four exact training events with saved t/noise/dropout and event LR"
                            if root_key == "retrac_root"
                            else f"mean denoising loss over {args.query_timestamp_count} "
                            "timestamps x MC1 and checkpoint LR"
                        ),
                        "query_timestamp_count": args.query_timestamp_count,
                    },
                )
        print(f"[score] Q{query['id']} complete", flush=True)


if __name__ == "__main__":
    main()
