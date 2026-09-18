#!/usr/bin/env python3
"""Replay one saved training interval and persist its exact stochastic events.

This is deliberately a validation tool first.  A per-example-gradient extractor
is only meaningful after the replayed end state agrees with the saved next
checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def flat_metrics(a, b):
    av = np.concatenate([np.asarray(x, np.float64).ravel() for x in jax.tree_util.tree_leaves(a)])
    bv = np.concatenate([np.asarray(x, np.float64).ravel() for x in jax.tree_util.tree_leaves(b)])
    delta = av - bv
    denom = max(float(np.linalg.norm(bv)), 1e-30)
    cos_denom = max(float(np.linalg.norm(av) * np.linalg.norm(bv)), 1e-30)
    return {
        "relative_l2": float(np.linalg.norm(delta) / denom),
        "cosine": float(np.dot(av, bv) / cos_denom),
        "max_abs": float(np.max(np.abs(delta))),
        "reference_l2": float(np.linalg.norm(bv)),
    }


def displacement_metrics(replayed, target, start):
    replay_delta = jax.tree_util.tree_map(lambda x, x0: x - x0, replayed, start)
    target_delta = jax.tree_util.tree_map(lambda x, x0: x - x0, target, start)
    metrics = flat_metrics(replay_delta, target_delta)
    replay_norm = np.sqrt(sum(
        float(np.sum(np.square(np.asarray(x, dtype=np.float64))))
        for x in jax.tree_util.tree_leaves(replay_delta)
    ))
    target_norm = metrics["reference_l2"]
    metrics["norm_ratio"] = float(replay_norm / max(target_norm, 1e-30))
    return metrics


def key_words(key):
    return np.asarray(jax.random.key_data(key), dtype=np.uint32).reshape(-1).tolist()


def load_payload(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--start-epoch", type=int, default=40)
    parser.add_argument("--end-epoch", type=int, default=44)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--extract-gradient-sketches", action="store_true")
    parser.add_argument("--proj-dim", type=int, default=4096)
    parser.add_argument(
        "--event-feature", choices=(
            "raw_gradient", "adamw_local_update", "adamw_hypothetical_update"
        ),
        default="raw_gradient",
    )
    parser.add_argument(
        "--fixed-checkpoint", action="store_true",
        help="Keep params and optimizer state fixed; only advance RNG to recover four historical events.",
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.end_epoch <= args.start_epoch:
        raise ValueError("--end-epoch must be greater than --start-epoch")

    shapes_root = Path(__file__).resolve().parents[1]
    legacy = shapes_root.parent / "legacy_jax"
    sys.path.insert(0, str(legacy))
    # The 3D-Shapes module re-exports this implementation with ``import *``;
    # private checkpoint helpers are therefore available only on the base module.
    module = importlib.import_module("DM__training_CIFAR5_MULTI_pixel")
    from dtrak.algorithm import build_countsketch_projector_jax
    ckpt_dir = shapes_root / "result" / args.experiment / "model" / "prompted_jax"
    start_path = ckpt_dir / f"seed_{args.train_seed}_epoch_{args.start_epoch:04d}.ckpt"
    end_path = ckpt_dir / f"seed_{args.train_seed}_epoch_{args.end_epoch:04d}.ckpt"
    start_payload = load_payload(start_path)
    end_payload = load_payload(end_path)
    cfg_dict = dict(start_payload["config"])
    cfg = module.TrainConfig(**cfg_dict)

    devices = module.choose_devices(cfg.prefer_device)
    requested = cfg.num_devices if cfg.num_devices is not None else len(devices)
    use_pmap = bool(cfg.use_data_parallel and requested > 1)
    if use_pmap:
        raise RuntimeError(
            "Exact replay currently requires the original run to be single-device. "
            f"Checkpoint config says use_data_parallel={cfg.use_data_parallel}, num_devices={cfg.num_devices}; "
            f"visible devices={len(devices)}. Add the original pmap topology before trusting a replay."
        )
    device = devices[0]

    ds = module.CIFAR10Dataset(
        root=cfg.data_root, batch_names=cfg.batch_names, use_test=cfg.use_test,
        class_names=cfg.class_names, normalize="minus_one_to_one", channels_last=True,
        exclude_ranges=cfg.exclude_ranges, exclude_indices=cfg.exclude_indices,
        cond_mode=cfg.cond_mode,
    )
    cfg = module.TrainConfig(**{**asdict(cfg), "num_classes": len(ds.label_names)})
    steps_per_epoch = len(ds) // cfg.batch_size
    total_steps = steps_per_epoch * cfg.epochs
    model = module.build_model(cfg)
    template = module.create_train_state(
        cfg, model, jax.random.PRNGKey(cfg.seed), device, total_steps
    )
    state, restored_epoch = module._restore_checkpoint(str(start_path), template)
    target, target_epoch = module._restore_checkpoint(str(end_path), template)
    initial_state = state
    if restored_epoch != args.start_epoch or target_epoch != args.end_epoch:
        raise RuntimeError(f"checkpoint epoch mismatch: {restored_epoch}, {target_epoch}")
    schedule = module.make_diffusion_schedule(cfg.timesteps, cfg.beta_start, cfg.beta_end)
    train_step = module.make_train_step(schedule, cfg)
    eval_step = module.make_eval_step(schedule, cfg)

    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must be in [0, --num-shards)")
    owned_indices = np.arange(args.shard_id, len(ds), args.num_shards, dtype=np.int64)
    owned_row = np.full((len(ds),), -1, dtype=np.int64)
    owned_row[owned_indices] = np.arange(len(owned_indices), dtype=np.int64)
    projector = None
    event_gradient = None
    batch_gradient = None
    if args.extract_gradient_sketches:
        checkpoint_index = args.start_epoch // 4 - 1
        projector = build_countsketch_projector_jax(
            state.params,
            args.proj_dim,
            seed_parts=(cfg.seed, "traj_tracin_projection", checkpoint_index),
            device=device,
        )

        def per_example_losses(params, x, y, noise, timesteps, dropout_rng):
            xt = module.q_sample(schedule, x, timesteps, noise)
            target_value = x if cfg.predict_x0 else noise

            pred = state.apply_fn(
                {"params": params}, xt, timesteps,
                y if cfg.class_cond else None,
                train=True, rngs={"dropout": dropout_rng},
            )
            axes = tuple(range(1, pred.ndim))
            return jnp.mean((pred - target_value) ** 2, axis=axes)

        def batch_gradient_fn(params, x, y, noise, timesteps, dropout_rng):
            def loss_fn(pp):
                return jnp.mean(per_example_losses(pp, x, y, noise, timesteps, dropout_rng))
            return jax.grad(loss_fn)(params)

        def one_event_gradient(
            params, opt_state, batch_grads, x, y, noise, timesteps,
            dropout_rng, position,
        ):

            def selected_loss(pp):
                return per_example_losses(pp, x, y, noise, timesteps, dropout_rng)[position]

            grads = jax.grad(selected_loss)(params)
            if args.event_feature == "raw_gradient":
                feature_tree = grads
            elif args.event_feature == "adamw_hypothetical_update":
                feature_tree, _ = state.tx.update(grads, opt_state, params)
            else:
                tangent = jax.tree_util.tree_map(
                    lambda value: value / jnp.asarray(cfg.batch_size, dtype=value.dtype),
                    grads,
                )

                def optimizer_updates(candidate_grads):
                    updates, _ = state.tx.update(candidate_grads, opt_state, params)
                    return updates

                _, feature_tree = jax.jvp(
                    optimizer_updates, (batch_grads,), (tangent,)
                )
            return projector(feature_tree)

        event_gradient = jax.jit(one_event_gradient)
        batch_gradient = jax.jit(batch_gradient_fn)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    event_name = (
        "batch_events.csv" if args.num_shards == 1
        else f"batch_events_shard_{args.shard_id:02d}_of_{args.num_shards:02d}.csv"
    )
    event_path = args.out_dir / event_name
    fieldnames = [
        "epoch", "batch", "global_step", "dataset_indices", "timesteps",
        "noise_key", "dropout_key", "next_state_key",
    ]
    with event_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        global_step = args.start_epoch * steps_per_epoch
        for epoch in range(args.start_epoch + 1, args.end_epoch + 1):
            gradient_part = (
                args.out_dir
                / f"event_gradient_epoch_{epoch:04d}_shard_{args.shard_id:02d}_of_{args.num_shards:02d}.npz"
            )
            compute_epoch_gradients = args.extract_gradient_sketches and not gradient_part.exists()
            if compute_epoch_gradients:
                epoch_features = np.empty((len(owned_indices), args.proj_dim), dtype=np.float32)
                epoch_timesteps = np.empty((len(owned_indices),), dtype=np.int32)
                epoch_batches = np.empty((len(owned_indices),), dtype=np.int32)
                epoch_positions = np.empty((len(owned_indices),), dtype=np.int16)
            indices = np.arange(len(ds), dtype=np.int64)
            np.random.default_rng(cfg.seed + epoch).shuffle(indices)
            for batch_no, start in enumerate(range(0, steps_per_epoch * cfg.batch_size, cfg.batch_size)):
                selected = indices[start:start + cfg.batch_size]
                x = jax.device_put(module.maybe_to_dtype(jnp.asarray(ds.images[selected]), cfg.use_bfloat16), device)
                y = jax.device_put(jnp.asarray(ds.labels[selected]), device)
                next_rng, noise_rng, t_rng, dropout_rng = jax.random.split(state.rng, 4)
                timesteps = np.asarray(
                    jax.random.randint(t_rng, (cfg.batch_size,), 0, cfg.timesteps), np.int32
                )
                if compute_epoch_gradients:
                    noise = jax.random.normal(noise_rng, x.shape, dtype=x.dtype)
                    t_device = jax.device_put(jnp.asarray(timesteps), device)
                    if args.event_feature == "adamw_local_update":
                        batch_grads = batch_gradient(
                            state.params, x, y, noise, t_device, dropout_rng
                        )
                    else:
                        # This argument is unused by raw/fixed-state hypothetical modes.
                        batch_grads = state.params
                    for position, dataset_index in enumerate(selected.tolist()):
                        row = int(owned_row[dataset_index])
                        if row < 0:
                            continue
                        feature = event_gradient(
                            state.params, state.opt_state, batch_grads,
                            x, y, noise, t_device, dropout_rng,
                            jnp.asarray(position, dtype=jnp.int32),
                        )
                        epoch_features[row] = np.asarray(feature, dtype=np.float32)
                        epoch_timesteps[row] = timesteps[position]
                        epoch_batches[row] = batch_no
                        epoch_positions[row] = position
                global_step += 1
                writer.writerow({
                    "epoch": epoch,
                    "batch": batch_no,
                    "global_step": global_step,
                    "dataset_indices": json.dumps(selected.tolist(), separators=(",", ":")),
                    "timesteps": json.dumps(timesteps.tolist(), separators=(",", ":")),
                    "noise_key": json.dumps(key_words(noise_rng)),
                    "dropout_key": json.dumps(key_words(dropout_rng)),
                    "next_state_key": json.dumps(key_words(next_rng)),
                })
                if args.fixed_checkpoint:
                    # Random-event recovery only. Parameters and AdamW history remain
                    # exactly those stored in the interval's starting checkpoint.
                    state = state.replace(rng=next_rng)
                else:
                    state, _ = train_step(state, x, y)
                if (batch_no + 1) % 100 == 0:
                    print(f"[replay] epoch={epoch} batch={batch_no + 1}/{steps_per_epoch}", flush=True)

            # Original training consumes exactly one deterministic eval batch/RNG step per epoch.
            eval_selected = np.arange(min(cfg.batch_size, len(ds)), dtype=np.int64)
            eval_x = jax.device_put(
                module.maybe_to_dtype(jnp.asarray(ds.images[eval_selected]), cfg.use_bfloat16), device
            )
            eval_y = jax.device_put(jnp.asarray(ds.labels[eval_selected]), device)
            if args.fixed_checkpoint:
                eval_next_rng, _, _ = jax.random.split(state.rng, 3)
                state = state.replace(rng=eval_next_rng)
            else:
                state, _ = eval_step(state, eval_x, eval_y)
            if compute_epoch_gradients:
                temporary_part = gradient_part.with_suffix(".tmp.npz")
                np.savez_compressed(
                    temporary_part, train_features=epoch_features,
                    dataset_indices=owned_indices, timesteps=epoch_timesteps,
                    batch_indices=epoch_batches, batch_positions=epoch_positions,
                    epoch=np.asarray(epoch, dtype=np.int32),
                    start_epoch=np.asarray(args.start_epoch, dtype=np.int32),
                    proj_dim=np.asarray(args.proj_dim, dtype=np.int32),
                    projection_checkpoint_index=np.asarray(checkpoint_index, dtype=np.int32),
                    feature_definition=np.asarray({
                        "raw_gradient": "CountSketch(fixed/replayed per-example diffusion-loss gradient)",
                        "adamw_local_update": "CountSketch(local AdamW update response to per-example gradient / batch_size)",
                        "adamw_hypothetical_update": "CountSketch(AdamW update from one per-example gradient using fixed checkpoint optimizer state)",
                    }[args.event_feature]),
                    event_feature=np.asarray(args.event_feature),
                )
                os.replace(temporary_part, gradient_part)
                print(f"[gradient saved] {gradient_part}", flush=True)
            print(f"[replay] epoch={epoch} complete (including eval RNG advance)", flush=True)

    report = {
        "start_checkpoint": str(start_path),
        "end_checkpoint": str(end_path),
        "start_epoch": args.start_epoch,
        "end_epoch": args.end_epoch,
        "dataset_size": len(ds),
        "batch_size": cfg.batch_size,
        "steps_per_epoch": steps_per_epoch,
        "events": (args.end_epoch - args.start_epoch) * steps_per_epoch,
        "gradient_sketches": bool(args.extract_gradient_sketches),
        "event_feature": args.event_feature,
        "fixed_checkpoint": bool(args.fixed_checkpoint),
        "gradient_shard": [args.shard_id, args.num_shards],
        "params": flat_metrics(state.params, target.params),
        "ema_params": flat_metrics(state.ema_params, target.ema_params),
        "parameter_displacement": displacement_metrics(
            state.params, target.params, initial_state.params
        ),
        "ema_displacement": displacement_metrics(
            state.ema_params, target.ema_params, initial_state.ema_params
        ),
        "rng_equal": bool(np.array_equal(np.asarray(state.rng), np.asarray(target.rng))),
        "step_replayed": int(np.asarray(state.step)),
        "step_target": int(np.asarray(target.step)),
    }
    report_name = (
        "replay_report.json" if args.num_shards == 1
        else f"replay_report_shard_{args.shard_id:02d}_of_{args.num_shards:02d}.json"
    )
    with (args.out_dir / report_name).open("w") as handle:
        json.dump(report, handle, indent=2)
    print("EXACT TRAINING-INTERVAL REPLAY VALIDATION")
    print(json.dumps(report, indent=2))
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
