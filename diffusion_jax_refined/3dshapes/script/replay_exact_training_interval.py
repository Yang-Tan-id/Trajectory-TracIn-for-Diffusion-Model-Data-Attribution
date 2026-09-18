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
    args = parser.parse_args()
    if args.end_epoch <= args.start_epoch:
        raise ValueError("--end-epoch must be greater than --start-epoch")

    shapes_root = Path(__file__).resolve().parents[1]
    legacy = shapes_root.parent / "legacy_jax"
    sys.path.insert(0, str(legacy))
    # The 3D-Shapes module re-exports this implementation with ``import *``;
    # private checkpoint helpers are therefore available only on the base module.
    module = importlib.import_module("DM__training_CIFAR5_MULTI_pixel")
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
    if restored_epoch != args.start_epoch or target_epoch != args.end_epoch:
        raise RuntimeError(f"checkpoint epoch mismatch: {restored_epoch}, {target_epoch}")
    schedule = module.make_diffusion_schedule(cfg.timesteps, cfg.beta_start, cfg.beta_end)
    train_step = module.make_train_step(schedule, cfg)
    eval_step = module.make_eval_step(schedule, cfg)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    event_path = args.out_dir / "batch_events.csv"
    fieldnames = [
        "epoch", "batch", "global_step", "dataset_indices", "timesteps",
        "noise_key", "dropout_key", "next_state_key",
    ]
    with event_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        global_step = args.start_epoch * steps_per_epoch
        for epoch in range(args.start_epoch + 1, args.end_epoch + 1):
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
                state, _ = train_step(state, x, y)
                if (batch_no + 1) % 100 == 0:
                    print(f"[replay] epoch={epoch} batch={batch_no + 1}/{steps_per_epoch}", flush=True)

            # Original training consumes exactly one deterministic eval batch/RNG step per epoch.
            eval_selected = np.arange(min(cfg.batch_size, len(ds)), dtype=np.int64)
            eval_x = jax.device_put(
                module.maybe_to_dtype(jnp.asarray(ds.images[eval_selected]), cfg.use_bfloat16), device
            )
            eval_y = jax.device_put(jnp.asarray(ds.labels[eval_selected]), device)
            state, _ = eval_step(state, eval_x, eval_y)
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
        "params": flat_metrics(state.params, target.params),
        "ema_params": flat_metrics(state.ema_params, target.ema_params),
        "rng_equal": bool(np.array_equal(np.asarray(state.rng), np.asarray(target.rng))),
        "step_replayed": int(np.asarray(state.step)),
        "step_target": int(np.asarray(target.step)),
    }
    with (args.out_dir / "replay_report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    print("EXACT TRAINING-INTERVAL REPLAY VALIDATION")
    print(json.dumps(report, indent=2))
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
