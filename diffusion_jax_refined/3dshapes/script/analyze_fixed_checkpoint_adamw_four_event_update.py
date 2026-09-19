#!/usr/bin/env python3
"""Compare four fixed-checkpoint AdamW event banks with checkpoint motion."""

from __future__ import annotations

import argparse
import csv
import importlib
import pickle
from dataclasses import asdict
from pathlib import Path
import sys

import jax
import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0.0 else float("nan")


def norm_ratio(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a) / max(float(np.linalg.norm(b)), 1e-30))


def load_event_mean(interval_dir: Path, epoch: int, expected_shards: int) -> np.ndarray:
    parts = []
    indices = []
    for shard in range(expected_shards):
        path = interval_dir / (
            f"event_gradient_epoch_{epoch:04d}_shard_{shard:02d}_of_{expected_shards:02d}.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            if str(payload["event_feature"].item()) != "adamw_hypothetical_update":
                raise ValueError(f"{path}: expected adamw_hypothetical_update")
            parts.append(np.asarray(payload["train_features"], dtype=np.float64))
            indices.append(np.asarray(payload["dataset_indices"], dtype=np.int64))
    all_indices = np.concatenate(indices)
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError(f"{interval_dir}: duplicate dataset indices across shards")
    return np.concatenate(parts, axis=0).mean(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--attribution-points", type=int, default=5000)
    parser.add_argument("--proj-dim", type=int, default=4096)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--first-interval", type=int, default=0)
    parser.add_argument("--last-interval", type=int, default=48)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    shapes_root = Path(__file__).resolve().parents[1]
    legacy_root = shapes_root.parent / "legacy_jax"
    sys.path.insert(0, str(legacy_root))
    module = importlib.import_module("DM__training_CIFAR5_MULTI_pixel")
    from dtrak.algorithm import _countsketch_project_grad_jax

    checkpoint_root = shapes_root / "result" / args.experiment / "model" / "prompted_jax"
    artifact_root = (
        shapes_root / "result" / args.experiment
        / f"fixed_checkpoint_adamw_four_events_n{args.attribution_points}"
    )

    # A single template is sufficient to deserialize every checkpoint.
    first_epoch = 4 * (args.first_interval + 1)
    first_path = checkpoint_root / f"seed_{args.train_seed}_epoch_{first_epoch:04d}.ckpt"
    with first_path.open("rb") as handle:
        first_payload = pickle.load(handle)
    cfg = module.TrainConfig(**dict(first_payload["config"]))
    devices = module.choose_devices("cpu")
    device = devices[0]
    dataset = module.CIFAR10Dataset(
        root=cfg.data_root, batch_names=cfg.batch_names, use_test=cfg.use_test,
        class_names=cfg.class_names, normalize="minus_one_to_one", channels_last=True,
        exclude_ranges=cfg.exclude_ranges, exclude_indices=cfg.exclude_indices,
        cond_mode=cfg.cond_mode,
    )
    cfg = module.TrainConfig(**{**asdict(cfg), "num_classes": len(dataset.label_names)})
    total_steps = (len(dataset) // cfg.batch_size) * cfg.epochs
    model = module.build_model(cfg)
    template = module.create_train_state(
        cfg, model, jax.random.PRNGKey(cfg.seed), device, total_steps
    )

    rows = []
    print(
        f"{'CKPT':>4s} {'EPOCH':>5s} {'EVENT1':>9s} {'EVENT2':>9s} "
        f"{'EVENT3':>9s} {'EVENT4':>9s} {'COMBINED':>9s}",
        flush=True,
    )
    print("-" * 69, flush=True)
    for interval in range(args.first_interval, args.last_interval + 1):
        start_epoch = 4 * (interval + 1)
        end_epoch = start_epoch + 4
        start_path = checkpoint_root / f"seed_{args.train_seed}_epoch_{start_epoch:04d}.ckpt"
        end_path = checkpoint_root / f"seed_{args.train_seed}_epoch_{end_epoch:04d}.ckpt"
        start_state, restored_start = module._restore_checkpoint(str(start_path), template)
        end_state, restored_end = module._restore_checkpoint(str(end_path), template)
        if restored_start != start_epoch or restored_end != end_epoch:
            raise RuntimeError(f"checkpoint epoch mismatch: {restored_start}, {restored_end}")

        delta = jax.tree_util.tree_map(
            lambda after, before: after - before, end_state.params, start_state.params
        )
        projected_delta = np.asarray(
            _countsketch_project_grad_jax(
                delta,
                args.proj_dim,
                seed_parts=(args.train_seed, "traj_tracin_projection", interval),
            ),
            dtype=np.float64,
        )
        interval_dir = artifact_root / f"epoch_{start_epoch}_{end_epoch}"
        event_vectors = [
            load_event_mean(interval_dir, epoch, args.num_shards)
            for epoch in range(start_epoch + 1, end_epoch + 1)
        ]
        combined = np.sum(event_vectors, axis=0)
        event_cosines = [cosine(vector, projected_delta) for vector in event_vectors]
        row = {
            "checkpoint": interval + 1,
            "start_epoch": start_epoch,
            "end_epoch": end_epoch,
            **{f"event_{slot}_cosine": value for slot, value in enumerate(event_cosines, 1)},
            "combined_cosine": cosine(combined, projected_delta),
            **{
                f"event_{slot}_norm_ratio": norm_ratio(vector, projected_delta)
                for slot, vector in enumerate(event_vectors, 1)
            },
            "combined_norm_ratio": norm_ratio(combined, projected_delta),
        }
        rows.append(row)
        print(
            f"{interval + 1:4d} {start_epoch:5d} "
            + " ".join(f"{value:+9.4f}" for value in event_cosines)
            + f" {row['combined_cosine']:+9.4f}",
            flush=True,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "per_checkpoint.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    matrix = np.asarray([
        [row[f"event_{slot}_cosine"] for slot in range(1, 5)]
        + [row["combined_cosine"]]
        for row in rows
    ])
    print("\nFIXED-CHECKPOINT ADAMW FOUR EVENTS vs NEXT CHECKPOINT UPDATE")
    print("columns: EVENT1 EVENT2 EVENT3 EVENT4 COMBINED")
    print("mean   " + " ".join(f"{value:+9.4f}" for value in matrix.mean(axis=0)))
    print("median " + " ".join(f"{value:+9.4f}" for value in np.median(matrix, axis=0)))
    print(">0     " + " ".join(f"{value:9.3f}" for value in (matrix > 0).mean(axis=0)))
    print(f"[saved] {args.out_dir / 'per_checkpoint.csv'}")


if __name__ == "__main__":
    main()
