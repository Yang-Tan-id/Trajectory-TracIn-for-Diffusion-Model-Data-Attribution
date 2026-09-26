"""Run Adam/clipping-aware raw SOURCE-DAS for one family/timestamp shard."""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import x3pixel_DM_training as base
from adam_clip_source_das_config import *
from adam_clip_source_das_x3 import X3AdamClipSourceComputer
from attribution_one_query import build_model, cond_for
from dataset_loader import ColorGridDataset
from source_das_x3 import TimestampAlignedTrainDataset, TrajectoryOutputComponentDataset


CONTRACT_VERSION = 1


def atomic_numpy(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def aligned_noises(timestamp_index, timestamp):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        int(TRAIN_SEED) * 1_000_003
        + int(timestamp_index) * 10_007
        + int(timestamp)
        + 740_000_000
    )
    return torch.randn(
        (N_TRAIN, ADAM_CLIP_SOURCE_TRAIN_MC, 3, 3, 3),
        generator=generator,
        dtype=torch.float32,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--timestamp-shard-index", type=int, required=True)
    parser.add_argument("--timestamp-shard-count", type=int, required=True)
    parser.add_argument(
        "--train-batch-size", type=int, default=ADAM_CLIP_SOURCE_TRAIN_BATCH_SIZE
    )
    args = parser.parse_args()
    if not 0 <= args.timestamp_shard_index < args.timestamp_shard_count:
        raise ValueError("invalid timestamp shard")
    if args.train_batch_size != BATCH_SIZE:
        raise ValueError(
            f"clipping replay requires original training batch size {BATCH_SIZE}; "
            f"got {args.train_batch_size}"
        )
    logging.basicConfig(
        level=logging.INFO,
        format=f"[adam-clip-source gpu={args.gpu} %(name)s] %(message)s",
    )
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    with open(QUERY_DIR / "manifest.json") as handle:
        records = [item for item in json.load(handle) if item["family"] == args.family]
    query_ids = [int(item["query_id"]) for item in records]
    trajectories = [np.load(Path(item["dir"]) / "trajectory_xt.npy") for item in records]
    timestamp_arrays = [np.load(Path(item["dir"]) / "trajectory_t.npy") for item in records]
    timestamps = timestamp_arrays[0]
    if any(not np.array_equal(timestamps, values) for values in timestamp_arrays):
        raise ValueError("query timestamp arrays differ")
    selected_timestamps = list(
        range(args.timestamp_shard_index, len(timestamps), args.timestamp_shard_count)
    )
    shard_root = (
        ADAM_CLIP_SOURCE_SHARD_ROOT
        / args.family
        / f"shard_{args.timestamp_shard_index:02d}_of_{args.timestamp_shard_count:02d}"
    )
    done_path = shard_root / "done.json"
    if done_path.is_file():
        print(f"[skip] Adam/clipping SOURCE shard complete: {done_path}", flush=True)
        return

    partial_paths = {
        variant: shard_root / f"partial_scores_{variant}.npy"
        for variant in ADAM_CLIP_SOURCE_METHODS
    }
    progress_path = shard_root / "progress.json"
    expected_shape = (len(records), N_TRAIN)
    contract = {
        "version": CONTRACT_VERSION,
        "family": args.family,
        "methods": ADAM_CLIP_SOURCE_METHODS,
        "query_ids": query_ids,
        "timestamp_indices": selected_timestamps,
        "checkpoint_epochs": list(ADAM_CLIP_SOURCE_CHECKPOINT_EPOCHS),
        "train_mc": ADAM_CLIP_SOURCE_TRAIN_MC,
        "train_batch_size": args.train_batch_size,
        "clip_norm": ADAM_CLIP_SOURCE_CLIP_NORM,
        "final_parameter_source": ADAM_CLIP_SOURCE_FINAL_PARAM_SOURCE,
    }
    completed_timestamps = []
    timestamp_diagnostics = {}
    if progress_path.is_file() and all(path.is_file() for path in partial_paths.values()):
        with open(progress_path) as handle:
            progress = json.load(handle)
        for key, expected in contract.items():
            if progress.get(key) != expected:
                raise ValueError(
                    f"partial contract mismatch {key}: "
                    f"saved={progress.get(key)!r}, current={expected!r}"
                )
        completed_timestamps = [int(value) for value in progress["completed_timestamps"]]
        timestamp_diagnostics = progress.get("timestamp_diagnostics", {})
        scores = {
            variant: np.load(path).astype(np.float64)
            for variant, path in partial_paths.items()
        }
        if any(value.shape != expected_shape for value in scores.values()):
            raise ValueError("partial score shape mismatch")
        print(
            f"[resume] {args.family} timestamps={len(completed_timestamps)}/"
            f"{len(selected_timestamps)}",
            flush=True,
        )
    else:
        scores = {
            variant: np.zeros(expected_shape, dtype=np.float64)
            for variant in ADAM_CLIP_SOURCE_METHODS
        }

    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    conditions = [cond_for(item, dataset, device).cpu() for item in records]
    checkpoint_segments = [
        [str(MODEL_DIR / "base" / args.family / f"epoch_{epoch:04d}.pt")]
        for epoch in ADAM_CLIP_SOURCE_CHECKPOINT_EPOCHS
    ]
    iterations = list(adam_clip_source_iters_per_segment())
    lr_sums = list(adam_clip_source_lr_sums_per_segment())
    learning_rates = [total / count for total, count in zip(lr_sums, iterations)]
    final_path = MODEL_DIR / "base" / args.family / f"epoch_{EPOCHS:04d}.pt"
    completed_set = set(completed_timestamps)
    remaining = [index for index in selected_timestamps if index not in completed_set]
    started = time.perf_counter()

    for position, snapshot_index in enumerate(remaining, start=1):
        timestamp_started = time.perf_counter()
        timestamp = int(timestamps[snapshot_index])
        print(
            f"[adam-clip-source {args.family}] timestamp={snapshot_index+1}/"
            f"{len(timestamps)} value={timestamp} "
            f"remaining_position={position}/{len(remaining)}",
            flush=True,
        )
        noises = aligned_noises(snapshot_index, timestamp)
        train_dataset = TimestampAlignedTrainDataset(dataset, args.family, noises)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        query_dataset = TrajectoryOutputComponentDataset(
            records, trajectories, conditions, snapshot_index
        )
        query_loader = DataLoader(
            query_dataset,
            batch_size=len(query_dataset),
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        final_model, _, _ = build_model(
            final_path, ADAM_CLIP_SOURCE_FINAL_PARAM_SOURCE, device
        )
        schedule = base.make_linear_schedule(T, device=device)
        from source_das_x3 import X3TimestampSourceTask

        task = X3TimestampSourceTask(timestamp, schedule, device)
        source = X3AdamClipSourceComputer(
            model=final_model,
            task=task,
            checkpoints_per_segment=checkpoint_segments,
            iters_per_segment=iterations,
            lrs_per_segment=learning_rates,
            n_epoch=1,
            use_true_fisher=False,
        )
        source.build_adam_clip_blocks(loader=train_loader)
        component_effects = source.compute_score_variants_with_loader(
            test_loader=query_loader,
            train_loader=train_loader,
        )
        for variant, effects in component_effects.items():
            expected_components = (
                len(records) * ADAM_CLIP_SOURCE_OUTPUT_DIM,
                N_TRAIN,
            )
            if tuple(effects.shape) != expected_components:
                raise ValueError(
                    f"unexpected {variant} component shape {effects.shape}"
                )
            contribution = (
                effects.to(torch.float64)
                .reshape(len(records), ADAM_CLIP_SOURCE_OUTPUT_DIM, N_TRAIN)
                .square()
                .sum(dim=1)
                / float(len(timestamps))
            )
            scores[variant] += contribution.cpu().numpy()
            atomic_numpy(partial_paths[variant], scores[variant])

        completed_timestamps.append(snapshot_index)
        completed_timestamps.sort()
        timestamp_diagnostics[str(snapshot_index)] = source.diagnostics()
        atomic_json(
            progress_path,
            {
                **contract,
                "completed_timestamps": completed_timestamps,
                "timestamp_diagnostics": timestamp_diagnostics,
            },
        )
        elapsed = time.perf_counter() - started
        eta = elapsed / position * (len(remaining) - position)
        print(
            f"[adam-clip-source {args.family}] completed timestamp="
            f"{snapshot_index+1}/{len(timestamps)} "
            f"timestamp_elapsed={(time.perf_counter()-timestamp_started)/3600:.2f}h "
            f"shard_elapsed={elapsed/3600:.2f}h eta≈{eta/3600:.2f}h",
            flush=True,
        )
        del component_effects, source, final_model, query_loader, train_loader, noises
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if sorted(completed_timestamps) != sorted(selected_timestamps):
        raise RuntimeError("not all assigned timestamps completed")
    for variant in ADAM_CLIP_SOURCE_METHODS:
        atomic_numpy(shard_root / f"scores_{variant}.npy", scores[variant])
    atomic_json(
        done_path,
        {
            **contract,
            "completed_timestamps": completed_timestamps,
            "timestamp_diagnostics": timestamp_diagnostics,
            "definition": "mean_t ||J_final_raw_on_cached_ema_trajectory @ adam_clip_source_delta||_2^2",
            "clip_jacobian": "frozen batch-scale approximation; dc/dtheta omitted",
            "query_normalization": {
                "unnormalized": "none",
                "jacobian_fro_rms": "one exact ||J_qt||_F/sqrt(27) denominator shared by all 27 output components; selected parameters only",
            },
            "weight_decay_in_decay_operator": False,
            "adam_first_moment_used": False,
            "adam_second_moment_used": True,
        },
    )
    print(f"[done] Adam/clipping SOURCE shard: {shard_root}", flush=True)


if __name__ == "__main__":
    main()
