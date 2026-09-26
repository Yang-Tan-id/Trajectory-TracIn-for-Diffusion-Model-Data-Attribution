"""Run exact timestamp-aligned SOURCE-DAS for one family/timestamp shard."""

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
from attribution_one_query import build_model, cond_for
from dataset_loader import ColorGridDataset
from exp_config import *
from source_das_config import *
from source_das_x3 import (
    TimestampAlignedTrainDataset,
    TrajectoryOutputComponentDataset,
    X3SourceComputer,
    X3TimestampSourceTask,
)


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
        (N_TRAIN, SOURCE_DAS_TRAIN_MC, 3, 3, 3),
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
        "--train-batch-size", type=int, default=SOURCE_DAS_TRAIN_SCORE_BATCH_SIZE
    )
    args = parser.parse_args()
    if not 0 <= args.timestamp_shard_index < args.timestamp_shard_count:
        raise ValueError("invalid timestamp shard")
    if args.train_batch_size <= 0:
        raise ValueError("--train-batch-size must be positive")
    logging.basicConfig(
        level=logging.INFO,
        format=f"[source-das gpu={args.gpu} %(name)s] %(message)s",
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
        SOURCE_DAS_SHARD_ROOT / args.family
        / f"shard_{args.timestamp_shard_index:02d}_of_{args.timestamp_shard_count:02d}"
    )
    done_path = shard_root / "done.json"
    if done_path.is_file():
        print(f"[skip] SOURCE-DAS shard complete: {done_path}", flush=True)
        return

    partial_path = shard_root / "partial_scores.npy"
    progress_path = shard_root / "progress.json"
    completed_timestamps = []
    expected_shape = (len(records), N_TRAIN)
    contract = {
        "version": CONTRACT_VERSION,
        "method": SOURCE_DAS_METHOD,
        "family": args.family,
        "query_ids": query_ids,
        "timestamp_indices": selected_timestamps,
        "checkpoint_epochs": list(SOURCE_DAS_CHECKPOINT_EPOCHS),
        "train_mc": SOURCE_DAS_TRAIN_MC,
        "train_batch_size": args.train_batch_size,
    }
    if partial_path.is_file() and progress_path.is_file():
        with open(progress_path) as handle:
            progress = json.load(handle)
        for key, expected in contract.items():
            if progress.get(key) != expected:
                raise ValueError(
                    f"partial contract mismatch {key}: "
                    f"saved={progress.get(key)!r}, current={expected!r}"
                )
        completed_timestamps = [int(value) for value in progress["completed_timestamps"]]
        scores = np.load(partial_path).astype(np.float64)
        if scores.shape != expected_shape:
            raise ValueError(f"partial score shape {scores.shape}, expected {expected_shape}")
        print(
            f"[resume] {args.family} timestamps={len(completed_timestamps)}/"
            f"{len(selected_timestamps)}",
            flush=True,
        )
    else:
        scores = np.zeros(expected_shape, dtype=np.float64)

    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    conditions = [cond_for(item, dataset, device).cpu() for item in records]
    checkpoint_segments = [
        [str(MODEL_DIR / "base" / args.family / f"epoch_{epoch:04d}.pt")]
        for epoch in SOURCE_DAS_CHECKPOINT_EPOCHS
    ]
    iterations = list(source_das_iters_per_segment())
    learning_rates = list(source_das_lrs_per_segment())
    final_path = MODEL_DIR / "base" / args.family / f"epoch_{EPOCHS:04d}.pt"
    remaining = [
        index for index in selected_timestamps if index not in set(completed_timestamps)
    ]
    started = time.perf_counter()

    for position, snapshot_index in enumerate(remaining, start=1):
        timestamp_started = time.perf_counter()
        timestamp = int(timestamps[snapshot_index])
        print(
            f"[source-das {args.family}] timestamp={snapshot_index+1}/"
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
            final_path, SOURCE_DAS_FINAL_PARAM_SOURCE, device
        )
        schedule = base.make_linear_schedule(T, device=device)
        task = X3TimestampSourceTask(timestamp, schedule, device)
        source = X3SourceComputer(
            model=final_model,
            task=task,
            checkpoints_per_segment=checkpoint_segments,
            iters_per_segment=iterations,
            lrs_per_segment=learning_rates,
            n_epoch=1,
            use_true_fisher=SOURCE_DAS_USE_TRUE_FISHER,
        )
        source.build_curvature_blocks(loader=train_loader)
        component_effects = source.compute_scores_with_loader(
            test_loader=query_loader,
            train_loader=train_loader,
        )
        if component_effects.shape != (
            len(records) * SOURCE_DAS_OUTPUT_DIM,
            N_TRAIN,
        ):
            raise ValueError(f"unexpected SOURCE component shape {component_effects.shape}")
        contribution = (
            component_effects.to(torch.float64)
            .reshape(len(records), SOURCE_DAS_OUTPUT_DIM, N_TRAIN)
            .square().sum(dim=1)
            / float(len(timestamps))
        )
        scores += contribution.cpu().numpy()
        completed_timestamps.append(snapshot_index)
        completed_timestamps.sort()
        atomic_numpy(partial_path, scores)
        atomic_json(
            progress_path,
            {**contract, "completed_timestamps": completed_timestamps},
        )
        elapsed = time.perf_counter() - started
        eta = elapsed / position * (len(remaining) - position)
        print(
            f"[source-das {args.family}] completed timestamp="
            f"{snapshot_index+1}/{len(timestamps)} "
            f"timestamp_elapsed={(time.perf_counter()-timestamp_started)/3600:.2f}h "
            f"shard_elapsed={elapsed/3600:.2f}h eta≈{eta/3600:.2f}h",
            flush=True,
        )
        del (
            noises, train_dataset, train_loader, query_dataset, query_loader,
            final_model, schedule, task, source, component_effects, contribution,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if sorted(completed_timestamps) != sorted(selected_timestamps):
        raise RuntimeError("not all assigned timestamps completed")
    atomic_numpy(shard_root / "scores.npy", scores)
    atomic_json(
        done_path,
        {
            **contract,
            "completed_timestamps": completed_timestamps,
            "definition": "mean_t norm2(J_final_ema_predicted_noise @ source_delta_raw_aligned_t_mc10)",
            "source_global_1_over_n_omitted": True,
            "groupnorm_parameters_excluded": True,
            "adamw_preconditioner_ignored": True,
            "simple_influence_root": str(SOURCE_DAS_SIMPLE_INFLUENCE_ROOT),
        },
    )
    print(f"[done] SOURCE-DAS shard: {shard_root}", flush=True)


if __name__ == "__main__":
    main()
