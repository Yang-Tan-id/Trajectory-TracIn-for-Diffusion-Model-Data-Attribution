"""Replay checkpoint intervals and diagnose predicted-noise response errors."""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp
from torch.utils.data import DataLoader, Dataset

import x3pixel_DM_training as base
from attribution_one_query import build_model, cond_for, model_paths, preload_dataset
from checkpoint_transition_diagnostic_config import *
from dataset_loader import ColorGridDataset
from forward_loss_alignment_config import replay_noise_path, replay_t_path
from train_worker import lr_at, set_seed


ORIGINAL_TIME_EMBEDDING = base.sinusoidal_time_embedding


def configure_training_precision():
    """Restore the model arithmetic used by the original float32 training."""
    base.sinusoidal_time_embedding = ORIGINAL_TIME_EMBEDDING
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def configure_evaluation_precision():
    """Use deterministic float64 arithmetic for small response differences."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    def float64_time_embedding(timestep, dimension):
        half = dimension // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(0, half, device=timestep.device, dtype=torch.float64)
            / (half - 1)
        )
        arguments = timestep.to(torch.float64).unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((torch.sin(arguments), torch.cos(arguments)), dim=1)
        if dimension % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    base.sinusoidal_time_embedding = float64_time_embedding


class IndexDataset(Dataset):
    def __len__(self):
        return N_TRAIN

    def __getitem__(self, index):
        return int(index)


def checkpoint_orders(dataset, device, needed_target_indices):
    """Recreate original shuffled minibatches for selected checkpoint intervals."""
    set_seed(TRAIN_SEED)
    loader = DataLoader(
        IndexDataset(),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    image, condition = dataset[0]
    dummy = base.CondEpsModel(
        in_ch=int(image.shape[0]),
        cond_dim=int(condition.numel()),
        base_ch=BASE_CH,
        time_dim=TIME_DIM,
    ).to(device)
    dummy_ema = copy.deepcopy(dummy).to(device).eval()
    dummy_schedule = base.make_linear_schedule(T, device=device)
    dummy_optimizer = torch.optim.AdamW(
        dummy.parameters(),
        lr=PEAK_LR,
        betas=(ADAM_B1, ADAM_B2),
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
    )
    del dummy, dummy_ema, dummy_schedule, dummy_optimizer

    needed = set(needed_target_indices)
    result = {index: [[] for _ in range(CTD_EVENTS_PER_INTERVAL)] for index in needed}
    for epoch in range(1, EPOCHS + 1):
        checkpoint_index = (epoch - 1) // CTD_EVENTS_PER_INTERVAL
        event_index = (epoch - 1) % CTD_EVENTS_PER_INTERVAL
        keep = checkpoint_index in needed
        for indices in loader:
            if keep:
                result[checkpoint_index][event_index].append(indices.numpy().copy())
    return result, len(loader)


def parameter_agreement(predicted, actual, eps=1e-30):
    dot = sum((left.double() * right.double()).sum() for left, right in zip(predicted, actual))
    predicted_sq = sum(left.double().square().sum() for left in predicted)
    actual_sq = sum(right.double().square().sum() for right in actual)
    cosine = dot / (predicted_sq.sqrt() * actual_sq.sqrt()).clamp_min(eps)
    error_sq = sum(
        (left.double() - right.double()).square().sum()
        for left, right in zip(predicted, actual)
    )
    return {
        "cosine": float(cosine),
        "relative_error": float(error_sq.sqrt() / actual_sq.sqrt().clamp_min(eps)),
        "predicted_norm": float(predicted_sq.sqrt()),
        "actual_norm": float(actual_sq.sqrt()),
    }


def point_metrics(predicted, actual, eps=1e-30):
    predicted = predicted.detach().double().flatten(1)
    actual = actual.detach().double().flatten(1)
    predicted_norm = predicted.norm(dim=1)
    actual_norm = actual.norm(dim=1)
    return {
        "predicted_l2": predicted_norm.cpu().numpy(),
        "vector_cosine": (
            (predicted * actual).sum(dim=1)
            / (predicted_norm * actual_norm).clamp_min(eps)
        ).cpu().numpy(),
        "vector_relative_error": (
            (predicted - actual).norm(dim=1) / actual_norm.clamp_min(eps)
        ).cpu().numpy(),
        "magnitude_relative_error": (
            (predicted_norm - actual_norm).abs() / actual_norm.clamp_min(eps)
        ).cpu().numpy(),
    }


def query_bank(records, dataset, device):
    states = []
    timesteps = []
    conditions = []
    query_ids = []
    timestamp_positions = []
    for record in records:
        query_id = int(record["query_id"])
        trajectory = np.load(Path(record["dir"]) / "trajectory_xt.npy")
        trajectory_t = np.load(Path(record["dir"]) / "trajectory_t.npy")
        if len(trajectory_t) != TRAJ_SNAPSHOTS:
            raise ValueError(f"q{query_id:02d}: expected {TRAJ_SNAPSHOTS} timestamps")
        condition = cond_for(record, dataset, device).cpu().numpy()
        states.append(trajectory[:, 0])
        timesteps.append(trajectory_t)
        conditions.append(np.repeat(condition, TRAJ_SNAPSHOTS, axis=0))
        query_ids.extend([query_id] * TRAJ_SNAPSHOTS)
        timestamp_positions.extend(range(TRAJ_SNAPSHOTS))
    return {
        "states": np.concatenate(states, axis=0),
        "timesteps": np.concatenate(timesteps, axis=0),
        "conditions": np.concatenate(conditions, axis=0),
        "query_ids": np.asarray(query_ids, dtype=np.int64),
        "timestamp_positions": np.asarray(timestamp_positions, dtype=np.int64),
    }


def evaluate_responses(
    start_model,
    target_state,
    replay_state,
    current_raw_tangent,
    current_scaled_tangent,
    bank,
    device,
    batch_size,
):
    # Float64 avoids cancellation when measuring small checkpoint differences.
    configure_evaluation_precision()
    start_model = start_model.double().eval()
    names = tuple(name for name, _ in start_model.named_parameters())
    initial = tuple(parameter.detach() for parameter in start_model.parameters())
    target = tuple(target_state[name].to(device=device, dtype=torch.float64) for name in names)
    replay = tuple(replay_state[name].to(device=device, dtype=torch.float64) for name in names)
    current_raw = tuple(
        current_raw_tangent[name].to(device=device, dtype=torch.float64)
        for name in names
    )
    current_scaled = tuple(
        current_scaled_tangent[name].to(device=device, dtype=torch.float64)
        for name in names
    )
    exact_delta = tuple(after - before for before, after in zip(initial, target))
    replay_delta = tuple(after - before for before, after in zip(initial, replay))

    total = len(bank["states"])
    arrays = {"actual_l2": np.empty(total, dtype=np.float64)}
    for method in CTD_METHODS:
        for metric in (
            "predicted_l2",
            "vector_cosine",
            "vector_relative_error",
            "magnitude_relative_error",
        ):
            arrays[f"{method}_{metric}"] = np.empty(total, dtype=np.float64)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        x = torch.from_numpy(bank["states"][start:end]).to(
            device=device, dtype=torch.float64
        )
        t = torch.from_numpy(bank["timesteps"][start:end]).to(
            device=device, dtype=torch.long
        )
        condition = torch.from_numpy(bank["conditions"][start:end]).to(
            device=device, dtype=torch.float64
        )

        def prediction(*parameter_values):
            return functional_call(
                start_model,
                dict(zip(names, parameter_values)),
                (x, t, condition),
            )

        with torch.no_grad():
            before = prediction(*initial)
            after = prediction(*target)
            actual = after - before
        predictions = {
            "exact_parameter_delta_start_jvp": jvp(
                prediction, initial, exact_delta
            )[1],
            "replayed_adamw_start_jvp": jvp(
                prediction, initial, replay_delta
            )[1],
            "current_interval_scaled_start_jvp": jvp(
                prediction, initial, current_scaled
            )[1],
        }

        def target_prediction(*parameter_values):
            return functional_call(
                start_model,
                dict(zip(names, parameter_values)),
                (x, t, condition),
            )

        predictions["current_bundle_raw_target_jvp"] = jvp(
            target_prediction, target, current_raw
        )[1]
        predictions["current_interval_scaled_target_jvp"] = jvp(
            target_prediction, target, current_scaled
        )[1]
        arrays["actual_l2"][start:end] = (
            actual.flatten(1).norm(dim=1).cpu().numpy()
        )
        for method, predicted in predictions.items():
            for metric, value in point_metrics(predicted, actual).items():
                arrays[f"{method}_{metric}"][start:end] = value
    return arrays, parameter_agreement(replay_delta, exact_delta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pair-shard-index", type=int, required=True)
    parser.add_argument("--pair-shard-count", type=int, required=True)
    parser.add_argument("--pair-indices", default="0-48")
    parser.add_argument("--query-ids", default="0-9")
    parser.add_argument("--query-batch-size", type=int, default=CTD_QUERY_BATCH_SIZE)
    args = parser.parse_args()

    pair_indices = parse_integer_selection(args.pair_indices, CTD_PAIR_INDICES)
    query_ids = parse_integer_selection(args.query_ids, CTD_QUERY_IDS)
    if any(index < 0 or index >= 49 for index in pair_indices):
        raise ValueError("pair indices must be in [0, 48]")
    assigned_pairs = pair_indices[
        args.pair_shard_index :: args.pair_shard_count
    ]
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    x_all, condition_all = preload_dataset(dataset, args.family, device)
    schedule = base.make_linear_schedule(T, device=device)
    paths = model_paths(args.family)
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    records = [
        record
        for record in manifest
        if int(record["query_id"]) in query_ids and record["family"] == args.family
    ]
    if not records:
        print(f"[skip] no selected queries for family={args.family}", flush=True)
        return
    bank = query_bank(records, dataset, device)
    target_indices = [pair_index + 1 for pair_index in assigned_pairs]
    orders, batches_per_epoch = checkpoint_orders(dataset, device, target_indices)
    replay_t = np.load(replay_t_path(), mmap_mode="r")
    replay_noise = np.load(replay_noise_path(), mmap_mode="r")
    total_steps = EPOCHS * batches_per_epoch
    family_root = CTD_ROOT / args.family
    family_root.mkdir(parents=True, exist_ok=True)

    for pair_position, pair_index in enumerate(assigned_pairs, start=1):
        output_path = family_root / f"pair_{pair_index:02d}.npz"
        metadata_path = family_root / f"pair_{pair_index:02d}.json"
        if output_path.is_file() and metadata_path.is_file():
            print(f"[skip] {output_path}", flush=True)
            continue
        started = time.perf_counter()
        configure_training_precision()
        target_index = pair_index + 1
        start_model, _, start_checkpoint = build_model(paths[pair_index], "raw", device)
        target_model, _, target_checkpoint = build_model(paths[target_index], "raw", device)
        replay_model, _, _ = build_model(paths[pair_index], "raw", device)
        replay_model.train()
        optimizer = torch.optim.AdamW(
            replay_model.parameters(),
            lr=PEAK_LR,
            betas=(ADAM_B1, ADAM_B2),
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY,
        )
        optimizer.load_state_dict(start_checkpoint["optimizer_state"])
        target_model.eval()
        current_gradient_sum = {
            name: torch.zeros_like(parameter)
            for name, parameter in target_model.named_parameters()
        }
        global_step = int(start_checkpoint["global_step"])

        for event_index, epoch_batches in enumerate(orders[target_index]):
            for batch_position, indices_np in enumerate(epoch_batches, start=1):
                indices = torch.from_numpy(indices_np).to(device=device, dtype=torch.long)
                x = x_all.index_select(0, indices)
                condition = condition_all.index_select(0, indices)
                timestep = torch.from_numpy(
                    np.array(
                        replay_t[target_index, indices_np, event_index], copy=True
                    )
                ).to(device=device, dtype=torch.long)
                noise = torch.from_numpy(
                    np.array(
                        replay_noise[target_index, indices_np, event_index], copy=True
                    )
                ).to(device=device, dtype=x.dtype)
                xt = base.q_sample(x, timestep, noise, schedule)

                target_model.zero_grad(set_to_none=True)
                target_loss = F.mse_loss(
                    target_model(xt, timestep, condition), noise
                )
                target_loss.backward()
                count = len(indices_np)
                for name, parameter in target_model.named_parameters():
                    current_gradient_sum[name].add_(parameter.grad, alpha=count)

                learning_rate = lr_at(global_step, total_steps)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                optimizer.zero_grad(set_to_none=True)
                replay_loss = F.mse_loss(
                    replay_model(xt, timestep, condition), noise
                )
                replay_loss.backward()
                torch.nn.utils.clip_grad_norm_(replay_model.parameters(), GRAD_CLIP)
                optimizer.step()
                global_step += 1
            print(
                f"[ctd gpu={args.gpu} {args.family}] pair={pair_index:02d} "
                f"event={event_index + 1}/{CTD_EVENTS_PER_INTERVAL}",
                flush=True,
            )

        target_state = {
            name: value.detach().clone()
            for name, value in target_model.named_parameters()
        }
        replay_state = {
            name: value.detach().clone()
            for name, value in replay_model.named_parameters()
        }
        # The Bundle implementation averages four per-point event gradients and
        # multiplies by one checkpoint LR. Multiplication by 4/B converts that
        # direction to an interval-scale SGD approximation; this constant does
        # not affect Bundle LDS ranks but makes magnitude diagnostics readable.
        checkpoint_lr = float(target_checkpoint["learning_rate_at_checkpoint"])
        current_raw_tangent = {
            name: (-checkpoint_lr / float(CTD_EVENTS_PER_INTERVAL)) * gradient
            for name, gradient in current_gradient_sum.items()
        }
        current_scaled_tangent = {
            name: (float(CTD_EVENTS_PER_INTERVAL) / float(BATCH_SIZE)) * value
            for name, value in current_raw_tangent.items()
        }
        arrays, replay_agreement = evaluate_responses(
            start_model,
            target_state,
            replay_state,
            current_raw_tangent,
            current_scaled_tangent,
            bank,
            device,
            args.query_batch_size,
        )
        np.savez_compressed(
            output_path,
            query_ids=bank["query_ids"],
            timestamp_positions=bank["timestamp_positions"],
            **arrays,
        )
        with open(metadata_path, "w") as handle:
            json.dump(
                {
                    "family": args.family,
                    "pair_index": pair_index,
                    "start_epoch": int(start_checkpoint["epoch"]),
                    "target_epoch": int(target_checkpoint["epoch"]),
                    "start_global_step": int(start_checkpoint["global_step"]),
                    "target_global_step": int(target_checkpoint["global_step"]),
                    "replayed_global_step": global_step,
                    "checkpoint_learning_rate": checkpoint_lr,
                    "replayed_adamw_parameter_agreement": replay_agreement,
                    "query_ids": sorted(set(bank["query_ids"].tolist())),
                    "methods": list(CTD_METHODS),
                },
                handle,
                indent=2,
            )
        print(
            f"[done] pair={pair_index:02d} replay-cos="
            f"{replay_agreement['cosine']:+.8f} replay-relerr="
            f"{replay_agreement['relative_error']:.3e} elapsed="
            f"{(time.perf_counter() - started) / 60:.1f}m",
            flush=True,
        )
        del start_model, target_model, replay_model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
