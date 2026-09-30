"""Create 40 one-step timestamp-block models and evaluate five target axes."""

import argparse
import importlib
import time

import numpy as np
import torch
import torch.nn.functional as F

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from null_same_direction_learning_config import nsdl_checkpoint_path
from null_tblock_cross_direction_config import *


shared = importlib.import_module("148_run_null_same_direction_learning_worker")


def target_directions(source_index, target_index, source_direction, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(
        NTCD_TARGET_DIRECTION_SEED_BASE
        + int(source_index)
        + N_TRAIN * int(target_index)
    )
    directions = torch.randn(
        (NTCD_TARGET_DIRECTION_COUNT, *source_direction.shape),
        generator=generator,
        device=device,
        dtype=source_direction.dtype,
    )
    source_norm = source_direction.norm()
    directions = directions * (
        source_norm
        / directions.flatten(1).norm(dim=1).clamp_min(NSDL_EPS)
    )[:, None, None, None]
    normalized = directions.flatten(1) / directions.flatten(1).norm(
        dim=1, keepdim=True
    ).clamp_min(NSDL_EPS)
    source_unit = source_direction.flatten() / source_norm.clamp_min(NSDL_EPS)
    return directions, normalized @ source_unit, normalized @ normalized.T


def one_block_update(
    model,
    optimizer,
    clip_norm,
    x0,
    condition,
    source_direction,
    timestamps,
    schedule,
    device,
):
    t = torch.tensor(timestamps, device=device, dtype=torch.long)
    noise = source_direction.unsqueeze(0).expand(len(t), -1, -1, -1)
    xt = base.q_sample(x0.expand(len(t), -1, -1, -1), t, noise, schedule)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(xt, t, condition.expand(len(t), -1))
    loss = F.mse_loss(prediction, noise)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    optimizer.step()
    return {
        "timestamp_start": int(timestamps[0]),
        "timestamp_end": int(timestamps[-1]),
        "batch_size": len(timestamps),
        "loss": float(loss.detach()),
        "gradient_norm_before_clip": float(gradient_norm),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


@torch.no_grad()
def prediction_delta_bank(
    baseline,
    updated,
    target_x0,
    target_condition,
    directions,
    schedule,
    device,
):
    timestamps = torch.arange(T, device=device, dtype=torch.long)
    parts = []
    for start in range(0, T, NTCD_EVAL_TIMESTAMP_BATCH):
        t = timestamps[start : start + NTCD_EVAL_TIMESTAMP_BATCH]
        width = len(t)
        direction_count = len(directions)
        flat_t = t.unsqueeze(0).expand(direction_count, -1).reshape(-1)
        flat_noise = directions[:, None].expand(
            -1, width, -1, -1, -1
        ).reshape(-1, 3, 3, 3)
        flat_x0 = target_x0.expand(direction_count * width, -1, -1, -1)
        flat_condition = target_condition.expand(direction_count * width, -1)
        xt = base.q_sample(flat_x0, flat_t, flat_noise, schedule)
        before = baseline(xt, flat_t, flat_condition)
        after = updated(xt, flat_t, flat_condition)
        parts.append(
            (after - before)
            .reshape(direction_count, width, 3, 3, 3)
            .detach()
            .cpu()
        )
    return torch.cat(parts, dim=1)


def analyze_delta(delta, direction_cosine):
    direction_count, timestamp_count = delta.shape[:2]
    flat = delta.double().reshape(direction_count, timestamp_count, -1)
    l2 = flat.norm(dim=2)
    mse = flat.square().mean(dim=2)
    max_abs = flat.abs().amax(dim=2)
    denominator = (
        flat.norm(dim=2).T[:, :, None]
        * flat.norm(dim=2).T[:, None, :]
    ).clamp_min(NSDL_EPS)
    per_timestamp_cosine = (
        torch.einsum("itd,jtd->tij", flat, flat) / denominator
    )
    global_flat = flat.reshape(direction_count, -1)
    global_cosine = (
        global_flat @ global_flat.T
        / (
            global_flat.norm(dim=1)[:, None]
            * global_flat.norm(dim=1)[None, :]
        ).clamp_min(NSDL_EPS)
    )
    upper = torch.triu_indices(direction_count, direction_count, offset=1)
    direction_cosine_cpu = direction_cosine.double().cpu()
    noise_pair_cosine = direction_cosine_cpu[upper[0], upper[1]]
    delta_pair_cosine = global_cosine[upper[0], upper[1]].cpu()
    if float(noise_pair_cosine.std(unbiased=False)) > 0 and float(
        delta_pair_cosine.std(unbiased=False)
    ) > 0:
        correlation = float(
            torch.corrcoef(torch.stack((noise_pair_cosine, delta_pair_cosine)))[0, 1]
        )
    else:
        correlation = float("nan")
    off_diagonal_per_t = per_timestamp_cosine[
        :, upper[0], upper[1]
    ].cpu()
    arrays = {
        "prediction_delta": delta.numpy(),
        "delta_l2": l2.cpu().numpy(),
        "delta_mse": mse.cpu().numpy(),
        "delta_max_abs": max_abs.cpu().numpy(),
        "per_timestamp_pairwise_delta_cosine": per_timestamp_cosine.cpu().numpy(),
        "global_pairwise_delta_cosine": global_cosine.cpu().numpy(),
    }
    summary = {
        "delta_l2_mean": float(l2.mean()),
        "delta_l2_std": float(l2.std(unbiased=False)),
        "delta_rmse_mean": float(mse.sqrt().mean()),
        "delta_max_abs": float(max_abs.max()),
        "global_off_diagonal_delta_cosine_mean": float(delta_pair_cosine.mean()),
        "global_off_diagonal_delta_cosine_std": float(
            delta_pair_cosine.std(unbiased=False)
        ),
        "per_timestamp_off_diagonal_delta_cosine_mean": float(
            off_diagonal_per_t.mean()
        ),
        "per_timestamp_off_diagonal_delta_cosine_std": float(
            off_diagonal_per_t.std(unbiased=False)
        ),
        "noise_cosine_vs_delta_cosine_correlation": correlation,
        "per_direction_delta_l2_mean": l2.mean(dim=1).cpu().tolist(),
    }
    return arrays, summary


def run_source(source_index, dataset, checkpoint, schedule, device):
    target_index = ntcd_target_index(source_index)
    output_dir = ntcd_source_dir(source_index)
    result_path = output_dir / "result.json"
    if result_path.is_file():
        print(
            f"[gpu {device.index}] skip source={source_index} target={target_index}",
            flush=True,
        )
        return
    source_image, source_condition = dataset[int(source_index)]
    target_image, target_condition = dataset[int(target_index)]
    source_x0 = source_image.unsqueeze(0).to(device)
    source_condition = source_condition.unsqueeze(0).to(device)
    target_x0 = target_image.unsqueeze(0).to(device)
    target_condition = target_condition.unsqueeze(0).to(device)
    source_direction = shared.fixed_direction(source_index, device)
    directions, cosine_to_source, direction_cosine = target_directions(
        source_index, target_index, source_direction, device
    )
    baseline = shared.make_model(dataset, checkpoint["model_state"], device)
    learning_rate = shared.checkpoint_learning_rate(checkpoint)
    started = time.perf_counter()
    saved_arrays = {
        "source_direction": source_direction.detach().cpu().numpy(),
        "target_directions": directions.detach().cpu().numpy(),
        "target_direction_cosine_to_source": cosine_to_source.detach().cpu().numpy(),
        "target_direction_pairwise_cosine": direction_cosine.detach().cpu().numpy(),
        "timestamps": np.arange(T, dtype=np.int64),
    }
    block_results = []
    for block_index, timestamp_block in enumerate(NTCD_TIMESTAMP_BLOCKS):
        updated = shared.make_model(dataset, checkpoint["model_state"], device)
        optimizer, clip_norm = shared.make_optimizer(
            updated, checkpoint, learning_rate
        )
        update = one_block_update(
            updated,
            optimizer,
            clip_norm,
            source_x0,
            source_condition,
            source_direction,
            timestamp_block,
            schedule,
            device,
        )
        delta = prediction_delta_bank(
            baseline,
            updated,
            target_x0,
            target_condition,
            directions,
            schedule,
            device,
        )
        arrays, metrics = analyze_delta(delta, direction_cosine)
        for name, value in arrays.items():
            saved_arrays[f"block_{block_index}_{name}"] = value
        block_dir = output_dir / f"block_{block_index}_{timestamp_block[0]:04d}_{timestamp_block[-1]:04d}"
        shared.atomic_torch_save(
            block_dir / "updated_model.pt",
            {
                "model_state": {
                    key: value.detach().cpu()
                    for key, value in updated.state_dict().items()
                },
                "optimizer_state": optimizer.state_dict(),
                "source_datapoint_index": int(source_index),
                "target_datapoint_index": int(target_index),
                "timestamp_block_index": block_index,
                "update": update,
            },
        )
        block_results.append(
            {
                "block_index": block_index,
                "timestamp_start": timestamp_block[0],
                "timestamp_end": timestamp_block[-1],
                "update": update,
                "metrics": metrics,
                "updated_model": str(block_dir / "updated_model.pt"),
            }
        )
        print(
            f"[gpu {device.index}] source={source_index} target={target_index} "
            f"block={block_index} t={timestamp_block[0]}..{timestamp_block[-1]} "
            f"target_delta_l2={metrics['delta_l2_mean']:.6e} "
            f"cross_dir_cos={metrics['global_off_diagonal_delta_cosine_mean']:+.4f}",
            flush=True,
        )
        del updated, optimizer, delta
        torch.cuda.empty_cache()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "target_direction_prediction_deltas.npz", **saved_arrays)
    result = {
        "source_datapoint_index": int(source_index),
        "target_datapoint_index": int(target_index),
        "source_condition": source_condition[0].detach().cpu().tolist(),
        "target_condition": target_condition[0].detach().cpu().tolist(),
        "target_uses_own_prompt": True,
        "null_epoch": NSDL_NULL_EPOCH,
        "null_checkpoint": str(nsdl_checkpoint_path()),
        "learning_rate": learning_rate,
        "target_direction_count": NTCD_TARGET_DIRECTION_COUNT,
        "target_direction_cosine_to_source": cosine_to_source.detach().cpu().tolist(),
        "timestamp_blocks_are_independent_null_branches": True,
        "blocks": block_results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    shared.atomic_json(result_path, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    checkpoint = torch.load(
        nsdl_checkpoint_path(), map_location="cpu", weights_only=False
    )
    schedule = base.make_linear_schedule(T, device=device)
    source_indices = nsdl_datapoint_indices()[args.shard_index :: args.shard_count]
    print(f"[gpu {args.gpu}] sources={list(source_indices)}", flush=True)
    for source_index in source_indices:
        run_source(source_index, dataset, checkpoint, schedule, device)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
