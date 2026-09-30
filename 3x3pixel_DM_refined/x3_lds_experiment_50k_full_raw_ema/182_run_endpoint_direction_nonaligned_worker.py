"""Compute loss projections with pollution and loss-target noises mismatched."""

import argparse
import importlib
import json
import time

import numpy as np
import torch
from torch.func import functional_call, jvp

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from null_same_direction_learning_config import nsdl_checkpoint_path
from endpoint_direction_nonaligned_config import *


shared = importlib.import_module("148_run_null_same_direction_learning_worker")
magnitude = importlib.import_module("173_run_opposite_gradient_magnitude_worker")


def nonaligned_loss_bank(
    baseline,
    names,
    parameters,
    parameter_tangent,
    endpoint,
    condition,
    directions,
    schedule,
    device,
    batch_size,
):
    total = EDMC_DIRECTION_COUNT * T
    output_dimension = 27
    values = torch.empty(total, dtype=torch.float64)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        flat_index = torch.arange(start, end, device=device)
        direction_index = torch.div(flat_index, T, rounding_mode="floor")
        timestamps = flat_index.remainder(T).long()
        pollution_noise = directions[direction_index]
        loss_direction_index = (direction_index + 1).remainder(
            EDMC_DIRECTION_COUNT
        )
        loss_target_noise = directions[loss_direction_index]
        x0 = endpoint.expand(end - start, -1, -1, -1)
        conditions = condition.expand(end - start, -1)
        xt = base.q_sample(x0, timestamps, pollution_noise, schedule)

        def prediction_fn(*parameter_values):
            return functional_call(
                baseline,
                dict(zip(names, parameter_values)),
                (xt, timestamps, conditions),
            )

        before, linear_delta = jvp(
            prediction_fn, parameters, parameter_tangent
        )
        before = before.detach()
        linear_delta = linear_delta.detach()
        residual = before - loss_target_noise
        values[start:end] = (
            (2.0 / output_dimension)
            * (residual.double() * linear_delta.double())
            .flatten(1)
            .sum(dim=1)
            .abs()
            .cpu()
        )
        print(
            f"[gpu {device.index}] examples={end}/{total}", flush=True
        )
    return values.reshape(EDMC_DIRECTION_COUNT, T).numpy()


def run_source(source_index, dataset, checkpoint, schedule, device, batch_size):
    target_index = ntcd_target_index(source_index)
    output_dir = edna_source_dir(source_index)
    done_path = output_dir / "done.json"
    if done_path.is_file():
        print(
            f"[gpu {device.index}] skip source={source_index} target={target_index}",
            flush=True,
        )
        return
    target_image, target_condition = dataset[int(target_index)]
    endpoint = target_image.unsqueeze(0).to(device)
    condition = target_condition.unsqueeze(0).to(device)
    aligned_dir = edmc_source_dir(source_index)
    with np.load(
        aligned_dir / "block_0_responses.npz", allow_pickle=False
    ) as archive:
        directions = torch.from_numpy(archive["directions"]).to(device)
    normalized = directions.double().flatten(1)
    normalized /= normalized.norm(dim=1, keepdim=True).clamp_min(NSDL_EPS)
    mismatch_cosine = (normalized * normalized.roll(-1, dims=0)).sum(dim=1)
    baseline = shared.make_model(dataset, checkpoint["model_state"], device)
    names = tuple(name for name, _ in baseline.named_parameters())
    parameters = tuple(baseline.parameters())
    fresh_dir = ntcd_source_dir(source_index, "fresh_sgd")
    with open(fresh_dir / "result.json") as handle:
        fresh_result = json.load(handle)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    blocks = []
    for block_index, timestamps in enumerate(NTCD_TIMESTAMP_BLOCKS):
        output_path = output_dir / f"block_{block_index}_nonaligned_loss.npz"
        if output_path.is_file():
            print(
                f"[gpu {device.index}] source={source_index} skip block={block_index}",
                flush=True,
            )
        else:
            updated_payload = torch.load(
                fresh_result["blocks"][block_index]["updated_model"],
                map_location="cpu",
                weights_only=False,
            )
            tangent = magnitude.parameter_delta(
                baseline, updated_payload["model_state"]
            )
            value = nonaligned_loss_bank(
                baseline,
                names,
                parameters,
                tangent,
                endpoint,
                condition,
                directions,
                schedule,
                device,
                batch_size,
            )
            np.savez_compressed(
                output_path,
                loss_abs_directional_derivative=value,
                pollution_to_loss_noise_cosine=mismatch_cosine.cpu().numpy(),
            )
            print(
                f"[gpu {device.index}] source={source_index} block={block_index} saved",
                flush=True,
            )
            del tangent
            torch.cuda.empty_cache()
        blocks.append(
            {
                "block_index": block_index,
                "timestamp_start": timestamps[0],
                "timestamp_end": timestamps[-1],
                "responses": str(output_path),
            }
        )
    shared.atomic_json(
        done_path,
        {
            "source_datapoint_index": int(source_index),
            "endpoint_datapoint_index": int(target_index),
            "loss_noise_mode": "cyclic_shift_by_one",
            "pollution_to_loss_noise_cosine_mean": float(mismatch_cosine.mean()),
            "pollution_to_loss_noise_cosine_std": float(
                mismatch_cosine.std(unbiased=False)
            ),
            "blocks": blocks,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=EDMC_DEFAULT_BATCH_SIZE)
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
    for source_index in nsdl_datapoint_indices()[
        args.shard_index :: args.shard_count
    ]:
        run_source(
            source_index, dataset, checkpoint, schedule, device, args.batch_size
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
