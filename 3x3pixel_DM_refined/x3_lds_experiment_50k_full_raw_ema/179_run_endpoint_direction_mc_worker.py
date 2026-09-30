"""Evaluate finite, JVP, and loss-projection responses over endpoint directions."""

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
from endpoint_direction_mc_config import *


shared = importlib.import_module("148_run_null_same_direction_learning_worker")
magnitude = importlib.import_module("173_run_opposite_gradient_magnitude_worker")


def endpoint_directions(source_index, target_index, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(
        EDMC_DIRECTION_SEED_BASE + int(source_index) + N_TRAIN * int(target_index)
    )
    return torch.randn(
        (EDMC_DIRECTION_COUNT, 3, 3, 3),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )


def response_bank(
    baseline,
    updated,
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
    actual_sq_l2 = torch.empty(total, dtype=torch.float64)
    jvp_sq_l2 = torch.empty(total, dtype=torch.float64)
    loss_abs_derivative = torch.empty(total, dtype=torch.float64)
    actual_abs_loss_change = torch.empty(total, dtype=torch.float64)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        flat_index = torch.arange(start, end, device=device)
        direction_index = torch.div(flat_index, T, rounding_mode="floor")
        timestamps = flat_index.remainder(T).long()
        noise = directions[direction_index]
        x0 = endpoint.expand(end - start, -1, -1, -1)
        conditions = condition.expand(end - start, -1)
        xt = base.q_sample(x0, timestamps, noise, schedule)

        def prediction_fn(*parameter_values):
            return functional_call(
                baseline,
                dict(zip(names, parameter_values)),
                (xt, timestamps, conditions),
            )

        before, linear_delta = jvp(
            prediction_fn, parameters, parameter_tangent
        )
        # No higher-order derivative is needed. Detaching here prevents the CPU
        # result buffers from acquiring one long CopySlices autograd graph over
        # every evaluation batch and makes them safely convertible to NumPy.
        before = before.detach()
        linear_delta = linear_delta.detach()
        with torch.no_grad():
            after = updated(xt, timestamps, conditions)
        finite_delta = after - before
        residual = before - noise
        actual_sq_l2[start:end] = (
            finite_delta.double().flatten(1).square().sum(dim=1).cpu()
        )
        jvp_sq_l2[start:end] = (
            linear_delta.double().flatten(1).square().sum(dim=1).cpu()
        )
        # Per-example MSE directional derivative:
        # dL[Delta theta] = 2 / D * <prediction-noise, J Delta theta>.
        loss_abs_derivative[start:end] = (
            (2.0 / output_dimension)
            * (residual.double() * linear_delta.double()).flatten(1).sum(dim=1).abs().cpu()
        )
        before_loss = residual.double().flatten(1).square().mean(dim=1)
        after_loss = (after.double() - noise.double()).flatten(1).square().mean(dim=1)
        actual_abs_loss_change[start:end] = (after_loss - before_loss).abs().cpu()
        print(
            f"[gpu {device.index}] examples={end}/{total}",
            flush=True,
        )
    shape = (EDMC_DIRECTION_COUNT, T)
    return {
        "actual_sq_l2": actual_sq_l2.reshape(shape).detach().numpy(),
        "jvp_sq_l2": jvp_sq_l2.reshape(shape).detach().numpy(),
        "loss_abs_directional_derivative": loss_abs_derivative.reshape(shape).detach().numpy(),
        "actual_abs_loss_change": actual_abs_loss_change.reshape(shape).detach().numpy(),
    }


def run_source(source_index, dataset, checkpoint, schedule, device, batch_size):
    target_index = ntcd_target_index(source_index)
    output_dir = edmc_source_dir(source_index)
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
    directions = endpoint_directions(source_index, target_index, device)
    baseline = shared.make_model(dataset, checkpoint["model_state"], device)
    names = tuple(name for name, _ in baseline.named_parameters())
    parameters = tuple(baseline.parameters())
    fresh_dir = ntcd_source_dir(source_index, "fresh_sgd")
    with open(fresh_dir / "result.json") as handle:
        fresh_result = json.load(handle)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    block_metadata = []
    for block_index, timestamps in enumerate(NTCD_TIMESTAMP_BLOCKS):
        block_path = output_dir / f"block_{block_index}_responses.npz"
        if block_path.is_file():
            print(
                f"[gpu {device.index}] source={source_index} skip block={block_index}",
                flush=True,
            )
            block_metadata.append(
                {
                    "block_index": block_index,
                    "timestamp_start": timestamps[0],
                    "timestamp_end": timestamps[-1],
                    "responses": str(block_path),
                }
            )
            continue
        updated_payload = torch.load(
            fresh_result["blocks"][block_index]["updated_model"],
            map_location="cpu",
            weights_only=False,
        )
        updated = shared.make_model(dataset, updated_payload["model_state"], device)
        parameter_tangent = magnitude.parameter_delta(
            baseline, updated_payload["model_state"]
        )
        responses = response_bank(
            baseline,
            updated,
            names,
            parameters,
            parameter_tangent,
            endpoint,
            condition,
            directions,
            schedule,
            device,
            batch_size,
        )
        np.savez_compressed(
            block_path,
            directions=directions.detach().cpu().numpy(),
            timestamps=np.arange(T, dtype=np.int64),
            **responses,
        )
        block_metadata.append(
            {
                "block_index": block_index,
                "timestamp_start": timestamps[0],
                "timestamp_end": timestamps[-1],
                "responses": str(block_path),
            }
        )
        print(
            f"[gpu {device.index}] source={source_index} block={block_index} saved",
            flush=True,
        )
        del updated, parameter_tangent
        torch.cuda.empty_cache()
    shared.atomic_json(
        done_path,
        {
            "source_datapoint_index": int(source_index),
            "endpoint_datapoint_index": int(target_index),
            "endpoint_uses_own_prompt": True,
            "optimizer_mode": "fresh_sgd",
            "direction_count": EDMC_DIRECTION_COUNT,
            "noise_level_count": T,
            "batch_size": batch_size,
            "blocks": block_metadata,
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
