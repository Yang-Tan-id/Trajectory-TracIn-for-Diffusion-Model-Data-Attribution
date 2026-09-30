"""Compare local and path-integrated JVPs with fixed-point finite changes."""

import argparse
import importlib
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from null_same_direction_learning_config import nsdl_checkpoint_path
from path_integrated_jvp_config import *


shared = importlib.import_module("148_run_null_same_direction_learning_worker")
magnitude = importlib.import_module("173_run_opposite_gradient_magnitude_worker")


def quadrature(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return tuple((float((node + 1.0) / 2.0), float(weight / 2.0)) for node, weight in zip(nodes, weights))


GAUSS2 = quadrature(2)
GAUSS4 = quadrature(4)


def configure_high_precision():
    """Make tiny finite parameter differences numerically meaningful."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")

    # The training helper hard-codes float32 in its sinusoidal embedding.  This
    # dtype-aware replacement lets the otherwise identical model run in float64.
    def float64_time_embedding(t, dim):
        half = dim // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(0, half, device=t.device, dtype=torch.float64)
            / (half - 1)
        )
        arguments = t.to(torch.float64).unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat(
            (torch.sin(arguments), torch.cos(arguments)), dim=1
        )
        if dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    base.sinusoidal_time_embedding = float64_time_embedding


def schedule_to_float64(schedule):
    for name in (
        "betas",
        "alphas",
        "alpha_bars",
        "sqrt_alpha_bars",
        "sqrt_one_minus_alpha_bars",
    ):
        setattr(schedule, name, getattr(schedule, name).double())
    return schedule


def point_metrics(predicted, actual):
    predicted = predicted.detach().double().flatten(1)
    actual = actual.detach().double().flatten(1)
    predicted_l2 = predicted.norm(dim=1)
    actual_l2 = actual.norm(dim=1)
    denominator = (predicted_l2 * actual_l2).clamp_min(NSDL_EPS)
    cosine = (predicted * actual).sum(dim=1) / denominator
    vector_relative_error = (predicted - actual).norm(dim=1) / actual_l2.clamp_min(
        NSDL_EPS
    )
    magnitude_relative_error = (predicted_l2 - actual_l2).abs() / actual_l2.clamp_min(
        NSDL_EPS
    )
    return {
        "predicted_l2": predicted_l2.cpu().numpy(),
        "vector_cosine": cosine.cpu().numpy(),
        "vector_relative_error": vector_relative_error.cpu().numpy(),
        "magnitude_relative_error": magnitude_relative_error.cpu().numpy(),
    }


def evaluate_block(
    model,
    names,
    parameters,
    tangent,
    target_x0,
    target_condition,
    directions,
    schedule,
    device,
    batch_size,
):
    direction_indices = torch.tensor(
        PIJVP_DIRECTION_INDICES, device=device, dtype=torch.long
    )
    timestamps_bank = torch.tensor(PIJVP_TIMESTAMPS, device=device, dtype=torch.long)
    chosen_directions = directions[direction_indices]
    total = PIJVP_DIRECTION_COUNT * PIJVP_TIMESTAMP_COUNT
    arrays = {
        "actual_l2": np.empty(total, dtype=np.float64),
    }
    for method in PIJVP_METHODS:
        for metric in (
            "predicted_l2",
            "vector_cosine",
            "vector_relative_error",
            "magnitude_relative_error",
        ):
            arrays[f"{method}_{metric}"] = np.empty(total, dtype=np.float64)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        flat = torch.arange(start, end, device=device)
        direction_position = torch.div(
            flat, PIJVP_TIMESTAMP_COUNT, rounding_mode="floor"
        )
        timestamp_position = flat.remainder(PIJVP_TIMESTAMP_COUNT)
        timestamps = timestamps_bank[timestamp_position]
        noise = chosen_directions[direction_position]
        x0 = target_x0.expand(end - start, -1, -1, -1)
        condition = target_condition.expand(end - start, -1)
        xt = base.q_sample(x0, timestamps, noise, schedule)

        def prediction_fn(*parameter_values):
            return functional_call(
                model,
                dict(zip(names, parameter_values)),
                (xt, timestamps, condition),
            )

        derivative_cache = {}

        def derivative_at(position):
            key = float(position)
            if key not in derivative_cache:
                path_parameters = tuple(
                    parameter + key * delta
                    for parameter, delta in zip(parameters, tangent)
                )
                _, derivative = jvp(prediction_fn, path_parameters, tangent)
                derivative_cache[key] = derivative
            return derivative_cache[key]

        start_derivative = derivative_at(0.0)

        def first_directional(*parameter_values):
            return jvp(prediction_fn, parameter_values, tangent)[1]

        _, second_derivative = jvp(first_directional, parameters, tangent)
        predictions = {
            "start_jvp": start_derivative,
            "second_order_taylor": start_derivative + 0.5 * second_derivative,
            "endpoint_trapezoid": 0.5
            * (start_derivative + derivative_at(1.0)),
            "gauss2_path_jvp": sum(
                weight * derivative_at(position) for position, weight in GAUSS2
            ),
            "gauss4_path_jvp": sum(
                weight * derivative_at(position) for position, weight in GAUSS4
            ),
        }
        with torch.no_grad():
            before = prediction_fn(*parameters)
            endpoint_parameters = tuple(
                parameter + delta for parameter, delta in zip(parameters, tangent)
            )
            after = prediction_fn(*endpoint_parameters)
            actual = after - before
        actual_l2 = actual.detach().double().flatten(1).norm(dim=1).cpu().numpy()
        arrays["actual_l2"][start:end] = actual_l2
        for method, prediction in predictions.items():
            metrics = point_metrics(prediction, actual)
            for metric, values in metrics.items():
                arrays[f"{method}_{metric}"][start:end] = values
        print(
            f"[gpu {device.index}] fixed-points={end}/{total}",
            flush=True,
        )
        del derivative_cache, predictions, second_derivative, actual, before, after

    shape = (PIJVP_DIRECTION_COUNT, PIJVP_TIMESTAMP_COUNT)
    return {name: value.reshape(shape) for name, value in arrays.items()}


def run_source(source_index, dataset, checkpoint, schedule, device, batch_size):
    target_index = ntcd_target_index(source_index)
    output_dir = pijvp_source_dir(source_index)
    done_path = output_dir / "done.json"
    if done_path.is_file():
        print(
            f"[gpu {device.index}] skip source={source_index} target={target_index}",
            flush=True,
        )
        return
    target_image, target_condition = dataset[int(target_index)]
    target_x0 = target_image.unsqueeze(0).to(device=device, dtype=torch.float64)
    target_condition = target_condition.unsqueeze(0).to(
        device=device, dtype=torch.float64
    )
    baseline = shared.make_model(dataset, checkpoint["model_state"], device).double()
    names = tuple(name for name, _ in baseline.named_parameters())
    parameters = tuple(parameter.detach() for parameter in baseline.parameters())
    edmc_dir = edmc_source_dir(source_index)
    with np.load(edmc_dir / "block_0_responses.npz", allow_pickle=False) as archive:
        directions = torch.from_numpy(archive["directions"]).to(
            device=device, dtype=torch.float64
        )
    fresh_dir = ntcd_source_dir(source_index, "fresh_sgd")
    with open(fresh_dir / "result.json") as handle:
        fresh_result = json.load(handle)
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = []
    started = time.perf_counter()
    for block_index, timestamp_block in enumerate(NTCD_TIMESTAMP_BLOCKS):
        block_path = output_dir / f"block_{block_index}_path_jvp.npz"
        if block_path.is_file():
            print(
                f"[gpu {device.index}] source={source_index} skip block={block_index}",
                flush=True,
            )
            blocks.append(str(block_path))
            continue
        payload = torch.load(
            fresh_result["blocks"][block_index]["updated_model"],
            map_location="cpu",
            weights_only=False,
        )
        tangent = magnitude.parameter_delta(baseline, payload["model_state"])
        responses = evaluate_block(
            baseline,
            names,
            parameters,
            tangent,
            target_x0,
            target_condition,
            directions,
            schedule,
            device,
            batch_size,
        )
        np.savez_compressed(
            block_path,
            direction_indices=np.asarray(PIJVP_DIRECTION_INDICES, dtype=np.int64),
            timestamps=np.asarray(PIJVP_TIMESTAMPS, dtype=np.int64),
            **responses,
        )
        blocks.append(str(block_path))
        print(
            f"[gpu {device.index}] source={source_index} block={block_index} saved",
            flush=True,
        )
        del tangent, responses, payload
        torch.cuda.empty_cache()
    shared.atomic_json(
        done_path,
        {
            "source_datapoint_index": int(source_index),
            "target_datapoint_index": int(target_index),
            "optimizer_mode": "fresh_sgd",
            "fixed_target_direction_count": PIJVP_DIRECTION_COUNT,
            "fixed_target_timestamp_count": PIJVP_TIMESTAMP_COUNT,
            "methods": list(PIJVP_METHODS),
            "precision_mode": PIJVP_PRECISION_MODE,
            "blocks": blocks,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=PIJVP_DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_high_precision()
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    checkpoint = torch.load(
        nsdl_checkpoint_path(), map_location="cpu", weights_only=False
    )
    schedule = schedule_to_float64(base.make_linear_schedule(T, device=device))
    for source_index in nsdl_datapoint_indices()[
        args.shard_index :: args.shard_count
    ]:
        run_source(
            source_index, dataset, checkpoint, schedule, device, args.batch_size
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
