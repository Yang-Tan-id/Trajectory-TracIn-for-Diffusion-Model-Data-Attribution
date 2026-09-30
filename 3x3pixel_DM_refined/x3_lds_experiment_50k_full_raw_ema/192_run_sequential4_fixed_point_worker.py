"""Validate fixed-point response prediction through four sequential updates."""

import argparse
import importlib
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from null_same_direction_learning_config import nsdl_checkpoint_path
from sequential4_fixed_point_config import *


shared = importlib.import_module("148_run_null_same_direction_learning_worker")
magnitude = importlib.import_module("173_run_opposite_gradient_magnitude_worker")
path_worker = importlib.import_module("188_run_path_integrated_jvp_worker")
source_worker = importlib.import_module("190_run_source_gradient_fixed_point_worker")


def parameter_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
    }


def replay_four_updates(source_index, dataset, checkpoint, device):
    source_worker.configure_training_precision()
    image, condition = dataset[int(source_index)]
    x0 = image.unsqueeze(0).to(device=device, dtype=torch.float32)
    condition = condition.unsqueeze(0).to(device=device, dtype=torch.float32)
    direction = shared.fixed_direction(source_index, device)
    schedule = base.make_linear_schedule(T, device=device)
    model = shared.make_model(dataset, checkpoint["model_state"], device)
    learning_rate = shared.checkpoint_learning_rate(checkpoint)
    optimizer, clip_norm = shared.make_optimizer(
        model, checkpoint, learning_rate
    )
    timestamps = torch.tensor(NSDL_TIMESTAMPS, device=device, dtype=torch.long)
    states = [parameter_state(model)]
    updates = []
    for update_index in range(NSDL_UPDATE_COUNT):
        start = update_index * NSDL_UPDATE_BATCH_SIZE
        end = start + NSDL_UPDATE_BATCH_SIZE
        t = timestamps[start:end]
        noise = direction.unsqueeze(0).expand(len(t), -1, -1, -1)
        xt = base.q_sample(x0.expand(len(t), -1, -1, -1), t, noise, schedule)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(xt, t, condition.expand(len(t), -1))
        loss = F.mse_loss(prediction, noise)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip_norm
        )
        optimizer.step()
        states.append(parameter_state(model))
        updates.append(
            {
                "update": update_index + 1,
                "timestamp_start": int(t[0]),
                "timestamp_end": int(t[-1]),
                "loss": float(loss.detach()),
                "gradient_norm_before_clip": float(gradient_norm),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
    del model, optimizer
    torch.cuda.empty_cache()
    return states, updates


def state_tuple(state, names, device):
    return tuple(
        state[name].to(device=device, dtype=torch.float64) for name in names
    )


def subtract_states(after, before):
    return tuple(right - left for left, right in zip(before, after))


def add_scaled(state, tangent, scale):
    return tuple(
        parameter + scale * delta for parameter, delta in zip(state, tangent)
    )


def tangent_agreement_double(predicted, actual):
    predicted_flat = torch.cat([value.reshape(-1) for value in predicted])
    actual_flat = torch.cat([value.reshape(-1) for value in actual])
    cosine = torch.dot(predicted_flat, actual_flat) / (
        predicted_flat.norm() * actual_flat.norm()
    ).clamp_min(NSDL_EPS)
    relative_error = (predicted_flat - actual_flat).norm() / actual_flat.norm().clamp_min(
        NSDL_EPS
    )
    return {
        "cosine": float(cosine),
        "relative_l2_error": float(relative_error),
        "predicted_norm": float(predicted_flat.norm()),
        "actual_norm": float(actual_flat.norm()),
    }


def evaluate_fixed_points(
    model,
    names,
    replay_states,
    saved_final,
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
    arrays = {"actual_l2": np.empty(total, dtype=np.float64)}
    for method in SEQ4_METHODS:
        for metric in (
            "predicted_l2",
            "vector_cosine",
            "vector_relative_error",
            "magnitude_relative_error",
        ):
            arrays[f"{method}_{metric}"] = np.empty(total, dtype=np.float64)

    step_tangents = [
        subtract_states(replay_states[index + 1], replay_states[index])
        for index in range(NSDL_UPDATE_COUNT)
    ]
    replay_total = subtract_states(replay_states[-1], replay_states[0])
    saved_total = subtract_states(saved_final, replay_states[0])

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        flat = torch.arange(start, end, device=device)
        direction_position = torch.div(
            flat, PIJVP_TIMESTAMP_COUNT, rounding_mode="floor"
        )
        timestamp_position = flat.remainder(PIJVP_TIMESTAMP_COUNT)
        timestamps = timestamps_bank[timestamp_position]
        noise = chosen_directions[direction_position]
        xt = base.q_sample(
            target_x0.expand(end - start, -1, -1, -1),
            timestamps,
            noise,
            schedule,
        )
        condition = target_condition.expand(end - start, -1)

        def prediction_fn(*parameter_values):
            return functional_call(
                model,
                dict(zip(names, parameter_values)),
                (xt, timestamps, condition),
            )

        def directional(state, tangent):
            return jvp(prediction_fn, state, tangent)[1]

        with torch.no_grad():
            actual = prediction_fn(*saved_final) - prediction_fn(*replay_states[0])

        predictions = {
            "saved_total_start_jvp_control": directional(
                replay_states[0], saved_total
            ),
            "replayed_total_start_jvp": directional(
                replay_states[0], replay_total
            ),
        }
        stepwise_start = 0.0
        stepwise_trapezoid = 0.0
        stepwise_gauss2 = 0.0
        for state, next_state, tangent in zip(
            replay_states[:-1], replay_states[1:], step_tangents
        ):
            derivative_start = directional(state, tangent)
            derivative_end = directional(next_state, tangent)
            stepwise_start = stepwise_start + derivative_start
            stepwise_trapezoid = stepwise_trapezoid + 0.5 * (
                derivative_start + derivative_end
            )
            for position, weight in path_worker.GAUSS2:
                stepwise_gauss2 = stepwise_gauss2 + weight * directional(
                    add_scaled(state, tangent, position), tangent
                )
        predictions["replayed_stepwise_start_jvp"] = stepwise_start
        predictions["replayed_stepwise_trapezoid_jvp"] = stepwise_trapezoid
        predictions["replayed_stepwise_gauss2_jvp"] = stepwise_gauss2

        arrays["actual_l2"][start:end] = (
            actual.detach().double().flatten(1).norm(dim=1).cpu().numpy()
        )
        for method, prediction in predictions.items():
            metrics = path_worker.point_metrics(prediction, actual)
            for metric, values in metrics.items():
                arrays[f"{method}_{metric}"][start:end] = values
        print(
            f"[gpu {device.index}] fixed-points={end}/{total}", flush=True
        )
        del actual, predictions
    shape = (PIJVP_DIRECTION_COUNT, PIJVP_TIMESTAMP_COUNT)
    return {name: value.reshape(shape) for name, value in arrays.items()}


def run_source(source_index, dataset, checkpoint, device, batch_size):
    target_index = ntcd_target_index(source_index)
    output_dir = seq4_source_dir(source_index)
    done_path = output_dir / "done.json"
    if done_path.is_file():
        print(
            f"[gpu {device.index}] skip source={source_index} target={target_index}",
            flush=True,
        )
        return

    replay_cpu_states, updates = replay_four_updates(
        source_index, dataset, checkpoint, device
    )
    saved_path = NSDL_POINT_DIR / f"i{source_index:05d}" / "updated_model.pt"
    if not saved_path.is_file():
        raise FileNotFoundError(
            f"{saved_path}; run the original four-update experiment first"
        )
    saved_payload = torch.load(saved_path, map_location="cpu", weights_only=False)

    path_worker.configure_high_precision()
    schedule = path_worker.schedule_to_float64(
        base.make_linear_schedule(T, device=device)
    )
    model = shared.make_model(dataset, checkpoint["model_state"], device).double()
    names = tuple(name for name, _ in model.named_parameters())
    replay_states = [state_tuple(state, names, device) for state in replay_cpu_states]
    saved_final = state_tuple(saved_payload["model_state"], names, device)
    replay_agreement = tangent_agreement_double(
        subtract_states(replay_states[-1], replay_states[0]),
        subtract_states(saved_final, replay_states[0]),
    )

    target_image, target_condition = dataset[int(target_index)]
    target_x0 = target_image.unsqueeze(0).to(device=device, dtype=torch.float64)
    target_condition = target_condition.unsqueeze(0).to(
        device=device, dtype=torch.float64
    )
    edmc_dir = edmc_source_dir(source_index)
    with np.load(edmc_dir / "block_0_responses.npz", allow_pickle=False) as archive:
        directions = torch.from_numpy(archive["directions"]).to(
            device=device, dtype=torch.float64
        )
    started = time.perf_counter()
    responses = evaluate_fixed_points(
        model,
        names,
        replay_states,
        saved_final,
        target_x0,
        target_condition,
        directions,
        schedule,
        device,
        batch_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / "sequential4_fixed_point_responses.npz"
    np.savez_compressed(
        response_path,
        direction_indices=np.asarray(PIJVP_DIRECTION_INDICES, dtype=np.int64),
        timestamps=np.asarray(PIJVP_TIMESTAMPS, dtype=np.int64),
        **responses,
    )
    shared.atomic_json(
        done_path,
        {
            "source_datapoint_index": int(source_index),
            "target_datapoint_index": int(target_index),
            "optimizer": "restored AdamW",
            "updates_are_sequential": True,
            "timestamps_per_update": NSDL_UPDATE_BATCH_SIZE,
            "replayed_final_vs_saved_final": replay_agreement,
            "updates": updates,
            "responses": str(response_path),
            "methods": list(SEQ4_METHODS),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    print(
        f"[gpu {device.index}] source={source_index} replay/saved "
        f"cos={replay_agreement['cosine']:+.8f} "
        f"relerr={replay_agreement['relative_l2_error']:.3e}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=SEQ4_DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    checkpoint = torch.load(
        nsdl_checkpoint_path(), map_location="cpu", weights_only=False
    )
    for source_index in nsdl_datapoint_indices()[
        args.shard_index :: args.shard_count
    ]:
        run_source(source_index, dataset, checkpoint, device, args.batch_size)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
