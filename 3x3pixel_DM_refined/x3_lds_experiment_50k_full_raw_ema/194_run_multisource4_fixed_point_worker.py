"""Combine four distinct datapoint/direction updates in output space."""

import argparse
import importlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from null_same_direction_learning_config import nsdl_checkpoint_path
from multisource4_fixed_point_config import *


shared = importlib.import_module("148_run_null_same_direction_learning_worker")
path_worker = importlib.import_module("188_run_path_integrated_jvp_worker")
source_worker = importlib.import_module("190_run_source_gradient_fixed_point_worker")
sequential_worker = importlib.import_module("192_run_sequential4_fixed_point_worker")
endpoint_worker = importlib.import_module("179_run_endpoint_direction_mc_worker")


def replay_multisource_updates(sequence_index, dataset, checkpoint, device):
    source_worker.configure_training_precision()
    source_indices = ms4_source_indices(sequence_index)
    schedule = base.make_linear_schedule(T, device=device)
    model = shared.make_model(dataset, checkpoint["model_state"], device)
    learning_rate = shared.checkpoint_learning_rate(checkpoint)
    optimizer, clip_norm = shared.make_optimizer(model, checkpoint, learning_rate)
    timestamp_bank = torch.tensor(NSDL_TIMESTAMPS, device=device, dtype=torch.long)
    states = [sequential_worker.parameter_state(model)]
    updates = []
    for update_index, source_index in enumerate(source_indices):
        image, condition = dataset[int(source_index)]
        x0 = image.unsqueeze(0).to(device=device, dtype=torch.float32)
        condition = condition.unsqueeze(0).to(device=device, dtype=torch.float32)
        direction = shared.fixed_direction(source_index, device)
        start = update_index * NSDL_UPDATE_BATCH_SIZE
        end = start + NSDL_UPDATE_BATCH_SIZE
        timestamps = timestamp_bank[start:end]
        noise = direction.unsqueeze(0).expand(len(timestamps), -1, -1, -1)
        xt = base.q_sample(
            x0.expand(len(timestamps), -1, -1, -1),
            timestamps,
            noise,
            schedule,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            xt, timestamps, condition.expand(len(timestamps), -1)
        )
        loss = F.mse_loss(prediction, noise)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip_norm
        )
        optimizer.step()
        states.append(sequential_worker.parameter_state(model))
        updates.append(
            {
                "update": update_index + 1,
                "source_datapoint_index": int(source_index),
                "noise_direction_seed": NSDL_DIRECTION_SEED_BASE + int(source_index),
                "timestamp_start": int(timestamps[0]),
                "timestamp_end": int(timestamps[-1]),
                "loss": float(loss.detach()),
                "gradient_norm_before_clip": float(gradient_norm),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
    del model, optimizer
    torch.cuda.empty_cache()
    return states, updates


def evaluate_fixed_points(
    model,
    names,
    states,
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
    timestamp_bank = torch.tensor(PIJVP_TIMESTAMPS, device=device, dtype=torch.long)
    selected_directions = directions[direction_indices]
    total = PIJVP_DIRECTION_COUNT * PIJVP_TIMESTAMP_COUNT
    arrays = {
        "actual_l2": np.empty(total, dtype=np.float64),
        "gauss2_termwise_squared": np.empty(total, dtype=np.float64),
        "gauss2_vector_sum_squared": np.empty(total, dtype=np.float64),
        "gauss2_cross_term_fraction": np.empty(total, dtype=np.float64),
    }
    for step_index in range(4):
        arrays[f"gauss2_step_{step_index}_l2"] = np.empty(total, dtype=np.float64)
    for method in MS4_METHODS:
        for metric in (
            "predicted_l2",
            "vector_cosine",
            "vector_relative_error",
            "magnitude_relative_error",
        ):
            arrays[f"{method}_{metric}"] = np.empty(total, dtype=np.float64)

    step_tangents = [
        sequential_worker.subtract_states(states[index + 1], states[index])
        for index in range(4)
    ]
    total_tangent = sequential_worker.subtract_states(states[-1], states[0])
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        flat = torch.arange(start, end, device=device)
        direction_position = torch.div(
            flat, PIJVP_TIMESTAMP_COUNT, rounding_mode="floor"
        )
        timestamp_position = flat.remainder(PIJVP_TIMESTAMP_COUNT)
        timestamps = timestamp_bank[timestamp_position]
        noise = selected_directions[direction_position]
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
            actual = prediction_fn(*states[-1]) - prediction_fn(*states[0])

        total_start = directional(states[0], total_tangent)
        stepwise_start = 0.0
        stepwise_gauss2 = 0.0
        gauss2_steps = []
        for state, tangent in zip(states[:-1], step_tangents):
            stepwise_start = stepwise_start + directional(state, tangent)
            step_response = 0.0
            for position, weight in path_worker.GAUSS2:
                step_response = step_response + weight * directional(
                    sequential_worker.add_scaled(state, tangent, position), tangent
                )
            gauss2_steps.append(step_response)
            stepwise_gauss2 = stepwise_gauss2 + step_response

        predictions = {
            "total_start_jvp": total_start,
            "stepwise_start_vector_sum": stepwise_start,
            "stepwise_gauss2_vector_sum": stepwise_gauss2,
        }
        actual_flat = actual.detach().double().flatten(1)
        actual_l2 = actual_flat.norm(dim=1)
        arrays["actual_l2"][start:end] = actual_l2.cpu().numpy()
        termwise_squared = torch.zeros_like(actual_l2)
        for step_index, step_response in enumerate(gauss2_steps):
            step_l2 = step_response.detach().double().flatten(1).norm(dim=1)
            arrays[f"gauss2_step_{step_index}_l2"][start:end] = step_l2.cpu().numpy()
            termwise_squared += step_l2.square()
        vector_sum_squared = (
            stepwise_gauss2.detach().double().flatten(1).square().sum(dim=1)
        )
        arrays["gauss2_termwise_squared"][start:end] = termwise_squared.cpu().numpy()
        arrays["gauss2_vector_sum_squared"][start:end] = vector_sum_squared.cpu().numpy()
        arrays["gauss2_cross_term_fraction"][start:end] = (
            (vector_sum_squared - termwise_squared)
            / vector_sum_squared.clamp_min(NSDL_EPS)
        ).cpu().numpy()
        for method, prediction in predictions.items():
            metrics = path_worker.point_metrics(prediction, actual)
            for metric, values in metrics.items():
                arrays[f"{method}_{metric}"][start:end] = values
        print(f"[gpu {device.index}] fixed-points={end}/{total}", flush=True)
        del actual, predictions, gauss2_steps
    shape = (PIJVP_DIRECTION_COUNT, PIJVP_TIMESTAMP_COUNT)
    return {name: value.reshape(shape) for name, value in arrays.items()}


def run_sequence(sequence_index, dataset, checkpoint, device, batch_size):
    source_indices = ms4_source_indices(sequence_index)
    target_index = ms4_target_index(sequence_index)
    output_dir = ms4_sequence_dir(sequence_index)
    done_path = output_dir / "done.json"
    if done_path.is_file():
        print(f"[gpu {device.index}] skip sequence={sequence_index}", flush=True)
        return
    cpu_states, updates = replay_multisource_updates(
        sequence_index, dataset, checkpoint, device
    )
    path_worker.configure_high_precision()
    schedule = path_worker.schedule_to_float64(
        base.make_linear_schedule(T, device=device)
    )
    model = shared.make_model(dataset, checkpoint["model_state"], device).double()
    names = tuple(name for name, _ in model.named_parameters())
    states = [
        sequential_worker.state_tuple(state, names, device) for state in cpu_states
    ]
    target_image, target_condition = dataset[int(target_index)]
    target_x0 = target_image.unsqueeze(0).to(device=device, dtype=torch.float64)
    target_condition = target_condition.unsqueeze(0).to(
        device=device, dtype=torch.float64
    )
    directions = endpoint_worker.endpoint_directions(
        source_indices[0], target_index, device
    ).to(dtype=torch.float64)
    started = time.perf_counter()
    responses = evaluate_fixed_points(
        model,
        names,
        states,
        target_x0,
        target_condition,
        directions,
        schedule,
        device,
        batch_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / "multisource4_fixed_point_responses.npz"
    np.savez_compressed(
        response_path,
        direction_indices=np.asarray(PIJVP_DIRECTION_INDICES, dtype=np.int64),
        timestamps=np.asarray(PIJVP_TIMESTAMPS, dtype=np.int64),
        **responses,
    )
    shared.atomic_json(
        done_path,
        {
            "sequence_index": int(sequence_index),
            "source_datapoint_indices": [int(value) for value in source_indices],
            "target_datapoint_index": int(target_index),
            "optimizer": "restored AdamW",
            "one_distinct_datapoint_and_noise_direction_per_update": True,
            "updates": updates,
            "methods": list(MS4_METHODS),
            "responses": str(response_path),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    print(
        f"[gpu {device.index}] sequence={sequence_index} "
        f"sources={source_indices} target={target_index} saved",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=MS4_DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    checkpoint = torch.load(
        nsdl_checkpoint_path(), map_location="cpu", weights_only=False
    )
    for sequence_index in range(args.shard_index, MS4_SEQUENCE_COUNT, args.shard_count):
        run_sequence(sequence_index, dataset, checkpoint, device, args.batch_size)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
