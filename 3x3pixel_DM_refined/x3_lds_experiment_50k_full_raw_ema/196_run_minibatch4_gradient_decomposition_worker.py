"""Validate a one-minibatch update as a sum of per-datapoint loss gradients."""

import argparse
import importlib
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from minibatch4_gradient_decomposition_config import *
from null_same_direction_learning_config import nsdl_checkpoint_path


shared = importlib.import_module("148_run_null_same_direction_learning_worker")
ORIGINAL_TIME_EMBEDDING = base.sinusoidal_time_embedding


def configure_training_precision():
    base.sinusoidal_time_embedding = ORIGINAL_TIME_EMBEDDING
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def configure_high_precision():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")

    def float64_time_embedding(timestamps, dimension):
        half = dimension // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(
                0, half, device=timestamps.device, dtype=torch.float64
            )
            / (half - 1)
        )
        arguments = timestamps.to(torch.float64).unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((torch.sin(arguments), torch.cos(arguments)), dim=1)
        if dimension % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    base.sinusoidal_time_embedding = float64_time_embedding


def schedule_to_float64(schedule):
    for name in (
        "betas", "alphas", "alpha_bars", "sqrt_alpha_bars",
        "sqrt_one_minus_alpha_bars",
    ):
        setattr(schedule, name, getattr(schedule, name).double())
    return schedule


def parameter_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
    }


def state_tuple(state, names, device):
    return tuple(state[name].to(device=device, dtype=torch.float64) for name in names)


def tangent_agreement(predicted, actual):
    predicted_flat = torch.cat([value.double().reshape(-1) for value in predicted])
    actual_flat = torch.cat([value.double().reshape(-1) for value in actual])
    denominator = (predicted_flat.norm() * actual_flat.norm()).clamp_min(NSDL_EPS)
    return {
        "cosine": float(torch.dot(predicted_flat, actual_flat) / denominator),
        "relative_l2_error": float(
            (predicted_flat - actual_flat).norm()
            / actual_flat.norm().clamp_min(NSDL_EPS)
        ),
        "predicted_norm": float(predicted_flat.norm()),
        "actual_norm": float(actual_flat.norm()),
    }


def point_metrics(predicted, actual):
    predicted = predicted.detach().double().flatten(1)
    actual = actual.detach().double().flatten(1)
    predicted_l2 = predicted.norm(dim=1)
    actual_l2 = actual.norm(dim=1)
    return {
        "predicted_l2": predicted_l2.cpu().numpy(),
        "vector_cosine": (
            (predicted * actual).sum(dim=1)
            / (predicted_l2 * actual_l2).clamp_min(NSDL_EPS)
        ).cpu().numpy(),
        "vector_relative_error": (
            (predicted - actual).norm(dim=1) / actual_l2.clamp_min(NSDL_EPS)
        ).cpu().numpy(),
        "magnitude_relative_error": (
            (predicted_l2 - actual_l2).abs() / actual_l2.clamp_min(NSDL_EPS)
        ).cpu().numpy(),
    }


def endpoint_directions(source_index, target_index, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(
        EDMC_DIRECTION_SEED_BASE + int(source_index) + N_TRAIN * int(target_index)
    )
    return torch.randn(
        (EDMC_DIRECTION_COUNT, 3, 3, 3), generator=generator,
        device=device, dtype=torch.float32,
    )


def clone_tuple(values):
    return tuple(value.detach().clone() for value in values)


def tuple_add(*groups):
    return tuple(sum(values) for values in zip(*groups))


def tuple_scale(values, scale):
    return tuple(value * scale for value in values)


def tuple_norm(values):
    return torch.sqrt(sum(value.double().square().sum() for value in values))


def per_datapoint_gradients(sequence_index, model, dataset, schedule, device):
    """Return four equally weighted loss gradients at the same parameters."""
    parameters = tuple(model.parameters())
    timestamp_bank = torch.tensor(NSDL_TIMESTAMPS, device=device, dtype=torch.long)
    gradients = []
    metadata = []
    for position, source_index in enumerate(mb4_source_indices(sequence_index)):
        image, condition = dataset[int(source_index)]
        x0 = image.unsqueeze(0).to(device=device, dtype=torch.float32)
        condition = condition.unsqueeze(0).to(device=device, dtype=torch.float32)
        direction = shared.fixed_direction(source_index, device)
        start = position * MB4_TIMESTAMPS_PER_DATAPOINT
        end = start + MB4_TIMESTAMPS_PER_DATAPOINT
        timestamps = timestamp_bank[start:end]
        noise = direction.unsqueeze(0).expand(len(timestamps), -1, -1, -1)
        xt = base.q_sample(
            x0.expand(len(timestamps), -1, -1, -1), timestamps, noise, schedule
        )
        prediction = model(
            xt, timestamps, condition.expand(len(timestamps), -1)
        )
        loss = F.mse_loss(prediction, noise)
        gradient = torch.autograd.grad(loss, parameters)
        gradients.append(clone_tuple(gradient))
        metadata.append(
            {
                "datapoint_index": int(source_index),
                "noise_direction_seed": NSDL_DIRECTION_SEED_BASE + int(source_index),
                "timestamp_start": int(timestamps[0]),
                "timestamp_end": int(timestamps[-1]),
                "loss": float(loss.detach()),
                "gradient_norm": float(tuple_norm(gradient)),
            }
        )
        del prediction, loss, gradient
    return gradients, metadata


def shared_clipped_gradients(per_gradients, clip_norm):
    count = len(per_gradients)
    batch_gradient = tuple_scale(tuple_add(*per_gradients), 1.0 / count)
    norm = tuple_norm(batch_gradient)
    clip_scale = min(1.0, float(clip_norm) / (float(norm) + 1e-6))
    clipped_batch = tuple_scale(batch_gradient, clip_scale)
    clipped_per_datapoint = [
        tuple_scale(gradient, clip_scale / count) for gradient in per_gradients
    ]
    return clipped_batch, clipped_per_datapoint, float(norm), float(clip_scale)


def assign_gradients(model, gradients):
    for parameter, gradient in zip(model.parameters(), gradients):
        parameter.grad = gradient.detach().clone()


def one_sgd_update(model, clipped_batch, clipped_per_datapoint, learning_rate):
    before = parameter_state(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    assign_gradients(model, clipped_batch)
    optimizer.step()
    after = parameter_state(model)
    names = tuple(name for name, _ in model.named_parameters())
    device = next(model.parameters()).device
    exact = tuple(
        after[name].to(device) - before[name].to(device)
        for name in names
    )
    contributions = [
        tuple_scale(gradient, -learning_rate)
        for gradient in clipped_per_datapoint
    ]
    reconstructed = tuple_add(*contributions)
    agreement = tangent_agreement(reconstructed, exact)
    del optimizer
    return before, after, contributions, exact, agreement


def one_restored_adamw_update(
    model,
    checkpoint,
    clipped_batch,
    clipped_per_datapoint,
    learning_rate,
):
    optimizer, _ = shared.make_optimizer(model, checkpoint, learning_rate)
    parameters = tuple(model.parameters())
    names = tuple(name for name, _ in model.named_parameters())
    before = parameter_state(model)
    old_moments = []
    for parameter in parameters:
        state = optimizer.state[parameter]
        old_moments.append(state["exp_avg"].detach().clone())

    optimizer.zero_grad(set_to_none=True)
    assign_gradients(model, clipped_batch)
    optimizer.step()
    after = parameter_state(model)
    exact = tuple(
        after[name].to(device=parameter.device) - before[name].to(device=parameter.device)
        for name, parameter in zip(names, parameters)
    )

    group = optimizer.param_groups[0]
    beta1, beta2 = group["betas"]
    eps = float(group["eps"])
    weight_decay = float(group["weight_decay"])
    baseline = []
    denominators = []
    step_sizes = []
    for parameter, old_moment in zip(parameters, old_moments):
        state = optimizer.state[parameter]
        step = float(state["step"].item())
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step
        denominator = state["exp_avg_sq"].detach().sqrt() / math.sqrt(
            bias_correction2
        )
        denominator = denominator + eps
        step_size = learning_rate / bias_correction1
        parameter_before = parameter.detach() - exact[len(baseline)]
        baseline.append(
            -learning_rate * weight_decay * parameter_before
            -step_size * beta1 * old_moment / denominator
        )
        denominators.append(denominator)
        step_sizes.append(step_size)

    contributions = []
    for gradient_group in clipped_per_datapoint:
        contributions.append(
            tuple(
                -step_size * (1.0 - beta1) * gradient / denominator
                for gradient, denominator, step_size in zip(
                    gradient_group, denominators, step_sizes
                )
            )
        )
    reconstructed = tuple_add(tuple(baseline), *contributions)
    agreement = tangent_agreement(reconstructed, exact)
    data_only = tuple_add(*contributions)
    baseline_fraction = float(tuple_norm(baseline) / tuple_norm(exact).clamp_min(NSDL_EPS))
    del optimizer
    return (
        before,
        after,
        contributions,
        tuple(baseline),
        data_only,
        exact,
        agreement,
        baseline_fraction,
    )


def build_updates(sequence_index, dataset, checkpoint, device):
    configure_training_precision()
    schedule = base.make_linear_schedule(T, device=device)
    gradient_model = shared.make_model(dataset, checkpoint["model_state"], device)
    per_gradients, datapoints = per_datapoint_gradients(
        sequence_index, gradient_model, dataset, schedule, device
    )
    clip_norm = float(checkpoint.get("config", {}).get("grad_clip_norm", GRAD_CLIP))
    clipped_batch, clipped_per, batch_norm, clip_scale = shared_clipped_gradients(
        per_gradients, clip_norm
    )
    learning_rate = shared.checkpoint_learning_rate(checkpoint)

    sgd_model = shared.make_model(dataset, checkpoint["model_state"], device)
    sgd = one_sgd_update(
        sgd_model, clipped_batch, clipped_per, learning_rate
    )
    adam_model = shared.make_model(dataset, checkpoint["model_state"], device)
    adam = one_restored_adamw_update(
        adam_model, checkpoint, clipped_batch, clipped_per, learning_rate
    )
    del gradient_model, sgd_model, adam_model, per_gradients
    torch.cuda.empty_cache()
    return {
        "datapoints": datapoints,
        "learning_rate": learning_rate,
        "batch_gradient_norm_before_clip": batch_norm,
        "clip_scale": clip_scale,
        "sgd": sgd,
        "adam": adam,
    }


def tangent_double(tangent, device):
    return tuple(value.detach().to(device=device, dtype=torch.float64) for value in tangent)


def evaluate_fixed_points(
    model,
    names,
    initial,
    sgd_final,
    adam_final,
    sgd_contributions,
    adam_contributions,
    adam_baseline,
    adam_exact,
    target_x0,
    target_condition,
    directions,
    schedule,
    device,
    batch_size,
):
    direction_indices = torch.tensor(PIJVP_DIRECTION_INDICES, device=device)
    timestamp_bank = torch.tensor(PIJVP_TIMESTAMPS, device=device)
    selected_directions = directions[direction_indices]
    total = PIJVP_DIRECTION_COUNT * PIJVP_TIMESTAMP_COUNT
    arrays = {
        "sgd_actual_l2": np.empty(total, dtype=np.float64),
        "adam_actual_l2": np.empty(total, dtype=np.float64),
        "sgd_termwise_squared": np.empty(total, dtype=np.float64),
        "sgd_vector_sum_squared": np.empty(total, dtype=np.float64),
        "adam_data_termwise_squared": np.empty(total, dtype=np.float64),
        "adam_data_vector_sum_squared": np.empty(total, dtype=np.float64),
    }
    for method in MB4_METHODS:
        for metric in (
            "predicted_l2",
            "vector_cosine",
            "vector_relative_error",
            "magnitude_relative_error",
        ):
            arrays[f"{method}_{metric}"] = np.empty(total, dtype=np.float64)

    sgd_sum = tuple_add(*sgd_contributions)
    adam_data_sum = tuple_add(*adam_contributions)
    adam_full_sum = tuple_add(adam_baseline, adam_data_sum)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        flat = torch.arange(start, end, device=device)
        direction_position = torch.div(flat, PIJVP_TIMESTAMP_COUNT, rounding_mode="floor")
        timestamp_position = flat.remainder(PIJVP_TIMESTAMP_COUNT)
        timestamps = timestamp_bank[timestamp_position]
        noise = selected_directions[direction_position]
        xt = base.q_sample(
            target_x0.expand(end - start, -1, -1, -1), timestamps, noise, schedule
        )
        condition = target_condition.expand(end - start, -1)

        def prediction_fn(*parameter_values):
            return functional_call(
                model, dict(zip(names, parameter_values)), (xt, timestamps, condition)
            )

        def directional(tangent):
            return jvp(prediction_fn, initial, tangent)[1]

        with torch.no_grad():
            sgd_actual = prediction_fn(*sgd_final) - prediction_fn(*initial)
            adam_actual = prediction_fn(*adam_final) - prediction_fn(*initial)
        predictions = {
            "sgd_per_datapoint_vector_sum": directional(sgd_sum),
            "adam_data_only_vector_sum": directional(adam_data_sum),
            "adam_baseline_plus_data_vector_sum": directional(adam_full_sum),
            "adam_exact_parameter_delta_control": directional(adam_exact),
        }
        for method, prediction in predictions.items():
            actual = sgd_actual if method.startswith("sgd_") else adam_actual
            for metric, values in point_metrics(prediction, actual).items():
                arrays[f"{method}_{metric}"][start:end] = values
        arrays["sgd_actual_l2"][start:end] = sgd_actual.double().flatten(1).norm(dim=1).cpu().numpy()
        arrays["adam_actual_l2"][start:end] = adam_actual.double().flatten(1).norm(dim=1).cpu().numpy()

        for prefix, contributions in (
            ("sgd", sgd_contributions),
            ("adam_data", adam_contributions),
        ):
            responses = [directional(tangent).double().flatten(1) for tangent in contributions]
            termwise = sum(response.square().sum(dim=1) for response in responses)
            vector_sum = sum(responses).square().sum(dim=1)
            arrays[f"{prefix}_termwise_squared"][start:end] = termwise.cpu().numpy()
            arrays[f"{prefix}_vector_sum_squared"][start:end] = vector_sum.cpu().numpy()
        print(f"[gpu {device.index}] fixed-points={end}/{total}", flush=True)
    shape = (PIJVP_DIRECTION_COUNT, PIJVP_TIMESTAMP_COUNT)
    return {name: value.reshape(shape) for name, value in arrays.items()}


def run_sequence(sequence_index, dataset, checkpoint, device, batch_size):
    output_dir = mb4_sequence_dir(sequence_index)
    done_path = output_dir / "done.json"
    if done_path.is_file():
        print(f"[gpu {device.index}] skip sequence={sequence_index}", flush=True)
        return
    update = build_updates(sequence_index, dataset, checkpoint, device)
    configure_high_precision()
    schedule = schedule_to_float64(base.make_linear_schedule(T, device=device))
    model = shared.make_model(dataset, checkpoint["model_state"], device).double()
    names = tuple(name for name, _ in model.named_parameters())
    initial = state_tuple(update["sgd"][0], names, device)
    sgd_final = state_tuple(update["sgd"][1], names, device)
    adam_final = state_tuple(update["adam"][1], names, device)
    sgd_contributions = [tangent_double(value, device) for value in update["sgd"][2]]
    adam_contributions = [tangent_double(value, device) for value in update["adam"][2]]
    adam_baseline = tangent_double(update["adam"][3], device)
    adam_exact = tangent_double(update["adam"][5], device)
    target_index = mb4_target_index(sequence_index)
    target_image, target_condition = dataset[target_index]
    target_x0 = target_image.unsqueeze(0).to(device=device, dtype=torch.float64)
    target_condition = target_condition.unsqueeze(0).to(device=device, dtype=torch.float64)
    directions = endpoint_directions(
        mb4_source_indices(sequence_index)[0], target_index, device
    ).to(dtype=torch.float64)
    started = time.perf_counter()
    responses = evaluate_fixed_points(
        model, names, initial, sgd_final, adam_final, sgd_contributions,
        adam_contributions, adam_baseline, adam_exact, target_x0,
        target_condition, directions, schedule, device, batch_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / "minibatch4_gradient_decomposition.npz"
    np.savez_compressed(
        response_path,
        direction_indices=np.asarray(PIJVP_DIRECTION_INDICES, dtype=np.int64),
        timestamps=np.asarray(PIJVP_TIMESTAMPS, dtype=np.int64),
        **responses,
    )
    shared.atomic_json(
        done_path,
        {
            "sequence_index": sequence_index,
            "source_datapoint_indices": list(mb4_source_indices(sequence_index)),
            "target_datapoint_index": target_index,
            "definition": "four per-datapoint losses averaged into one minibatch and one optimizer step",
            "datapoints": update["datapoints"],
            "learning_rate": update["learning_rate"],
            "batch_gradient_norm_before_clip": update["batch_gradient_norm_before_clip"],
            "shared_clip_scale": update["clip_scale"],
            "sgd_parameter_reconstruction": update["sgd"][4],
            "adam_parameter_reconstruction": update["adam"][6],
            "adam_history_baseline_norm_over_actual": update["adam"][7],
            "responses": str(response_path),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    print(f"[gpu {device.index}] sequence={sequence_index} saved", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=MB4_DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    checkpoint = torch.load(nsdl_checkpoint_path(), map_location="cpu", weights_only=False)
    for sequence_index in range(args.shard_index, MB4_SEQUENCE_COUNT, args.shard_count):
        run_sequence(sequence_index, dataset, checkpoint, device, args.batch_size)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
