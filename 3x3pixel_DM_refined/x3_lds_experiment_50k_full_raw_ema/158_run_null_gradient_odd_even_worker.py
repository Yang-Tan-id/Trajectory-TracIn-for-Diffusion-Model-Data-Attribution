"""Evaluate positive, negative, even, and odd loss-gradient JVP tangents."""

import argparse
import importlib
import json
import time

import numpy as np
import torch
from torch.func import functional_call, jvp

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from null_gradient_cross_direction_config import *


shared = importlib.import_module("151_run_null_gradient_cross_direction_worker")


PREDICTORS = (
    "plus_loss_sgd_jvp",
    "minus_loss_sgd_jvp",
    "even_loss_sgd_jvp",
    "odd_loss_sgd_jvp",
)


def combine(left, right, left_scale, right_scale):
    return tuple(
        left_scale * left_value + right_scale * right_value
        for left_value, right_value in zip(left, right)
    )


def parameter_cosine(left, right):
    numerator = sum(
        (left_value.double() * right_value.double()).sum()
        for left_value, right_value in zip(left, right)
    )
    left_norm = sum(
        value.double().square().sum() for value in left
    ).sqrt()
    right_norm = sum(
        value.double().square().sum() for value in right
    ).sqrt()
    return float(numerator / (left_norm * right_norm).clamp_min(NGCD_EPS))


def evaluate_tangent_bank(
    null_model,
    next_model,
    names,
    parameters,
    tangents,
    checkpoint_tangent,
    x0,
    condition,
    direction,
    schedule,
    device,
):
    timestamps = torch.tensor(NSDL_TIMESTAMPS, device=device, dtype=torch.long)
    actual_parts = []
    predicted_parts = {name: [] for name in tangents}
    checkpoint_parts = []
    for start in range(0, T, NSDL_UPDATE_BATCH_SIZE):
        t = timestamps[start : start + NSDL_UPDATE_BATCH_SIZE]
        noise = direction.unsqueeze(0).expand(len(t), -1, -1, -1)
        xt = base.q_sample(x0.expand(len(t), -1, -1, -1), t, noise, schedule)
        expanded_condition = condition.expand(len(t), -1)

        def prediction_fn(*parameter_values):
            return functional_call(
                null_model,
                dict(zip(names, parameter_values)),
                (xt, t, expanded_condition),
            )

        null_prediction = None
        for predictor, tangent in tangents.items():
            primal, predicted_delta = jvp(prediction_fn, parameters, tangent)
            if null_prediction is None:
                null_prediction = primal
            predicted_parts[predictor].append(predicted_delta.detach().cpu())
        _, checkpoint_delta = jvp(
            prediction_fn, parameters, checkpoint_tangent
        )
        with torch.no_grad():
            actual_delta = next_model(xt, t, expanded_condition) - null_prediction
        actual_parts.append(actual_delta.detach().cpu())
        checkpoint_parts.append(checkpoint_delta.detach().cpu())
    arrays = {
        "actual_delta": torch.cat(actual_parts),
        "checkpoint_parameter_delta_jvp": torch.cat(checkpoint_parts),
    }
    arrays.update(
        {
            predictor: torch.cat(parts)
            for predictor, parts in predicted_parts.items()
        }
    )
    metrics = {
        predictor: shared.comparison_metrics(value, arrays["actual_delta"])
        for predictor, value in arrays.items()
        if predictor != "actual_delta"
    }
    return metrics, arrays


def run_datapoint(
    datapoint_index,
    dataset,
    null_checkpoint,
    next_checkpoint,
    schedule,
    device,
    evaluation_prompt,
    point_dir,
):
    output_dir = point_dir / f"i{datapoint_index:05d}"
    result_path = output_dir / "result.json"
    if result_path.is_file():
        print(f"[gpu {device.index}] skip datapoint={datapoint_index}", flush=True)
        return
    image, loss_condition = dataset[int(datapoint_index)]
    x0 = image.unsqueeze(0).to(device)
    loss_condition = loss_condition.unsqueeze(0).to(device)
    if evaluation_prompt == "random":
        evaluation_condition, prompt_source_index = (
            shared.random_evaluation_condition(
                datapoint_index, dataset, loss_condition[0], device
            )
        )
    else:
        evaluation_condition = loss_condition
        prompt_source_index = int(datapoint_index)
    direction = shared.fixed_direction(datapoint_index, device)
    null_model = shared.make_model(dataset, null_checkpoint["model_state"], device)
    next_model = shared.make_model(dataset, next_checkpoint["model_state"], device)
    names = tuple(name for name, _ in null_model.named_parameters())
    parameters = tuple(null_model.parameters())
    next_parameters = tuple(next_model.parameters())
    started = time.perf_counter()
    plus_gradient, plus_norm, plus_losses = shared.mean_positive_loss_gradient(
        null_model, x0, loss_condition, direction, schedule, device
    )
    minus_gradient, minus_norm, minus_losses = shared.mean_positive_loss_gradient(
        null_model, x0, loss_condition, -direction, schedule, device
    )
    tangents = {
        "plus_loss_sgd_jvp": tuple(-value for value in plus_gradient),
        "minus_loss_sgd_jvp": tuple(-value for value in minus_gradient),
        "even_loss_sgd_jvp": combine(
            plus_gradient, minus_gradient, -0.5, -0.5
        ),
        "odd_loss_sgd_jvp": combine(
            plus_gradient, minus_gradient, -0.5, +0.5
        ),
    }
    checkpoint_tangent = shared.parameter_delta(parameters, next_parameters)
    direction_results = {}
    saved_arrays = {"training_noise_direction": direction.detach().cpu().numpy()}
    for direction_name, evaluation_direction in (
        ("same", direction),
        ("opposite", -direction),
    ):
        metrics, arrays = evaluate_tangent_bank(
            null_model,
            next_model,
            names,
            parameters,
            tangents,
            checkpoint_tangent,
            x0,
            evaluation_condition,
            evaluation_direction,
            schedule,
            device,
        )
        direction_results[direction_name] = metrics
        for name, value in arrays.items():
            saved_arrays[f"{direction_name}_{name}"] = value.numpy()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "odd_even_direction_arrays.npz", **saved_arrays)
    result = {
        "datapoint_index": int(datapoint_index),
        "null_epoch": NGCD_NULL_EPOCH,
        "next_epoch": NGCD_NEXT_EPOCH,
        "evaluation_prompt_mode": evaluation_prompt,
        "loss_prompt_source_index": int(datapoint_index),
        "evaluation_prompt_source_index": prompt_source_index,
        "loss_condition": loss_condition[0].detach().cpu().tolist(),
        "evaluation_condition": evaluation_condition[0].detach().cpu().tolist(),
        "plus_loss_gradient_norm": plus_norm,
        "minus_loss_gradient_norm": minus_norm,
        "plus_loss_batch_values": plus_losses,
        "minus_loss_batch_values": minus_losses,
        "parameter_cosines_vs_checkpoint_delta": {
            name: parameter_cosine(tangent, checkpoint_tangent)
            for name, tangent in tangents.items()
        },
        "directions": direction_results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    shared.atomic_json(result_path, result)
    print(
        f"[gpu {device.index}] datapoint={datapoint_index} "
        f"opposite plus={direction_results['opposite']['plus_loss_sgd_jvp']['global_cosine']:+.4f} "
        f"minus={direction_results['opposite']['minus_loss_sgd_jvp']['global_cosine']:+.4f} "
        f"even={direction_results['opposite']['even_loss_sgd_jvp']['global_cosine']:+.4f} "
        f"odd={direction_results['opposite']['odd_loss_sgd_jvp']['global_cosine']:+.4f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument(
        "--evaluation-prompt", choices=("original", "random"), default="random"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    null_checkpoint = torch.load(
        ngcd_checkpoint_path(NGCD_NULL_EPOCH), map_location="cpu", weights_only=False
    )
    next_checkpoint = torch.load(
        ngcd_checkpoint_path(NGCD_NEXT_EPOCH), map_location="cpu", weights_only=False
    )
    schedule = base.make_linear_schedule(T, device=device)
    _, point_dir, _, _ = ngcd_odd_even_output_paths(args.evaluation_prompt)
    indices = nsdl_datapoint_indices()[args.shard_index :: args.shard_count]
    print(f"[gpu {args.gpu}] datapoints={list(indices)}", flush=True)
    for datapoint_index in indices:
        run_datapoint(
            datapoint_index,
            dataset,
            null_checkpoint,
            next_checkpoint,
            schedule,
            device,
            args.evaluation_prompt,
            point_dir,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
