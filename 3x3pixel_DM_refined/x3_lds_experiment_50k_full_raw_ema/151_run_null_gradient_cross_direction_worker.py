"""Test whether a same-direction loss gradient predicts cross-direction change."""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from null_gradient_cross_direction_config import *


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def make_model(dataset, state, device):
    model = base.CondEpsModel(
        in_ch=3,
        cond_dim=len(dataset.vocab),
        base_ch=BASE_CH,
        time_dim=TIME_DIM,
    ).to(device)
    model.load_state_dict(state, strict=True)
    return model.eval()


def fixed_direction(datapoint_index, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(NSDL_DIRECTION_SEED_BASE + int(datapoint_index))
    return torch.randn(
        (3, 3, 3), generator=generator, device=device, dtype=torch.float32
    )


def random_evaluation_condition(
    datapoint_index, dataset, loss_condition, device
):
    generator = np.random.default_rng(
        NGCD_RANDOM_PROMPT_SEED_BASE + int(datapoint_index)
    )
    loss_cpu = loss_condition.detach().cpu()
    for candidate_index in generator.permutation(N_TRAIN):
        _, candidate = dataset[int(candidate_index)]
        if not torch.equal(candidate, loss_cpu):
            return (
                candidate.unsqueeze(0).to(device),
                int(candidate_index),
            )
    raise RuntimeError("could not find a random prompt different from loss prompt")


def mean_positive_loss_gradient(
    model, x0, condition, direction, schedule, device
):
    parameters = tuple(model.parameters())
    accumulated = [torch.zeros_like(parameter) for parameter in parameters]
    timestamps = torch.tensor(
        NSDL_TIMESTAMPS, device=device, dtype=torch.long
    )
    batches = 0
    batch_losses = []
    for start in range(0, T, NSDL_UPDATE_BATCH_SIZE):
        t = timestamps[start : start + NSDL_UPDATE_BATCH_SIZE]
        noise = direction.unsqueeze(0).expand(len(t), -1, -1, -1)
        xt = base.q_sample(x0.expand(len(t), -1, -1, -1), t, noise, schedule)
        prediction = model(xt, t, condition.expand(len(t), -1))
        loss = F.mse_loss(prediction, noise)
        gradients = torch.autograd.grad(loss, parameters)
        for destination, gradient in zip(accumulated, gradients):
            destination.add_(gradient)
        batches += 1
        batch_losses.append(float(loss.detach()))
    gradient = tuple(value / float(batches) for value in accumulated)
    squared_norm = sum(value.double().square().sum() for value in gradient)
    return gradient, float(squared_norm.sqrt()), batch_losses


def comparison_metrics(predicted, actual):
    predicted_flat = predicted.reshape(-1).double()
    actual_flat = actual.reshape(-1).double()
    predicted_norm = predicted_flat.norm()
    actual_norm = actual_flat.norm()
    cosine = torch.dot(predicted_flat, actual_flat) / (
        predicted_norm * actual_norm
    ).clamp_min(NGCD_EPS)
    scale = torch.dot(predicted_flat, actual_flat) / predicted_flat.square().sum().clamp_min(
        NGCD_EPS
    )
    residual = actual_flat - scale * predicted_flat
    timestamp_predicted = predicted.reshape(T, -1).double()
    timestamp_actual = actual.reshape(T, -1).double()
    per_timestamp_cosine = (
        (timestamp_predicted * timestamp_actual).sum(1)
        / (
            timestamp_predicted.norm(dim=1)
            * timestamp_actual.norm(dim=1)
        ).clamp_min(NGCD_EPS)
    )
    return {
        "global_cosine": float(cosine),
        "per_timestamp_cosine_mean": float(per_timestamp_cosine.mean()),
        "per_timestamp_cosine_std": float(
            per_timestamp_cosine.std(unbiased=False)
        ),
        "per_timestamp_cosine_median": float(per_timestamp_cosine.median()),
        "per_timestamp_positive_fraction": float(
            (per_timestamp_cosine > 0).double().mean()
        ),
        "predicted_norm": float(predicted_norm),
        "actual_norm": float(actual_norm),
        "norm_ratio": float(
            predicted_norm / actual_norm.clamp_min(NGCD_EPS)
        ),
        "best_scalar": float(scale),
        "best_scaled_relative_residual": float(
            residual.norm() / actual_norm.clamp_min(NGCD_EPS)
        ),
    }


@torch.no_grad()
def parameter_delta(null_parameters, next_parameters):
    return tuple(
        next_parameter.detach() - null_parameter.detach()
        for null_parameter, next_parameter in zip(
            null_parameters, next_parameters
        )
    )


def evaluate_direction(
    null_model,
    next_model,
    names,
    parameters,
    sgd_tangent,
    checkpoint_tangent,
    x0,
    condition,
    direction,
    schedule,
    device,
):
    timestamps = torch.tensor(
        NSDL_TIMESTAMPS, device=device, dtype=torch.long
    )
    actual_parts = []
    sgd_parts = []
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

        null_prediction, sgd_prediction_delta = jvp(
            prediction_fn, parameters, sgd_tangent
        )
        _, checkpoint_prediction_delta = jvp(
            prediction_fn, parameters, checkpoint_tangent
        )
        with torch.no_grad():
            actual_prediction_delta = (
                next_model(xt, t, expanded_condition) - null_prediction
            )
        actual_parts.append(actual_prediction_delta.detach().cpu())
        sgd_parts.append(sgd_prediction_delta.detach().cpu())
        checkpoint_parts.append(checkpoint_prediction_delta.detach().cpu())
    actual = torch.cat(actual_parts, dim=0)
    sgd = torch.cat(sgd_parts, dim=0)
    checkpoint = torch.cat(checkpoint_parts, dim=0)
    return {
        "single_point_sgd_jvp": comparison_metrics(sgd, actual),
        "checkpoint_parameter_delta_jvp": comparison_metrics(
            checkpoint, actual
        ),
    }, actual, sgd, checkpoint


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
        evaluation_condition, prompt_source_index = random_evaluation_condition(
            datapoint_index, dataset, loss_condition[0], device
        )
    else:
        evaluation_condition = loss_condition
        prompt_source_index = int(datapoint_index)
    direction = fixed_direction(datapoint_index, device)
    null_model = make_model(dataset, null_checkpoint["model_state"], device)
    next_model = make_model(dataset, next_checkpoint["model_state"], device)
    names = tuple(name for name, _ in null_model.named_parameters())
    parameters = tuple(null_model.parameters())
    next_parameters = tuple(next_model.parameters())
    started = time.perf_counter()
    loss_gradient, gradient_norm, batch_losses = mean_positive_loss_gradient(
        null_model, x0, loss_condition, direction, schedule, device
    )
    # A gradient-descent update has parameter tangent -g. Its scale is
    # irrelevant for the direction cosine tested here.
    sgd_tangent = tuple(-value for value in loss_gradient)
    checkpoint_tangent = parameter_delta(parameters, next_parameters)
    parameter_cosine = float(
        sum(
            (left.double() * right.double()).sum()
            for left, right in zip(sgd_tangent, checkpoint_tangent)
        )
        / (
            sum(value.double().square().sum() for value in sgd_tangent).sqrt()
            * sum(
                value.double().square().sum()
                for value in checkpoint_tangent
            ).sqrt()
        ).clamp_min(NGCD_EPS)
    )
    direction_results = {}
    arrays = {}
    for direction_name, evaluation_direction in (
        ("same", direction),
        ("opposite", -direction),
    ):
        metrics, actual, sgd, checkpoint = evaluate_direction(
            null_model,
            next_model,
            names,
            parameters,
            sgd_tangent,
            checkpoint_tangent,
            x0,
            evaluation_condition,
            evaluation_direction,
            schedule,
            device,
        )
        direction_results[direction_name] = metrics
        arrays[f"{direction_name}_actual_delta"] = actual.numpy()
        arrays[f"{direction_name}_single_point_sgd_jvp"] = sgd.numpy()
        arrays[f"{direction_name}_checkpoint_delta_jvp"] = checkpoint.numpy()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "predicted_noise_direction_arrays.npz",
        training_noise_direction=direction.detach().cpu().numpy(),
        **arrays,
    )
    result = {
        "datapoint_index": int(datapoint_index),
        "null_epoch": NGCD_NULL_EPOCH,
        "next_epoch": NGCD_NEXT_EPOCH,
        "loss_prompt_source_index": int(datapoint_index),
        "evaluation_prompt_mode": evaluation_prompt,
        "evaluation_prompt_source_index": prompt_source_index,
        "loss_condition": loss_condition[0].detach().cpu().tolist(),
        "evaluation_condition": evaluation_condition[0].detach().cpu().tolist(),
        "timestamps": list(NSDL_TIMESTAMPS),
        "same_direction_loss_batch_values": batch_losses,
        "same_direction_loss_gradient_norm": gradient_norm,
        "parameter_cosine_single_point_sgd_vs_checkpoint_delta": parameter_cosine,
        "directions": direction_results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(result_path, result)
    primary = direction_results["opposite"]["single_point_sgd_jvp"]
    control = direction_results["opposite"]["checkpoint_parameter_delta_jvp"]
    print(
        f"[gpu {device.index}] datapoint={datapoint_index} "
        f"opposite_single_cos={primary['global_cosine']:+.6f} "
        f"opposite_control_cos={control['global_cosine']:+.6f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument(
        "--evaluation-prompt", choices=("original", "random"), default="original"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    null_checkpoint = torch.load(
        ngcd_checkpoint_path(NGCD_NULL_EPOCH),
        map_location="cpu",
        weights_only=False,
    )
    next_checkpoint = torch.load(
        ngcd_checkpoint_path(NGCD_NEXT_EPOCH),
        map_location="cpu",
        weights_only=False,
    )
    schedule = base.make_linear_schedule(T, device=device)
    _, point_dir, _, _ = ngcd_output_paths(args.evaluation_prompt)
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
