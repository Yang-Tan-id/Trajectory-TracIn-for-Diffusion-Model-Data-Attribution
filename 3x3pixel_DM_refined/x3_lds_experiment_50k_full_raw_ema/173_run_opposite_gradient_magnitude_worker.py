"""Predict fresh-SGD output-change magnitudes using plus/minus loss gradients."""

import argparse
import importlib
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.func import functional_call, jvp

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from null_same_direction_learning_config import nsdl_checkpoint_path
from opposite_gradient_magnitude_config import *


shared = importlib.import_module("148_run_null_same_direction_learning_worker")
tblock = importlib.import_module("167_run_null_tblock_cross_direction_worker")


def block_gradient(model, x0, condition, direction, timestamps, schedule, device):
    parameters = tuple(model.parameters())
    t = torch.tensor(timestamps, device=device, dtype=torch.long)
    noise = direction.unsqueeze(0).expand(len(t), -1, -1, -1)
    xt = base.q_sample(x0.expand(len(t), -1, -1, -1), t, noise, schedule)
    prediction = model(xt, t, condition.expand(len(t), -1))
    loss = F.mse_loss(prediction, noise)
    gradient = torch.autograd.grad(loss, parameters)
    norm = sum(value.double().square().sum() for value in gradient).sqrt()
    return tuple(value.detach() for value in gradient), float(norm), float(loss.detach())


def combine(left, right, left_scale, right_scale):
    return tuple(
        left_scale * left_value + right_scale * right_value
        for left_value, right_value in zip(left, right)
    )


def clipped_sgd_tangent(gradient, learning_rate, clip_norm):
    norm = sum(value.double().square().sum() for value in gradient).sqrt()
    scale = min(1.0, clip_norm / max(float(norm), NSDL_EPS))
    tangent = tuple(-learning_rate * scale * value for value in gradient)
    tangent_norm = sum(value.double().square().sum() for value in tangent).sqrt()
    return tangent, float(norm), float(scale), float(tangent_norm)


def parameter_delta(model, updated_state):
    return tuple(
        updated_state[name].to(parameter.device).detach() - parameter.detach()
        for name, parameter in model.named_parameters()
    )


def tangent_agreement(predicted, actual):
    predicted_flat = torch.cat([value.double().reshape(-1) for value in predicted])
    actual_flat = torch.cat([value.double().reshape(-1) for value in actual])
    difference = predicted_flat - actual_flat
    denominator = actual_flat.norm().clamp_min(NSDL_EPS)
    cosine = torch.dot(predicted_flat, actual_flat) / (
        predicted_flat.norm() * actual_flat.norm()
    ).clamp_min(NSDL_EPS)
    return {
        "relative_l2_error": float(difference.norm() / denominator),
        "cosine": float(cosine),
        "predicted_norm": float(predicted_flat.norm()),
        "actual_norm": float(actual_flat.norm()),
    }


def predict_bank_jvp(
    model,
    names,
    parameters,
    tangent,
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

        def prediction_fn(*parameter_values):
            return functional_call(
                model,
                dict(zip(names, parameter_values)),
                (xt, flat_t, flat_condition),
            )

        _, predicted = jvp(prediction_fn, parameters, tangent)
        parts.append(
            predicted.reshape(direction_count, width, 3, 3, 3).detach().cpu()
        )
    return torch.cat(parts, dim=1)


def safe_correlation(left, right, rank=False):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) <= 0 or np.std(right) <= 0:
        return float("nan")
    if rank:
        return float(spearmanr(left, right).statistic)
    return float(np.corrcoef(left, right)[0, 1])


def comparison_metrics(predicted, actual):
    predicted = predicted.double().reshape(*predicted.shape[:2], -1)
    actual = actual.double().reshape(*actual.shape[:2], -1)
    predicted_l2 = predicted.norm(dim=2)
    actual_l2 = actual.norm(dim=2)
    denominator = (predicted_l2 * actual_l2).clamp_min(NSDL_EPS)
    point_cosine = (predicted * actual).sum(dim=2) / denominator
    predicted_flat = predicted.reshape(-1)
    actual_flat = actual.reshape(-1)
    global_cosine = torch.dot(predicted_flat, actual_flat) / (
        predicted_flat.norm() * actual_flat.norm()
    ).clamp_min(NSDL_EPS)
    predicted_np = predicted_l2.cpu().numpy().reshape(-1)
    actual_np = actual_l2.cpu().numpy().reshape(-1)
    ratio = predicted_np / np.maximum(actual_np, NSDL_EPS)
    relative_error = np.abs(predicted_np - actual_np) / np.maximum(
        actual_np, NSDL_EPS
    )
    metrics = {
        "magnitude_pearson": safe_correlation(predicted_np, actual_np),
        "magnitude_spearman": safe_correlation(predicted_np, actual_np, rank=True),
        "magnitude_ratio_mean": float(np.mean(ratio)),
        "magnitude_ratio_median": float(np.median(ratio)),
        "magnitude_relative_error_mean": float(np.mean(relative_error)),
        "magnitude_relative_error_median": float(np.median(relative_error)),
        "vector_global_cosine": float(global_cosine),
        "vector_point_cosine_mean": float(point_cosine.mean()),
        "vector_point_cosine_positive_fraction": float(
            (point_cosine > 0).double().mean()
        ),
        "actual_l2_mean": float(actual_l2.mean()),
        "predicted_l2_mean": float(predicted_l2.mean()),
    }
    arrays = {
        "predicted_l2": predicted_l2.cpu().numpy(),
        "actual_l2": actual_l2.cpu().numpy(),
        "point_vector_cosine": point_cosine.cpu().numpy(),
    }
    return metrics, arrays


def run_source(source_index, dataset, checkpoint, schedule, device):
    target_index = ntcd_target_index(source_index)
    output_dir = ogm_source_dir(source_index)
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
    directions, _, _ = tblock.target_directions(
        source_index, target_index, source_direction, device
    )
    model = shared.make_model(dataset, checkpoint["model_state"], device)
    names = tuple(name for name, _ in model.named_parameters())
    parameters = tuple(model.parameters())
    learning_rate = shared.checkpoint_learning_rate(checkpoint)
    clip_norm = float(checkpoint.get("config", {}).get("grad_clip_norm", GRAD_CLIP))
    fresh_dir = ntcd_source_dir(source_index, "fresh_sgd")
    with open(fresh_dir / "result.json") as handle:
        fresh_result = json.load(handle)
    actual_archive = np.load(
        fresh_dir / "target_direction_prediction_deltas.npz", allow_pickle=False
    )
    saved_arrays = {}
    block_results = []
    started = time.perf_counter()
    for block_index, timestamps in enumerate(NTCD_TIMESTAMP_BLOCKS):
        plus, plus_norm, plus_loss = block_gradient(
            model,
            source_x0,
            source_condition,
            source_direction,
            timestamps,
            schedule,
            device,
        )
        minus, minus_norm, minus_loss = block_gradient(
            model,
            source_x0,
            source_condition,
            -source_direction,
            timestamps,
            schedule,
            device,
        )
        gradient_bank = {
            "plus_gradient_sgd": plus,
            "minus_gradient_sgd": minus,
            "even_gradient_sgd": combine(plus, minus, 0.5, 0.5),
            "odd_gradient_sgd": combine(plus, minus, 0.5, -0.5),
        }
        tangent_bank = {}
        tangent_metadata = {}
        for name, gradient in gradient_bank.items():
            tangent, gradient_norm, clip_scale, tangent_norm = clipped_sgd_tangent(
                gradient, learning_rate, clip_norm
            )
            tangent_bank[name] = tangent
            tangent_metadata[name] = {
                "gradient_norm": gradient_norm,
                "clip_scale": clip_scale,
                "tangent_norm": tangent_norm,
            }
        updated_payload = torch.load(
            fresh_result["blocks"][block_index]["updated_model"],
            map_location="cpu",
            weights_only=False,
        )
        actual_tangent = parameter_delta(model, updated_payload["model_state"])
        actual_tangent_norm = sum(
            value.double().square().sum() for value in actual_tangent
        ).sqrt()
        tangent_bank["actual_parameter_delta_jvp"] = actual_tangent
        tangent_metadata["actual_parameter_delta_jvp"] = {
            "gradient_norm": None,
            "clip_scale": None,
            "tangent_norm": float(actual_tangent_norm),
        }
        plus_step_agreement = tangent_agreement(
            tangent_bank["plus_gradient_sgd"], actual_tangent
        )
        # Ground truth is exactly one clipped fresh-SGD step on the plus loss.
        # Float32 parameter subtraction adds a small reconstruction error, but
        # a large mismatch invalidates the claimed comparison.
        if plus_step_agreement["cosine"] < 0.999:
            raise RuntimeError(
                "reconstructed plus-gradient SGD step does not match saved "
                f"fresh-SGD update: {plus_step_agreement}"
            )
        actual = torch.from_numpy(
            actual_archive[f"block_{block_index}_prediction_delta"]
        )
        candidate_results = {}
        saved_arrays[f"block_{block_index}_actual_l2"] = (
            actual.double().reshape(*actual.shape[:2], -1).norm(dim=2).numpy()
        )
        for candidate, tangent in tangent_bank.items():
            predicted = predict_bank_jvp(
                model,
                names,
                parameters,
                tangent,
                target_x0,
                target_condition,
                directions,
                schedule,
                device,
            )
            metrics, arrays = comparison_metrics(predicted, actual)
            candidate_results[candidate] = {
                "tangent": tangent_metadata[candidate],
                "metrics": metrics,
            }
            for array_name, value in arrays.items():
                if array_name == "actual_l2":
                    continue
                saved_arrays[
                    f"block_{block_index}_{candidate}_{array_name}"
                ] = value
        block_results.append(
            {
                "block_index": block_index,
                "timestamp_start": timestamps[0],
                "timestamp_end": timestamps[-1],
                "plus_loss": plus_loss,
                "minus_loss": minus_loss,
                "plus_gradient_norm": plus_norm,
                "minus_gradient_norm": minus_norm,
                "plus_step_vs_saved_fresh_sgd": plus_step_agreement,
                "candidates": candidate_results,
            }
        )
        print(
            f"[gpu {device.index}] source={source_index} block={block_index} "
            f"plus_rho={candidate_results['plus_gradient_sgd']['metrics']['magnitude_spearman']:+.4f} "
            f"minus_rho={candidate_results['minus_gradient_sgd']['metrics']['magnitude_spearman']:+.4f} "
            f"even_rho={candidate_results['even_gradient_sgd']['metrics']['magnitude_spearman']:+.4f}",
            flush=True,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "magnitude_predictions.npz", **saved_arrays)
    shared.atomic_json(
        result_path,
        {
            "source_datapoint_index": int(source_index),
            "target_datapoint_index": int(target_index),
            "actual_update": "fresh_sgd_plus_direction",
            "learning_rate": learning_rate,
            "clip_norm": clip_norm,
            "target_direction_count": NTCD_TARGET_DIRECTION_COUNT,
            "target_timestamps": T,
            "blocks": block_results,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


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
    for source_index in source_indices:
        run_source(source_index, dataset, checkpoint, schedule, device)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
