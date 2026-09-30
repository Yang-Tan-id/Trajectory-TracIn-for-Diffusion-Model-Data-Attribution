"""Test source-point gradients on a different target datapoint."""

import argparse
import importlib
import time

import numpy as np
import torch

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from null_gradient_cross_direction_config import *


shared = importlib.import_module("151_run_null_gradient_cross_direction_worker")
odd_even = importlib.import_module("158_run_null_gradient_odd_even_worker")


def run_pair(
    source_index,
    target_index,
    dataset,
    null_checkpoint,
    next_checkpoint,
    schedule,
    device,
    pair_dir,
):
    output_dir = pair_dir / f"source_{source_index:05d}_target_{target_index:05d}"
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
    target_x0 = target_image.unsqueeze(0).to(device)
    source_condition = source_condition.unsqueeze(0).to(device)
    target_condition = target_condition.unsqueeze(0).to(device)
    direction = shared.fixed_direction(source_index, device)

    null_model = shared.make_model(dataset, null_checkpoint["model_state"], device)
    next_model = shared.make_model(dataset, next_checkpoint["model_state"], device)
    names = tuple(name for name, _ in null_model.named_parameters())
    parameters = tuple(null_model.parameters())
    next_parameters = tuple(next_model.parameters())
    started = time.perf_counter()

    plus_gradient, plus_norm, plus_losses = shared.mean_positive_loss_gradient(
        null_model,
        source_x0,
        source_condition,
        direction,
        schedule,
        device,
    )
    minus_gradient, minus_norm, minus_losses = shared.mean_positive_loss_gradient(
        null_model,
        source_x0,
        source_condition,
        -direction,
        schedule,
        device,
    )
    tangents = {
        "plus_loss_sgd_jvp": tuple(-value for value in plus_gradient),
        "even_loss_sgd_jvp": odd_even.combine(
            plus_gradient, minus_gradient, -0.5, -0.5
        ),
    }
    checkpoint_tangent = shared.parameter_delta(parameters, next_parameters)
    direction_results = {}
    saved_arrays = {"shared_noise_direction": direction.detach().cpu().numpy()}
    for direction_name, evaluation_direction in (
        ("same", direction),
        ("opposite", -direction),
    ):
        metrics, arrays = odd_even.evaluate_tangent_bank(
            null_model,
            next_model,
            names,
            parameters,
            tangents,
            checkpoint_tangent,
            target_x0,
            target_condition,
            evaluation_direction,
            schedule,
            device,
        )
        direction_results[direction_name] = metrics
        for name, value in arrays.items():
            saved_arrays[f"{direction_name}_{name}"] = value.numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "cross_datapoint_direction_arrays.npz", **saved_arrays
    )
    result = {
        "source_datapoint_index": int(source_index),
        "target_datapoint_index": int(target_index),
        "null_epoch": NGCD_NULL_EPOCH,
        "next_epoch": NGCD_NEXT_EPOCH,
        "source_condition": source_condition[0].detach().cpu().tolist(),
        "target_condition": target_condition[0].detach().cpu().tolist(),
        "target_uses_own_prompt": True,
        "plus_loss_gradient_norm": plus_norm,
        "minus_loss_gradient_norm": minus_norm,
        "plus_loss_batch_values": plus_losses,
        "minus_loss_batch_values": minus_losses,
        "parameter_cosines_vs_checkpoint_delta": {
            name: odd_even.parameter_cosine(tangent, checkpoint_tangent)
            for name, tangent in tangents.items()
        },
        "directions": direction_results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    shared.atomic_json(result_path, result)
    print(
        f"[gpu {device.index}] source={source_index} target={target_index} "
        f"same plus={direction_results['same']['plus_loss_sgd_jvp']['global_cosine']:+.4f} "
        f"even={direction_results['same']['even_loss_sgd_jvp']['global_cosine']:+.4f} | "
        f"opposite plus={direction_results['opposite']['plus_loss_sgd_jvp']['global_cosine']:+.4f} "
        f"even={direction_results['opposite']['even_loss_sgd_jvp']['global_cosine']:+.4f}",
        flush=True,
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
    null_checkpoint = torch.load(
        ngcd_checkpoint_path(NGCD_NULL_EPOCH), map_location="cpu", weights_only=False
    )
    next_checkpoint = torch.load(
        ngcd_checkpoint_path(NGCD_NEXT_EPOCH), map_location="cpu", weights_only=False
    )
    schedule = base.make_linear_schedule(T, device=device)
    _, pair_dir, _, _ = ngcd_cross_datapoint_output_paths()
    source_indices = nsdl_datapoint_indices()[args.shard_index :: args.shard_count]
    for source_index in source_indices:
        target_index = ngcd_cross_datapoint_target_index(source_index)
        run_pair(
            source_index,
            target_index,
            dataset,
            null_checkpoint,
            next_checkpoint,
            schedule,
            device,
            pair_dir,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
