"""Predict fixed target responses using only a reconstructed source SGD step."""

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
from source_gradient_fixed_point_config import *


shared = importlib.import_module("148_run_null_same_direction_learning_worker")
magnitude = importlib.import_module("173_run_opposite_gradient_magnitude_worker")
path_worker = importlib.import_module("188_run_path_integrated_jvp_worker")

ORIGINAL_TIME_EMBEDDING = base.sinusoidal_time_embedding


def configure_training_precision():
    """Restore the precision mode used by the original float32 source update."""
    base.sinusoidal_time_embedding = ORIGINAL_TIME_EMBEDDING
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def quantize_float32_parameter_step(parameters, tangent):
    """Include the rounding caused by storing the updated float32 parameters."""
    return tuple(
        (parameter.detach() + delta.detach()) - parameter.detach()
        for parameter, delta in zip(parameters, tangent)
    )


def reconstruct_source_tangents(
    source_index, dataset, checkpoint, schedule, device
):
    configure_training_precision()
    source_image, source_condition = dataset[int(source_index)]
    source_x0 = source_image.unsqueeze(0).to(device=device, dtype=torch.float32)
    source_condition = source_condition.unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    source_direction = shared.fixed_direction(source_index, device)
    model = shared.make_model(dataset, checkpoint["model_state"], device)
    parameters = tuple(model.parameters())
    learning_rate = shared.checkpoint_learning_rate(checkpoint)
    clip_norm = float(checkpoint.get("config", {}).get("grad_clip_norm", GRAD_CLIP))
    fresh_dir = ntcd_source_dir(source_index, "fresh_sgd")
    with open(fresh_dir / "result.json") as handle:
        fresh_result = json.load(handle)

    results = []
    for block_index, timestamps in enumerate(NTCD_TIMESTAMP_BLOCKS):
        gradient, gradient_norm, loss = magnitude.block_gradient(
            model,
            source_x0,
            source_condition,
            source_direction,
            timestamps,
            schedule,
            device,
        )
        predicted, _, clip_scale, predicted_norm = magnitude.clipped_sgd_tangent(
            gradient, learning_rate, clip_norm
        )
        quantized = quantize_float32_parameter_step(parameters, predicted)
        payload = torch.load(
            fresh_result["blocks"][block_index]["updated_model"],
            map_location="cpu",
            weights_only=False,
        )
        exact = magnitude.parameter_delta(model, payload["model_state"])
        results.append(
            {
                "source_gradient_sgd": tuple(value.detach() for value in predicted),
                "source_gradient_sgd_float32_quantized": tuple(
                    value.detach() for value in quantized
                ),
                "exact_parameter_delta_control": tuple(
                    value.detach() for value in exact
                ),
                "updated_model_path": fresh_result["blocks"][block_index][
                    "updated_model"
                ],
                "metadata": {
                    "loss": loss,
                    "gradient_norm": gradient_norm,
                    "clip_scale": clip_scale,
                    "predicted_tangent_norm": predicted_norm,
                    "unquantized_vs_exact": magnitude.tangent_agreement(
                        predicted, exact
                    ),
                    "quantized_vs_exact": magnitude.tangent_agreement(
                        quantized, exact
                    ),
                },
            }
        )
    del model, parameters
    torch.cuda.empty_cache()
    return results


def evaluate_block(
    model,
    names,
    parameters,
    tangents,
    exact_tangent,
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
    for method in SGFP_METHODS:
        for metric in (
            "predicted_l2",
            "vector_cosine",
            "vector_relative_error",
            "magnitude_relative_error",
        ):
            arrays[f"{method}_{metric}"] = np.empty(total, dtype=np.float64)

    exact_endpoint = tuple(
        parameter + delta for parameter, delta in zip(parameters, exact_tangent)
    )
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

        with torch.no_grad():
            before = prediction_fn(*parameters)
            after = prediction_fn(*exact_endpoint)
            actual = after - before
        arrays["actual_l2"][start:end] = (
            actual.detach().double().flatten(1).norm(dim=1).cpu().numpy()
        )
        for method, tangent in tangents.items():
            _, predicted = jvp(prediction_fn, parameters, tangent)
            metrics = path_worker.point_metrics(predicted, actual)
            for metric, values in metrics.items():
                arrays[f"{method}_{metric}"][start:end] = values
        print(
            f"[gpu {device.index}] fixed-points={end}/{total}",
            flush=True,
        )
        del before, after, actual
    shape = (PIJVP_DIRECTION_COUNT, PIJVP_TIMESTAMP_COUNT)
    return {name: value.reshape(shape) for name, value in arrays.items()}


def run_source(source_index, dataset, checkpoint, device, batch_size):
    target_index = ntcd_target_index(source_index)
    output_dir = sgfp_source_dir(source_index)
    done_path = output_dir / "done.json"
    if done_path.is_file():
        print(
            f"[gpu {device.index}] skip source={source_index} target={target_index}",
            flush=True,
        )
        return

    training_schedule = base.make_linear_schedule(T, device=device)
    reconstructed = reconstruct_source_tangents(
        source_index, dataset, checkpoint, training_schedule, device
    )
    path_worker.configure_high_precision()
    evaluation_schedule = path_worker.schedule_to_float64(
        base.make_linear_schedule(T, device=device)
    )
    target_image, target_condition = dataset[int(target_index)]
    target_x0 = target_image.unsqueeze(0).to(device=device, dtype=torch.float64)
    target_condition = target_condition.unsqueeze(0).to(
        device=device, dtype=torch.float64
    )
    model = shared.make_model(dataset, checkpoint["model_state"], device).double()
    names = tuple(name for name, _ in model.named_parameters())
    parameters = tuple(parameter.detach() for parameter in model.parameters())
    edmc_dir = edmc_source_dir(source_index)
    with np.load(edmc_dir / "block_0_responses.npz", allow_pickle=False) as archive:
        directions = torch.from_numpy(archive["directions"]).to(
            device=device, dtype=torch.float64
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = []
    started = time.perf_counter()
    for block_index, result in enumerate(reconstructed):
        block_path = output_dir / f"block_{block_index}_source_gradient_jvp.npz"
        tangents = {
            method: tuple(value.to(dtype=torch.float64) for value in result[method])
            for method in SGFP_METHODS
        }
        updated_payload = torch.load(
            result["updated_model_path"], map_location="cpu", weights_only=False
        )
        tangents["exact_parameter_delta_control"] = tuple(
            updated_payload["model_state"][name].to(
                device=device, dtype=torch.float64
            )
            - parameter
            for name, parameter in zip(names, parameters)
        )
        responses = evaluate_block(
            model,
            names,
            parameters,
            tangents,
            tangents["exact_parameter_delta_control"],
            target_x0,
            target_condition,
            directions,
            evaluation_schedule,
            device,
            batch_size,
        )
        np.savez_compressed(
            block_path,
            direction_indices=np.asarray(PIJVP_DIRECTION_INDICES, dtype=np.int64),
            timestamps=np.asarray(PIJVP_TIMESTAMPS, dtype=np.int64),
            **responses,
        )
        blocks.append(
            {
                "block_index": block_index,
                "responses": str(block_path),
                **result["metadata"],
            }
        )
        print(
            f"[gpu {device.index}] source={source_index} block={block_index} "
            f"gradient/exact-cos={result['metadata']['unquantized_vs_exact']['cosine']:+.6f} "
            f"quantized/exact-cos={result['metadata']['quantized_vs_exact']['cosine']:+.6f}",
            flush=True,
        )
        del tangents, responses, updated_payload
        torch.cuda.empty_cache()
    shared.atomic_json(
        done_path,
        {
            "source_datapoint_index": int(source_index),
            "target_datapoint_index": int(target_index),
            "prediction_uses_saved_parameter_delta": False,
            "source_update_reconstruction": "baseline loss gradient + clipping + fresh SGD learning rate",
            "target_evaluation_precision": PIJVP_PRECISION_MODE,
            "methods": list(SGFP_METHODS),
            "blocks": blocks,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=SGFP_DEFAULT_BATCH_SIZE)
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
