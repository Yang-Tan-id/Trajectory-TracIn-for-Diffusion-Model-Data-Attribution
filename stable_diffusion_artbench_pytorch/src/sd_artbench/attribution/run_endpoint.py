"""Streaming raw/AdamW Endpoint-TracIn and Endpoint-Pollute for LoRA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import ArtBenchConfig
from .optimizer import adamw_full_direction
from .projection import CountSketchProjector, stable_seed
from .runtime import (
    AttributionArtBenchDataset,
    build_runtime,
    cfg_predictions,
    deterministic_noise,
    encode_image,
    encode_prompt,
    gradients,
    list_checkpoints,
    load_queries,
    load_query_trajectories,
    load_raw_checkpoint,
    mean_simple_gradient,
    mean_trajectory_gradient,
    simple_loss,
)
from .scoring import TRANSFORMS, VARIANTS, score_variants


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split() if item]


def parse_args() -> argparse.Namespace:
    defaults = ArtBenchConfig()
    root = defaults.index_root.parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lambdalabs/miniSD-diffusers")
    parser.add_argument("--checkpoint-root", type=Path, default=root / "outputs/artbench2-lora")
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, default=defaults.metadata_csv)
    parser.add_argument("--image-root", type=Path, default=defaults.image_root)
    parser.add_argument("--index-path", type=Path, default=defaults.index_root / "idx-train.pkl")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/attribution-endpoint")
    parser.add_argument("--methods", default="endpoint_tracin,endpoint_pollute")
    parser.add_argument("--checkpoint-indices", type=parse_ints)
    parser.add_argument("--query-indices", type=parse_ints)
    parser.add_argument("--timestep-count", type=int, default=50)
    parser.add_argument("--timestep-batch-size", type=int, default=8)
    parser.add_argument("--proj-dim", type=int, default=32768)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-end", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def project_dual(runtime, gradients_raw, projector, max_grad_norm):
    raw, raw_norm = projector.project(gradients_raw)
    adamw, learning_rate = adamw_full_direction(
        runtime.parameters,
        gradients_raw,
        runtime.optimizer,
        max_grad_norm=max_grad_norm,
    )
    adamw_projected, adamw_norm = projector.project(adamw)
    return raw, adamw_projected, raw_norm, adamw_norm, learning_rate


def save_result(path: Path, scores: np.ndarray, metadata: dict, indices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            scores=scores,
            train_metadata_indices=indices,
            transform_names=np.asarray(TRANSFORMS),
            variant_names=np.asarray(VARIANTS),
            manifest_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(path)


def load_result(
    path: Path, expected_shape: tuple[int, ...], expected_metadata: dict | None = None
) -> tuple[np.ndarray, int]:
    if not path.is_file():
        return np.zeros(expected_shape, dtype=np.float64), -1
    with np.load(path, allow_pickle=False) as payload:
        scores = np.asarray(payload["scores"], dtype=np.float64)
        metadata = json.loads(str(np.asarray(payload["manifest_json"]).item()))
    if scores.shape != expected_shape:
        raise ValueError(f"{path}: score shape {scores.shape} != {expected_shape}")
    if expected_metadata is not None:
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"{path}: saved {key}={metadata.get(key)!r} != requested {value!r}"
                )
    return scores, int(metadata.get("completed_through_checkpoint_slot", -1))


def load_pollute_state(
    path: Path,
    score_shape: tuple[int, ...],
    timestep_count: int,
    expected_metadata: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    timestamp_shape = (timestep_count,) + score_shape
    if not path.is_file():
        return (
            np.zeros(score_shape, dtype=np.float64),
            np.zeros(score_shape, dtype=np.float64),
            np.zeros(timestamp_shape, dtype=np.float32),
            -1,
        )
    with np.load(path, allow_pickle=False) as payload:
        linear = np.asarray(payload["linear"], dtype=np.float64)
        termwise = np.asarray(payload["termwise_squared"], dtype=np.float64)
        timestamps = np.asarray(payload["timestamp_components"], dtype=np.float32)
        completed = int(np.asarray(payload["completed_slot"]).item())
        metadata = json.loads(str(np.asarray(payload["manifest_json"]).item()))
    if linear.shape != score_shape or termwise.shape != score_shape:
        raise ValueError(f"{path}: endpoint-pollute score state has the wrong shape")
    if timestamps.shape != timestamp_shape:
        raise ValueError(f"{path}: {timestamps.shape} != {timestamp_shape}")
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"{path}: saved {key}={metadata.get(key)!r} != requested {value!r}"
            )
    return linear, termwise, timestamps, completed


def save_pollute_state(
    path: Path,
    linear: np.ndarray,
    termwise_squared: np.ndarray,
    timestamp_components: np.ndarray,
    completed_slot: int,
    metadata: dict,
) -> None:
    """Atomically commit one whole checkpoint's pollute accumulation."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            linear=linear,
            termwise_squared=termwise_squared,
            timestamp_components=timestamp_components,
            completed_slot=np.asarray(completed_slot, dtype=np.int64),
            manifest_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    methods = {item.strip() for item in args.methods.split(",") if item.strip()}
    unknown = methods.difference({"endpoint_tracin", "endpoint_pollute"})
    if unknown:
        raise ValueError(f"unknown endpoint methods: {sorted(unknown)}")
    if not methods:
        raise ValueError("at least one endpoint method is required")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    runtime = build_runtime(
        model_name=args.model,
        rank=args.rank,
        device=device,
        mixed_precision=args.mixed_precision,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    checkpoints = list_checkpoints(args.checkpoint_root)
    checkpoint_indices = (
        args.checkpoint_indices
        if args.checkpoint_indices is not None
        else list(range(len(checkpoints)))
    )
    if any(index < 0 or index >= len(checkpoints) for index in checkpoint_indices):
        raise ValueError(f"checkpoint indices must be in [0, {len(checkpoints) - 1}]")
    queries = load_queries(args.query_manifest)
    if args.query_indices is not None:
        queries = [queries[index] for index in args.query_indices]
    if not queries:
        raise ValueError("at least one query is required")
    query_trajectories = load_query_trajectories(
        queries, expected_steps=args.timestep_count
    )
    timesteps = query_trajectories[0].timesteps.tolist()
    dataset = AttributionArtBenchDataset(
        args.metadata_csv, args.image_root, args.index_path, runtime.tokenizer
    )
    train_end = len(dataset) if args.train_end is None else min(args.train_end, len(dataset))
    if args.train_start < 0 or args.train_start >= train_end:
        raise ValueError(f"invalid training range [{args.train_start}, {train_end})")
    train_positions = list(range(args.train_start, train_end))
    metadata_indices = np.asarray(
        [dataset.metadata_indices[position] for position in train_positions], dtype=np.int64
    )

    query_hidden = []
    for query in queries:
        query_hidden.append(encode_prompt(runtime, query.prompt))
    unconditional_hidden = encode_prompt(runtime, "")

    # VAE posterior modes and text embeddings do not depend on checkpoints.
    # Keep them on CPU after first use; this avoids millions of redundant VAE
    # encodes in the timestamp-wise endpoint-pollute loop.
    train_encoding_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def train_encoding(position: int) -> tuple[torch.Tensor, torch.Tensor]:
        if position not in train_encoding_cache:
            item = dataset[position]
            latent = encode_image(runtime, item["pixel_values"]).detach().cpu()
            hidden = encode_prompt(runtime, item["input_ids"]).detach().cpu()
            train_encoding_cache[position] = (latent, hidden)
        latent, hidden = train_encoding_cache[position]
        return (
            latent.to(device=device, dtype=runtime.weight_dtype),
            hidden.to(device=device, dtype=runtime.weight_dtype),
        )

    shape = (len(TRANSFORMS), len(VARIANTS), len(queries), len(train_positions))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_signature = {
        "model": args.model,
        "rank": args.rank,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "checkpoint_indices": checkpoint_indices,
        "timesteps": timesteps,
        "timestep_batch_size": args.timestep_batch_size,
        "proj_dim": args.proj_dim,
        "train_positions": [args.train_start, train_end],
        "query_ids": [query.query_id for query in queries],
        "query_manifest": str(args.query_manifest.resolve()),
        "query_definition": "fixed_ddim_trajectory_reference_predicted_noise",
        "guidance_scales": [query.guidance_scale for query in queries],
        "max_grad_norm": args.max_grad_norm,
    }
    tracin_scores, tracin_completed = load_result(
        args.output_dir / "endpoint_tracin.npz", shape, run_signature
    )
    pollute_linear = np.zeros(shape, dtype=np.float64)
    pollute_termwise_squared = np.zeros(shape, dtype=np.float64)
    pollute_completed = -1
    pollute_timestamp_components = None
    if "endpoint_pollute" in methods:
        (
            pollute_linear,
            pollute_termwise_squared,
            pollute_timestamp_components,
            pollute_completed,
        ) = load_pollute_state(
            args.output_dir / "endpoint_pollute_state.npz",
            shape,
            args.timestep_count,
            run_signature,
        )

    for checkpoint_slot, checkpoint_index in enumerate(checkpoint_indices):
        checkpoint = checkpoints[checkpoint_index]
        checkpoint_lr = load_raw_checkpoint(runtime, checkpoint)
        projector = CountSketchProjector(
            args.proj_dim,
            stable_seed("sd_artbench", "projection", checkpoint_index),
        )
        print(
            f"[checkpoint {checkpoint_slot + 1}/{len(checkpoint_indices)}] "
            f"{checkpoint.name} lr={checkpoint_lr:.8g}",
            flush=True,
        )

        did_tracin = "endpoint_tracin" in methods and checkpoint_slot > tracin_completed
        did_pollute = "endpoint_pollute" in methods and checkpoint_slot > pollute_completed
        if did_tracin:
            query_features = []
            for query, trajectory, hidden in zip(
                queries, query_trajectories, query_hidden
            ):
                gradient = mean_trajectory_gradient(
                    runtime,
                    trajectory,
                    hidden,
                    unconditional_hidden,
                    query.guidance_scale,
                    timestep_batch_size=args.timestep_batch_size,
                )
                query_features.append(projector.project(gradient)[0])
            query_matrix = torch.stack(query_features)
            for destination, position in enumerate(
                tqdm(train_positions, desc=f"Endpoint-TracIn {checkpoint.name}")
            ):
                latent, hidden = train_encoding(position)
                gradient = mean_simple_gradient(
                    runtime,
                    latent,
                    hidden,
                    timesteps,
                    seed_parts=("endpoint_tracin_train", checkpoint_index, position),
                    timestep_batch_size=args.timestep_batch_size,
                )
                raw, adamw, _, _, reconstructed_lr = project_dual(
                    runtime, gradient, projector, args.max_grad_norm
                )
                if not np.isclose(checkpoint_lr, reconstructed_lr):
                    raise RuntimeError("optimizer LR changed while scoring")
                train = (raw, adamw)
                for transform_slot, feature in enumerate(train):
                    contribution = score_variants(
                        query_matrix, feature.unsqueeze(0)
                    )[..., 0]
                    tracin_scores[transform_slot, :, :, destination] += (
                        checkpoint_lr * contribution.detach().cpu().numpy()
                    )
            tracin_completed = checkpoint_slot

        if did_pollute:
            if checkpoint_index + 1 >= len(checkpoints):
                print(f"[pollute] skip final checkpoint {checkpoint.name}: no next checkpoint")
                pollute_completed = checkpoint_slot
            else:
                # Cache next-checkpoint predictions, then restore current parameters.
                load_raw_checkpoint(runtime, checkpoints[checkpoint_index + 1])
                next_predictions: list[list[torch.Tensor]] = []
                with torch.no_grad():
                    for query, trajectory, hidden in zip(
                        queries, query_trajectories, query_hidden
                    ):
                        current = []
                        for timestamp_slot, timestep in enumerate(timesteps):
                            trajectory_latent = torch.as_tensor(
                                trajectory.latents[timestamp_slot : timestamp_slot + 1],
                                device=device,
                                dtype=runtime.weight_dtype,
                            )
                            current.append(
                                cfg_predictions(
                                    runtime,
                                    trajectory_latent,
                                    hidden,
                                    unconditional_hidden,
                                    [int(timestep)],
                                    query.guidance_scale,
                                ).detach()
                            )
                        next_predictions.append(current)
                load_raw_checkpoint(runtime, checkpoint)
                for timestamp_slot, timestep in enumerate(timesteps):
                    query_features = []
                    for query_slot, (query, trajectory, hidden) in enumerate(
                        zip(queries, query_trajectories, query_hidden)
                    ):
                        trajectory_latent = torch.as_tensor(
                            trajectory.latents[timestamp_slot : timestamp_slot + 1],
                            device=device,
                            dtype=runtime.weight_dtype,
                        )
                        prediction = cfg_predictions(
                            runtime,
                            trajectory_latent,
                            hidden,
                            unconditional_hidden,
                            [int(timestep)],
                            query.guidance_scale,
                        )
                        delta = (next_predictions[query_slot][timestamp_slot] - prediction).detach()
                        delta_norm = delta.float().norm().clamp_min(1e-12)
                        pollute_loss = (prediction.float() * delta.float()).mean()
                        gradient = gradients(pollute_loss, runtime.parameters)
                        feature = projector.project(gradient)[0] / delta_norm
                        query_features.append(feature)
                    query_matrix = torch.stack(query_features)
                    for destination, position in enumerate(
                        tqdm(
                            train_positions,
                            desc=f"Pollute {checkpoint.name} t={int(timestep)}",
                            leave=False,
                        )
                    ):
                        latent, hidden = train_encoding(position)
                        noise = deterministic_noise(
                            latent, "endpoint_pollute", checkpoint_index, int(timestep)
                        )
                        gradient = gradients(
                            simple_loss(runtime, latent, hidden, int(timestep), noise),
                            runtime.parameters,
                        )
                        raw, adamw, _, _, _ = project_dual(
                            runtime, gradient, projector, args.max_grad_norm
                        )
                        for transform_slot, feature in enumerate((raw, adamw)):
                            dots = score_variants(query_matrix, feature.unsqueeze(0))[..., 0]
                            values = dots.detach().cpu().numpy()
                            weighted = checkpoint_lr * values / len(timesteps)
                            pollute_linear[transform_slot, :, :, destination] += weighted
                            pollute_termwise_squared[
                                transform_slot, :, :, destination
                            ] += checkpoint_lr * np.square(values) / len(timesteps)
                            pollute_timestamp_components[
                                timestamp_slot, transform_slot, :, :, destination
                            ] += weighted
                pollute_completed = checkpoint_slot

        metadata = {
            **run_signature,
            "parameter_source": "raw_lora",
            "completed_through_checkpoint_slot": checkpoint_slot,
            "projection": "checkpoint_specific_countsketch",
            "adamw_semantics": "next full AdamW direction from saved moments; LR applied once",
        }
        if "endpoint_tracin" in methods:
            save_result(
                args.output_dir / "endpoint_tracin.npz",
                tracin_scores,
                {
                    **metadata,
                    "method": "endpoint_tracin",
                    "reduction": "linear",
                    "completed_through_checkpoint_slot": tracin_completed,
                },
                metadata_indices,
            )
        if "endpoint_pollute" in methods:
            assert pollute_timestamp_components is not None
            save_pollute_state(
                args.output_dir / "endpoint_pollute_state.npz",
                pollute_linear,
                pollute_termwise_squared,
                pollute_timestamp_components,
                pollute_completed,
                run_signature,
            )
            timestamp_sum_squared = np.zeros(shape, dtype=np.float64)
            for timestamp_slot in range(len(timesteps)):
                component = np.asarray(
                    pollute_timestamp_components[timestamp_slot], dtype=np.float64
                )
                timestamp_sum_squared += np.square(component)
            reductions = {
                "linear": pollute_linear,
                "termwise_squared": pollute_termwise_squared,
                "timestamp_sum_squared": timestamp_sum_squared,
                "total_sum_squared": np.square(pollute_linear),
            }
            for reduction, values in reductions.items():
                save_result(
                    args.output_dir / f"endpoint_pollute_{reduction}.npz",
                    values,
                    {
                        **metadata,
                        "method": "endpoint_pollute",
                        "reduction": reduction,
                        "completed_through_checkpoint_slot": pollute_completed,
                    },
                    metadata_indices,
                )


if __name__ == "__main__":
    main()
