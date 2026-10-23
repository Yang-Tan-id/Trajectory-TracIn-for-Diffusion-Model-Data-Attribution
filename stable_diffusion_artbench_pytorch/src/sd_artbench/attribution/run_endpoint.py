"""Streaming raw/AdamW Endpoint-TracIn and Endpoint-Pollute for LoRA."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from ..config import ArtBenchConfig
from .autograd import (
    batched_loss_gradients,
    compare_gradient_batches,
    functional_per_example_gradients,
    LinearGradSampleCapture,
    scalar_loop_loss_gradients,
)
from .optimizer import adamw_full_direction, adamw_full_direction_batch
from .projection import CountSketchProjector, stable_seed
from .runtime import (
    AttributionArtBenchDataset,
    build_runtime,
    deterministic_noise,
    encode_image,
    encode_prompt,
    list_checkpoints,
    load_queries,
    load_raw_checkpoint,
    mean_simple_gradient,
    select_timesteps,
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
    parser.add_argument("--timestep-batch-size", type=int, default=16)
    parser.add_argument("--pollute-query-batch-size", type=int, default=8)
    parser.add_argument("--pollute-train-batch-size", type=int, default=8)
    parser.add_argument("--per-example-chunk-size", type=int, default=8)
    parser.add_argument(
        "--per-example-backend",
        choices=("legacy", "torch_func", "scalar_loop", "linear_hooks"),
        default="legacy",
        help=(
            "legacy uses is_grads_batched; torch_func uses functional_call + "
            "vmap(grad); scalar_loop shares a batched forward but uses one "
            "ordinary scalar autograd VJP per example; linear_hooks extracts "
            "exact per-example LoRA Linear gradients from one ordinary backward"
        ),
    )
    parser.add_argument(
        "--profile-pollute-batches",
        type=int,
        default=0,
        help="CUDA-synchronize and time the first N calls of each Pollute stage.",
    )
    parser.add_argument(
        "--debug-compare-gradient-backends",
        action="store_true",
        help=(
            "compare legacy and torch.func gradients before projection for the "
            "first Pollute query chunk and first Pollute train chunk"
        ),
    )
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
    for name in (
        "timestep_batch_size",
        "pollute_query_batch_size",
        "pollute_train_batch_size",
        "per_example_chunk_size",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.profile_pollute_batches < 0:
        raise ValueError("--profile-pollute-batches cannot be negative")
    if args.debug_compare_gradient_backends and args.per_example_backend != "torch_func":
        raise ValueError(
            "--debug-compare-gradient-backends requires --per-example-backend torch_func"
        )
    methods = {item.strip() for item in args.methods.split(",") if item.strip()}
    unknown = methods.difference({"endpoint_tracin", "endpoint_pollute"})
    if unknown:
        raise ValueError(f"unknown endpoint methods: {sorted(unknown)}")
    if not methods:
        raise ValueError("at least one endpoint method is required")
    print(f"[per-example gradients] backend={args.per_example_backend}", flush=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    profile_totals: dict[str, float] = defaultdict(float)
    profile_counts: dict[str, int] = defaultdict(int)
    compared_query_gradients = False
    compared_train_gradients = False

    def print_gradient_comparison(kind: str, diagnostics: list[dict[str, float]]) -> None:
        relative = np.asarray([item["relative_l2"] for item in diagnostics])
        cosine = np.asarray([item["cosine"] for item in diagnostics])
        ratio = np.asarray([item["norm_ratio"] for item in diagnostics])
        print(
            f"[gradient-backend-compare] kind={kind} examples={len(diagnostics)} "
            f"rel_l2_mean={relative.mean():.8e} rel_l2_max={relative.max():.8e} "
            f"cosine_mean={cosine.mean():.10f} cosine_min={cosine.min():.10f} "
            f"norm_ratio_mean={ratio.mean():.8f} "
            f"norm_ratio_range=[{ratio.min():.8f},{ratio.max():.8f}]",
            flush=True,
        )

    def one_loss_gradient(loss: torch.Tensor) -> list[torch.Tensor]:
        values = torch.autograd.grad(
            loss,
            runtime.parameters,
            allow_unused=True,
            retain_graph=True,
        )
        return [
            (
                torch.zeros_like(parameter)
                if value is None
                else value.detach()
            ).unsqueeze(0)
            for parameter, value in zip(runtime.parameters, values)
        ]

    def profile_start(stage: str) -> float | None:
        if profile_counts[stage] >= args.profile_pollute_batches:
            return None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def profile_stop(stage: str, started: float | None) -> None:
        if started is None:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        profile_totals[stage] += time.perf_counter() - started
        profile_counts[stage] += 1

    def print_pollute_profile() -> None:
        for stage in (
            "query_next_forward",
            "query_current_forward",
            "query_vjp",
            "query_projection",
            "train_encode_transfer",
            "train_forward",
            "train_vjp",
            "train_raw_projection",
            "train_adamw",
            "train_adamw_projection",
            "train_scoring",
        ):
            count = profile_counts[stage]
            if count:
                total_ms = profile_totals[stage] * 1000.0
                print(
                    f"[pollute-profile] stage={stage} calls={count} "
                    f"total_ms={total_ms:.3f} avg_ms={total_ms / count:.3f}",
                    flush=True,
                )

    runtime = build_runtime(
        model_name=args.model,
        rank=args.rank,
        device=device,
        mixed_precision=args.mixed_precision,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    linear_capture = (
        LinearGradSampleCapture(runtime.unet, runtime.parameters)
        if args.per_example_backend == "linear_hooks"
        else None
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
    timesteps = select_timesteps(args.timestep_count)
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

    query_latents = []
    query_hidden = []
    for query in queries:
        with Image.open(query.image_path) as image:
            query_latents.append(encode_image(runtime, image))
        query_hidden.append(encode_prompt(runtime, query.prompt))

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
        "pollute_query_batch_size": args.pollute_query_batch_size,
        "pollute_train_batch_size": args.pollute_train_batch_size,
        "per_example_chunk_size": args.per_example_chunk_size,
        "per_example_backend": args.per_example_backend,
        "proj_dim": args.proj_dim,
        "train_positions": [args.train_start, train_end],
        "query_ids": [query.query_id for query in queries],
        "query_manifest": str(args.query_manifest.resolve()),
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
            for query, latent, hidden in zip(queries, query_latents, query_hidden):
                gradient = mean_simple_gradient(
                    runtime,
                    latent,
                    hidden,
                    timesteps,
                    seed_parts=("endpoint_tracin", checkpoint_index, query.query_id),
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
                # Cache batched next-checkpoint predictions on CPU, then restore
                # current parameters.  Query and train examples all use the same
                # checkpoint/timestep noise, matching the established endpoint
                # Pollute definition.
                load_raw_checkpoint(runtime, checkpoints[checkpoint_index + 1])
                next_predictions: list[torch.Tensor] = []
                with torch.no_grad():
                    for timestep in timesteps:
                        prediction_chunks = []
                        shared_noise = deterministic_noise(
                            query_latents[0],
                            "endpoint_pollute",
                            checkpoint_index,
                            int(timestep),
                        )
                        for start in range(
                            0, len(queries), args.pollute_query_batch_size
                        ):
                            profile_started = profile_start("query_next_forward")
                            end = min(
                                start + args.pollute_query_batch_size, len(queries)
                            )
                            latents = torch.cat(query_latents[start:end], dim=0)
                            hidden = torch.cat(query_hidden[start:end], dim=0)
                            noise = shared_noise.expand_as(latents)
                            ts = torch.full(
                                (end - start,),
                                int(timestep),
                                device=device,
                                dtype=torch.long,
                            )
                            noisy = runtime.scheduler.add_noise(latents, noise, ts)
                            prediction_chunks.append(
                                runtime.unet(noisy, ts, hidden).sample.detach().cpu()
                            )
                            profile_stop("query_next_forward", profile_started)
                        next_predictions.append(torch.cat(prediction_chunks, dim=0))
                load_raw_checkpoint(runtime, checkpoint)
                for timestamp_slot, timestep in enumerate(timesteps):
                    shared_query_noise = deterministic_noise(
                        query_latents[0],
                        "endpoint_pollute",
                        checkpoint_index,
                        int(timestep),
                    )
                    query_feature_chunks = []
                    for start in range(
                        0, len(queries), args.pollute_query_batch_size
                    ):
                        profile_started = profile_start("query_current_forward")
                        end = min(start + args.pollute_query_batch_size, len(queries))
                        latents = torch.cat(query_latents[start:end], dim=0)
                        hidden = torch.cat(query_hidden[start:end], dim=0)
                        noise = shared_query_noise.expand_as(latents)
                        ts = torch.full(
                            (end - start,),
                            int(timestep),
                            device=device,
                            dtype=torch.long,
                        )
                        noisy = runtime.scheduler.add_noise(latents, noise, ts)
                        if linear_capture is not None:
                            linear_capture.start(end - start)
                        if args.per_example_backend in {
                            "legacy",
                            "scalar_loop",
                            "linear_hooks",
                        }:
                            prediction = runtime.unet(noisy, ts, hidden).sample
                        else:
                            with torch.no_grad():
                                prediction = runtime.unet(noisy, ts, hidden).sample
                        target = next_predictions[timestamp_slot][start:end].to(
                            device=device, dtype=prediction.dtype
                        )
                        delta = (target - prediction).detach()
                        delta_norms = delta.float().flatten(1).norm(dim=1).clamp_min(1e-12)
                        losses = None
                        if args.per_example_backend in {
                            "legacy",
                            "scalar_loop",
                            "linear_hooks",
                        }:
                            losses = (prediction.float() * delta.float()).flatten(1).mean(
                                dim=1
                            )
                        profile_stop("query_current_forward", profile_started)
                        features = torch.empty(
                            (end - start, args.proj_dim),
                            device=device,
                            dtype=torch.float32,
                        )
                        if args.per_example_backend == "legacy":
                            assert losses is not None
                            gradient_iterator = iter(
                                batched_loss_gradients(
                                    losses,
                                    runtime.parameters,
                                    list(range(end - start)),
                                    chunk_size=args.per_example_chunk_size,
                                )
                            )
                        elif args.per_example_backend == "scalar_loop":
                            assert losses is not None
                            gradient_iterator = iter(
                                scalar_loop_loss_gradients(
                                    losses,
                                    runtime.parameters,
                                    list(range(end - start)),
                                    chunk_size=args.per_example_chunk_size,
                                )
                            )
                        elif args.per_example_backend == "linear_hooks":
                            assert losses is not None
                            assert linear_capture is not None
                            gradient_iterator = iter(
                                linear_capture.loss_gradients(
                                    losses,
                                    chunk_size=args.per_example_chunk_size,
                                )
                            )
                        else:
                            gradient_iterator = iter(
                                functional_per_example_gradients(
                                    runtime.unet,
                                    runtime.parameters,
                                    noisy,
                                    ts,
                                    hidden,
                                    delta,
                                    loss="pollute",
                                    chunk_size=args.per_example_chunk_size,
                                )
                            )
                        legacy_debug_iterator = None
                        loop_debug_gradients = None
                        if (
                            args.debug_compare_gradient_backends
                            and not compared_query_gradients
                        ):
                            debug_prediction = runtime.unet(noisy, ts, hidden).sample
                            debug_losses = (
                                debug_prediction.float() * delta.float()
                            ).flatten(1).mean(dim=1)
                            loop_debug_gradients = one_loss_gradient(debug_losses[0])
                            legacy_debug_iterator = iter(
                                batched_loss_gradients(
                                    debug_losses,
                                    runtime.parameters,
                                    list(range(end - start)),
                                    chunk_size=args.per_example_chunk_size,
                                )
                            )
                        while True:
                            debug_slots = None
                            debug_gradients = None
                            if legacy_debug_iterator is not None:
                                debug_slots, debug_gradients = next(
                                    legacy_debug_iterator
                                )
                            profile_started = profile_start("query_vjp")
                            try:
                                sample_slots, gradient_batch = next(gradient_iterator)
                            except StopIteration:
                                break
                            profile_stop("query_vjp", profile_started)
                            if debug_gradients is not None:
                                assert debug_slots is not None
                                if debug_slots != sample_slots:
                                    raise RuntimeError(
                                        "gradient backend comparison chunk mismatch"
                                    )
                                print_gradient_comparison(
                                    "query_func_vs_legacy",
                                    compare_gradient_batches(
                                        debug_gradients, gradient_batch
                                    ),
                                )
                                assert loop_debug_gradients is not None
                                print_gradient_comparison(
                                    "query_legacy_vs_loop",
                                    compare_gradient_batches(
                                        loop_debug_gradients,
                                        [value[:1] for value in debug_gradients],
                                    ),
                                )
                                print_gradient_comparison(
                                    "query_func_vs_loop",
                                    compare_gradient_batches(
                                        loop_debug_gradients,
                                        [value[:1] for value in gradient_batch],
                                    ),
                                )
                                compared_query_gradients = True
                                legacy_debug_iterator = None
                            profile_started = profile_start("query_projection")
                            projected, _ = projector.project_batch(gradient_batch)
                            rows = torch.as_tensor(
                                sample_slots, device=device, dtype=torch.long
                            )
                            features[rows] = projected / delta_norms[rows, None]
                            profile_stop("query_projection", profile_started)
                        query_feature_chunks.append(features)
                    query_matrix = torch.cat(query_feature_chunks, dim=0)

                    train_progress = tqdm(
                        total=len(train_positions),
                        desc=f"Pollute {checkpoint.name} t={int(timestep)}",
                        leave=False,
                    )
                    for start in range(
                        0, len(train_positions), args.pollute_train_batch_size
                    ):
                        profile_started = profile_start("train_encode_transfer")
                        end = min(
                            start + args.pollute_train_batch_size,
                            len(train_positions),
                        )
                        encoded = [
                            train_encoding(position)
                            for position in train_positions[start:end]
                        ]
                        latents = torch.cat([item[0] for item in encoded], dim=0)
                        hidden = torch.cat([item[1] for item in encoded], dim=0)
                        profile_stop("train_encode_transfer", profile_started)
                        shared_train_noise = deterministic_noise(
                            latents[:1],
                            "endpoint_pollute",
                            checkpoint_index,
                            int(timestep),
                        )
                        noise = shared_train_noise.expand_as(latents)
                        ts = torch.full(
                            (end - start,),
                            int(timestep),
                            device=device,
                            dtype=torch.long,
                        )
                        noisy = runtime.scheduler.add_noise(latents, noise, ts)
                        if runtime.scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif runtime.scheduler.config.prediction_type == "v_prediction":
                            target = runtime.scheduler.get_velocity(latents, noise, ts)
                        else:
                            raise ValueError(
                                "unsupported prediction type "
                                f"{runtime.scheduler.config.prediction_type}"
                            )
                        if linear_capture is not None:
                            linear_capture.start(end - start)
                        if args.per_example_backend in {
                            "legacy",
                            "scalar_loop",
                            "linear_hooks",
                        }:
                            profile_started = profile_start("train_forward")
                            prediction = runtime.unet(noisy, ts, hidden).sample
                            losses = F.mse_loss(
                                prediction.float(), target.float(), reduction="none"
                            ).mean(dim=(1, 2, 3))
                            profile_stop("train_forward", profile_started)
                            if args.per_example_backend == "legacy":
                                gradient_iterator = iter(
                                    batched_loss_gradients(
                                        losses,
                                        runtime.parameters,
                                        list(range(end - start)),
                                        chunk_size=args.per_example_chunk_size,
                                    )
                                )
                            elif args.per_example_backend == "scalar_loop":
                                gradient_iterator = iter(
                                    scalar_loop_loss_gradients(
                                        losses,
                                        runtime.parameters,
                                        list(range(end - start)),
                                        chunk_size=args.per_example_chunk_size,
                                    )
                                )
                            else:
                                assert linear_capture is not None
                                gradient_iterator = iter(
                                    linear_capture.loss_gradients(
                                        losses,
                                        chunk_size=args.per_example_chunk_size,
                                    )
                                )
                        else:
                            gradient_iterator = iter(
                                functional_per_example_gradients(
                                    runtime.unet,
                                    runtime.parameters,
                                    noisy,
                                    ts,
                                    hidden,
                                    target,
                                    loss="mse",
                                    chunk_size=args.per_example_chunk_size,
                                )
                            )
                        legacy_debug_iterator = None
                        loop_debug_gradients = None
                        if (
                            args.debug_compare_gradient_backends
                            and not compared_train_gradients
                        ):
                            debug_prediction = runtime.unet(noisy, ts, hidden).sample
                            debug_losses = F.mse_loss(
                                debug_prediction.float(),
                                target.float(),
                                reduction="none",
                            ).mean(dim=(1, 2, 3))
                            loop_debug_gradients = one_loss_gradient(debug_losses[0])
                            legacy_debug_iterator = iter(
                                batched_loss_gradients(
                                    debug_losses,
                                    runtime.parameters,
                                    list(range(end - start)),
                                    chunk_size=args.per_example_chunk_size,
                                )
                            )
                        while True:
                            debug_slots = None
                            debug_gradients = None
                            if legacy_debug_iterator is not None:
                                debug_slots, debug_gradients = next(
                                    legacy_debug_iterator
                                )
                            profile_started = profile_start("train_vjp")
                            try:
                                sample_slots, gradient_batch = next(gradient_iterator)
                            except StopIteration:
                                break
                            profile_stop("train_vjp", profile_started)

                            if debug_gradients is not None:
                                assert debug_slots is not None
                                if debug_slots != sample_slots:
                                    raise RuntimeError(
                                        "gradient backend comparison chunk mismatch"
                                    )
                                print_gradient_comparison(
                                    "train_func_vs_legacy",
                                    compare_gradient_batches(
                                        debug_gradients, gradient_batch
                                    ),
                                )
                                assert loop_debug_gradients is not None
                                print_gradient_comparison(
                                    "train_legacy_vs_loop",
                                    compare_gradient_batches(
                                        loop_debug_gradients,
                                        [value[:1] for value in debug_gradients],
                                    ),
                                )
                                print_gradient_comparison(
                                    "train_func_vs_loop",
                                    compare_gradient_batches(
                                        loop_debug_gradients,
                                        [value[:1] for value in gradient_batch],
                                    ),
                                )
                                compared_train_gradients = True
                                legacy_debug_iterator = None

                            profile_started = profile_start("train_raw_projection")
                            raw, _ = projector.project_batch(gradient_batch)
                            profile_stop("train_raw_projection", profile_started)

                            profile_started = profile_start("train_adamw")
                            adamw_directions, reconstructed_lr = (
                                adamw_full_direction_batch(
                                    runtime.parameters,
                                    gradient_batch,
                                    runtime.optimizer,
                                    max_grad_norm=args.max_grad_norm,
                                )
                            )
                            profile_stop("train_adamw", profile_started)

                            profile_started = profile_start("train_adamw_projection")
                            adamw, _ = projector.project_batch(adamw_directions)
                            profile_stop("train_adamw_projection", profile_started)
                            if not np.isclose(checkpoint_lr, reconstructed_lr):
                                raise RuntimeError("optimizer LR changed while scoring")
                            profile_started = profile_start("train_scoring")
                            destinations = [start + slot for slot in sample_slots]
                            for transform_slot, feature in enumerate((raw, adamw)):
                                dots = score_variants(query_matrix, feature)
                                values_batch = dots.detach().cpu().numpy()
                                for local_slot, destination in enumerate(destinations):
                                    values = values_batch[..., local_slot]
                                    weighted = checkpoint_lr * values / len(timesteps)
                                    pollute_linear[
                                        transform_slot, :, :, destination
                                    ] += weighted
                                    pollute_termwise_squared[
                                        transform_slot, :, :, destination
                                    ] += (
                                        checkpoint_lr
                                        * np.square(values)
                                        / len(timesteps)
                                    )
                                    pollute_timestamp_components[
                                        timestamp_slot,
                                        transform_slot,
                                        :,
                                        :,
                                        destination,
                                    ] += weighted
                            profile_stop("train_scoring", profile_started)
                        train_progress.update(end - start)
                    train_progress.close()
                pollute_completed = checkpoint_slot
                print_pollute_profile()

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
