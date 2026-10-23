"""ReTrac event replay for raw LoRA checkpoints.

The default ``exact_batch`` mode uses the replayed full-batch forward/dropout
graph for per-example gradients.  ``batch1`` is the lower-memory established
pollute-style approximation: it replays noise/timestep events and LR, then
differentiates each selected sample independently with dropout disabled.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..config import ArtBenchConfig
from ..train_lora import IndexedArtBenchDataset, set_attention_dropout
from .optimizer import adamw_full_direction
from .projection import CountSketchProjector, stable_seed
from .runtime import (
    build_runtime,
    encode_image,
    encode_prompt,
    gradients,
    list_checkpoints,
    load_queries,
    load_raw_checkpoint,
    mean_simple_gradient,
    select_timesteps,
)
from .scoring import TRANSFORMS


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
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/attribution-retrac")
    parser.add_argument("--checkpoint-indices", type=parse_ints)
    parser.add_argument("--query-indices", type=parse_ints)
    parser.add_argument("--timestep-count", type=int, default=100)
    parser.add_argument("--proj-dim", type=int, default=32768)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--conditioning-dropout", type=float, default=0.1)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument(
        "--replay-mode", choices=("batch1", "exact_batch"), default="exact_batch"
    )
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-end", type=int)
    parser.add_argument(
        "--max-replay-batches",
        type=int,
        help="Debug/smoke-test limit; omit for the complete checkpoint interval.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def restore_accelerate_rng(checkpoint: Path) -> None:
    candidates = sorted(checkpoint.glob("random_states*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"no random_states*.pkl in {checkpoint}")
    state = torch.load(candidates[0], map_location="cpu")
    if "random_state" in state:
        random.setstate(state["random_state"])
    if "numpy_random_seed" in state:
        np.random.set_state(state["numpy_random_seed"])
    if "torch_manual_seed" in state:
        torch.set_rng_state(state["torch_manual_seed"])
    cuda_state = state.get("torch_cuda_manual_seed")
    if cuda_state is not None and torch.cuda.is_available():
        if isinstance(cuda_state, (list, tuple)):
            torch.cuda.set_rng_state_all(cuda_state)
        else:
            torch.cuda.set_rng_state(cuda_state)


def cosine_lr(base_lr: float, step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_result(path: Path, scores: np.ndarray, indices: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            scores=scores,
            train_metadata_indices=indices,
            transform_names=np.asarray(TRANSFORMS),
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


def main() -> None:
    args = parse_args()
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
    # ReTrac needs a following interval, so the final checkpoint is not a source.
    checkpoint_indices = (
        args.checkpoint_indices
        if args.checkpoint_indices is not None
        else list(range(len(checkpoints) - 1))
    )
    if any(index < 0 or index + 1 >= len(checkpoints) for index in checkpoint_indices):
        raise ValueError("every ReTrac checkpoint must have a following checkpoint")
    queries = load_queries(args.query_manifest)
    if args.query_indices is not None:
        queries = [queries[index] for index in args.query_indices]
    if not queries:
        raise ValueError("at least one query is required")
    timesteps = select_timesteps(args.timestep_count)
    dataset = IndexedArtBenchDataset(
        args.metadata_csv, args.image_root, args.index_path, runtime.tokenizer, 256
    )
    train_end = len(dataset) if args.train_end is None else min(args.train_end, len(dataset))
    if args.train_start < 0 or args.train_start >= train_end:
        raise ValueError(f"invalid training range [{args.train_start}, {train_end})")
    selected_positions = set(range(args.train_start, train_end))
    metadata_indices = np.asarray(
        [int(dataset.frame.iloc[position].name) for position in range(args.train_start, train_end)],
        dtype=np.int64,
    )
    destination_by_position = {
        position: offset for offset, position in enumerate(range(args.train_start, train_end))
    }
    score_shape = (len(TRANSFORMS), len(queries), len(selected_positions))
    run_signature = {
        "model": args.model,
        "rank": args.rank,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "conditioning_dropout": args.conditioning_dropout,
        "attention_dropout": args.attention_dropout,
        "checkpoint_indices": checkpoint_indices,
        "timesteps": timesteps,
        "proj_dim": args.proj_dim,
        "train_positions": [args.train_start, train_end],
        "query_ids": [query.query_id for query in queries],
        "query_manifest": str(args.query_manifest.resolve()),
        "replay_mode": args.replay_mode,
        "max_grad_norm": args.max_grad_norm,
        "debug_max_replay_batches": args.max_replay_batches,
    }
    result_name = "retrac_debug.npz" if args.max_replay_batches is not None else "retrac.npz"
    result_path = args.output_dir / result_name
    scores, completed_slot = load_result(result_path, score_shape, run_signature)
    query_latents = []
    query_hidden = []
    for query in queries:
        with Image.open(query.image_path) as image:
            query_latents.append(encode_image(runtime, image))
        query_hidden.append(encode_prompt(runtime, query.prompt))

    for checkpoint_slot, checkpoint_index in enumerate(checkpoint_indices):
        if checkpoint_slot <= completed_slot:
            print(
                f"[resume] skip ReTrac checkpoint slot {checkpoint_slot} "
                f"(index {checkpoint_index})",
                flush=True,
            )
            continue
        checkpoint = checkpoints[checkpoint_index]
        checkpoint_lr = load_raw_checkpoint(runtime, checkpoint)
        training_state = json.loads(
            (checkpoint / "training_state.json").read_text(encoding="utf-8")
        )
        next_state = json.loads(
            (checkpoints[checkpoint_index + 1] / "training_state.json").read_text(
                encoding="utf-8"
            )
        )
        start_epoch = int(training_state["completed_epoch"])
        end_epoch = int(next_state["completed_epoch"])
        start_step = int(training_state["global_step"])
        replay_arguments = {
            "model": args.model,
            "rank": args.rank,
            "attention_dropout": args.attention_dropout,
            "conditioning_dropout": args.conditioning_dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        }
        for key, replay_value in replay_arguments.items():
            if key in training_state and training_state[key] != replay_value:
                raise ValueError(
                    f"replay argument {key}={replay_value!r} does not match "
                    f"checkpoint value {training_state[key]!r}"
                )
        projector = CountSketchProjector(
            args.proj_dim, stable_seed("sd_artbench", "projection", checkpoint_index)
        )
        runtime.unet.eval()
        set_attention_dropout(runtime.unet, 0.0)
        query_features = []
        for query, latent, hidden in zip(queries, query_latents, query_hidden):
            gradient = mean_simple_gradient(
                runtime,
                latent,
                hidden,
                timesteps,
                seed_parts=("retrac_query", checkpoint_index, query.query_id),
                full_l2_each=True,
            )
            query_features.append(projector.project(gradient)[0])
        query_matrix = torch.stack(query_features)

        # Restore after all query work so replay begins from the exact saved RNG state.
        restore_accelerate_rng(checkpoint)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        updates_per_epoch = math.ceil(
            len(dataloader) / args.gradient_accumulation_steps
        )
        if start_step != start_epoch * updates_per_epoch:
            raise ValueError(
                "replay arguments do not match the saved training run: "
                f"checkpoint has step={start_step} at epoch={start_epoch}, but "
                f"batch_size={args.batch_size} and accumulation="
                f"{args.gradient_accumulation_steps} imply "
                f"step={start_epoch * updates_per_epoch}"
            )
        total_steps = updates_per_epoch * args.epochs
        warmup_steps = math.ceil(total_steps * 0.1)
        expected_checkpoint_lr = cosine_lr(
            args.learning_rate, start_step, total_steps, warmup_steps
        )
        if not np.isclose(checkpoint_lr, expected_checkpoint_lr, rtol=1e-5, atol=1e-10):
            raise ValueError(
                "replay LR schedule does not match the saved optimizer: "
                f"checkpoint lr={checkpoint_lr}, reconstructed lr="
                f"{expected_checkpoint_lr}. Check --epochs and --learning-rate."
            )
        set_attention_dropout(runtime.unet, args.attention_dropout)
        microbatch = 0
        total_microbatches = (end_epoch - start_epoch) * len(dataloader)
        progress = tqdm(
            total=total_microbatches,
            desc=f"ReTrac {checkpoint.name}->{checkpoints[checkpoint_index + 1].name}",
        )
        for replay_epoch in range(start_epoch, end_epoch):
            for batch_index, batch in enumerate(dataloader):
                if (
                    args.max_replay_batches is not None
                    and microbatch >= args.max_replay_batches
                ):
                    break
                positions = [int(value) for value in batch["position"]]
                pixels = batch["pixel_values"].to(device, dtype=runtime.weight_dtype)
                input_ids = batch["input_ids"].to(device)
                # Preserve the original RNG call order from train_lora.py.
                drop_condition = (
                    torch.rand(input_ids.shape[0], device=device)
                    < args.conditioning_dropout
                )
                dropped_ids = input_ids.clone()
                empty_ids = runtime.tokenizer(
                    "",
                    max_length=runtime.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids[0].to(device)
                dropped_ids[drop_condition] = empty_ids
                with torch.no_grad():
                    latents = runtime.vae.encode(pixels).latent_dist.sample()
                    latents = latents * runtime.vae.config.scaling_factor
                    noise = torch.randn_like(latents)
                    batch_timesteps = torch.randint(
                        0,
                        runtime.scheduler.config.num_train_timesteps,
                        (latents.shape[0],),
                        device=device,
                    ).long()
                    noisy = runtime.scheduler.add_noise(latents, noise, batch_timesteps)
                    replay_hidden = runtime.text_encoder(dropped_ids)[0]
                    clean_hidden = runtime.text_encoder(input_ids)[0]

                selected = [slot for slot, position in enumerate(positions) if position in selected_positions]
                losses = None
                if args.replay_mode == "exact_batch":
                    runtime.unet.train()
                    prediction = runtime.unet(
                        noisy, batch_timesteps, replay_hidden
                    ).sample
                    losses = F.mse_loss(
                        prediction.float(), noise.float(), reduction="none"
                    ).mean(dim=(1, 2, 3))
                else:
                    # Consume the original attention-dropout RNG even for batches
                    # without selected examples, then use independent eval gradients.
                    runtime.unet.train()
                    with torch.no_grad():
                        runtime.unet(noisy, batch_timesteps, replay_hidden)
                    runtime.unet.eval()

                # Accelerate flushes a partial accumulation at every epoch end,
                # so update indices must not carry the remainder across epochs.
                update_slot = (
                    replay_epoch * updates_per_epoch
                    + batch_index // args.gradient_accumulation_steps
                )
                event_lr = cosine_lr(
                    args.learning_rate,
                    update_slot,
                    total_steps,
                    warmup_steps,
                )
                for selected_offset, sample_slot in enumerate(selected):
                    if args.replay_mode == "exact_batch":
                        gradient = gradients(
                            losses[sample_slot],
                            runtime.parameters,
                            retain_graph=selected_offset + 1 < len(selected),
                        )
                    else:
                        prediction = runtime.unet(
                            noisy[sample_slot : sample_slot + 1],
                            batch_timesteps[sample_slot : sample_slot + 1],
                            clean_hidden[sample_slot : sample_slot + 1],
                        ).sample
                        loss = F.mse_loss(
                            prediction.float(),
                            noise[sample_slot : sample_slot + 1].float(),
                            reduction="mean",
                        )
                        gradient = gradients(loss, runtime.parameters)
                    raw_feature, _ = projector.project(
                        gradient, full_l2_normalize=True
                    )
                    adamw_direction, _ = adamw_full_direction(
                        runtime.parameters,
                        gradient,
                        runtime.optimizer,
                        max_grad_norm=args.max_grad_norm,
                    )
                    adamw_feature, _ = projector.project(
                        adamw_direction, full_l2_normalize=True
                    )
                    destination = destination_by_position[positions[sample_slot]]
                    train_matrix = torch.stack((raw_feature, adamw_feature))
                    dots = query_matrix @ train_matrix.T
                    scores[:, :, destination] += event_lr * dots.T.detach().cpu().numpy()
                microbatch += 1
                progress.update(1)
            # DataLoader recreates workers/iterator per epoch, matching training.
            if (
                args.max_replay_batches is not None
                and microbatch >= args.max_replay_batches
            ):
                break
        progress.close()
        metadata = {
            **run_signature,
            "method": "retrac",
            "parameter_source": "raw_lora",
            "train_transforms": list(TRANSFORMS),
            "normalization": "full_parameter_l2_per_loss_before_projection",
            "completed_through_checkpoint_slot": checkpoint_slot,
            "projection": "checkpoint_specific_countsketch",
            "score_sign": "positive_gradient_dot; adamw_full_contains_update_minus_sign",
            "checkpoint_lr_observed": checkpoint_lr,
            "complete_interval": args.max_replay_batches is None,
        }
        save_result(result_path, scores, metadata_indices, metadata)


if __name__ == "__main__":
    main()
