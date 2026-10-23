"""Generate fixed-seed Base-to-LoRA checkpoint trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from .config import ArtBenchConfig
from .train_lora import create_lora_layers, set_attention_dropout


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split() if item]


def parse_args() -> argparse.Namespace:
    root = ArtBenchConfig().index_root.parent.parent
    parser = argparse.ArgumentParser(
        description="Render the same query at Base and successive raw LoRA checkpoints."
    )
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--query-indices", type=parse_ints, default=[0])
    parser.add_argument("--model", default="lambdalabs/miniSD-diffusers")
    parser.add_argument(
        "--checkpoint-root", type=Path, default=root / "outputs" / "artbench2-lora"
    )
    parser.add_argument("--checkpoint-indices", type=parse_ints)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "checkpoint-trajectories",
    )
    parser.add_argument(
        "--rank",
        type=int,
        help="LoRA rank override. By default it is inferred from the checkpoint.",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--gif-frame-ms", type=int, default=500)
    parser.add_argument("--skip-gif", action="store_true")
    parser.add_argument("--disable-safety-checker", action="store_true")
    return parser.parse_args()


def checkpoint_epoch(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[-1])
    except ValueError as error:
        raise ValueError(f"malformed checkpoint directory: {path}") from error


def list_checkpoints(root: Path) -> list[Path]:
    checkpoints = sorted(root.glob("checkpoint-epoch-*"), key=checkpoint_epoch)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint-epoch-* directories found in {root}")
    return checkpoints


def find_model_file(checkpoint: Path) -> Path:
    for stem in ("pytorch_model", "model"):
        candidates = sorted(checkpoint.glob(f"{stem}*"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"no saved model state found in {checkpoint}")


def read_lora_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    model_path = find_model_file(checkpoint)
    if model_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(model_path), device="cpu")
    return torch.load(model_path, map_location="cpu")


def infer_lora_rank(checkpoint: Path) -> int:
    state = read_lora_state(checkpoint)
    down_ranks = {
        int(value.shape[0])
        for name, value in state.items()
        if name.endswith("_lora.down.weight") and value.ndim == 2
    }
    up_ranks = {
        int(value.shape[1])
        for name, value in state.items()
        if name.endswith("_lora.up.weight") and value.ndim == 2
    }
    if len(down_ranks) != 1 or down_ranks != up_ranks:
        raise ValueError(
            f"could not infer one consistent LoRA rank from {checkpoint}: "
            f"down={sorted(down_ranks)}, up={sorted(up_ranks)}"
        )
    return down_ranks.pop()


def load_lora_state(lora_layers: torch.nn.Module, checkpoint: Path) -> None:
    state = read_lora_state(checkpoint)
    lora_layers.load_state_dict(state, strict=True)


def load_queries(path: Path, indices: Sequence[int]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("query manifest must contain a non-empty JSON list")
    records = []
    for index in indices:
        if not (0 <= index < len(payload)):
            raise ValueError(f"query index {index} is outside [0, {len(payload) - 1}]")
        item = dict(payload[index])
        item["query_index"] = index
        item["seed"] = int(item["seed"])
        records.append(item)
    return records


def render(
    pipeline: StableDiffusionPipeline,
    *,
    prompt: str,
    seed: int,
    steps: int,
    guidance_scale: float,
    expected_timesteps: Sequence[int],
) -> Image.Image:
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    output = pipeline(
        prompt,
        height=256,
        width=256,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    actual_timesteps = [int(value) for value in pipeline.scheduler.timesteps.tolist()]
    if actual_timesteps != list(expected_timesteps):
        raise RuntimeError(
            f"DDIM schedule mismatch: actual={actual_timesteps}, "
            f"expected={list(expected_timesteps)}"
        )
    detected = getattr(output, "nsfw_content_detected", None)
    if detected is not None and len(detected) and bool(detected[0]):
        raise RuntimeError(
            "a trajectory frame was safety-flagged; exact checkpoint comparisons "
            "cannot replace its seed, so rerun with --disable-safety-checker"
        )
    image = output.images[0].convert("RGB")
    extrema = image.getextrema()
    if all(low == 0 and high == 0 for low, high in extrema):
        raise RuntimeError(
            "a trajectory frame is all black; rerun with --disable-safety-checker"
        )
    return image


def caption_frame(image: Image.Image, label: str, prompt: str, seed: int) -> Image.Image:
    header = 48
    canvas = Image.new("RGB", (image.width, image.height + header), "white")
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), f"{label} | seed={seed}", fill="black")
    prompt_line = prompt if len(prompt) <= 42 else prompt[:39] + "..."
    draw.text((6, 25), prompt_line, fill="black")
    return canvas


def write_visualizations(
    query_dir: Path,
    frames: Sequence[tuple[str, Path]],
    prompt: str,
    seed: int,
    frame_ms: int,
    write_gif: bool,
) -> None:
    labeled = []
    for label, path in frames:
        with Image.open(path) as image:
            labeled.append(caption_frame(image.convert("RGB"), label, prompt, seed))
    if write_gif:
        labeled[0].save(
            query_dir / "trajectory.gif",
            save_all=True,
            append_images=labeled[1:],
            duration=frame_ms,
            loop=0,
            optimize=False,
        )
    else:
        (query_dir / "trajectory.gif").unlink(missing_ok=True)
    columns = 5
    gap = 8
    tile_width, tile_height = labeled[0].size
    rows = (len(labeled) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (
            columns * tile_width + (columns + 1) * gap,
            rows * tile_height + (rows + 1) * gap,
        ),
        (225, 225, 225),
    )
    for slot, image in enumerate(labeled):
        x = gap + (slot % columns) * (tile_width + gap)
        y = gap + (slot // columns) * (tile_height + gap)
        sheet.paste(image, (x, y))
    sheet.save(query_dir / "timeline.jpg", quality=92)


def main() -> None:
    args = parse_args()
    if args.rank is not None and args.rank < 1:
        raise ValueError("--rank must be positive")
    if args.steps < 1 or args.gif_frame_ms < 1:
        raise ValueError("steps and gif-frame-ms must be positive")
    queries = load_queries(args.query_manifest, args.query_indices)
    checkpoints = list_checkpoints(args.checkpoint_root)
    if args.checkpoint_indices is not None:
        if any(index < 0 or index >= len(checkpoints) for index in args.checkpoint_indices):
            raise ValueError(
                f"checkpoint indices must be inside [0, {len(checkpoints) - 1}]"
            )
        checkpoints = [checkpoints[index] for index in args.checkpoint_indices]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipeline_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if args.disable_safety_checker:
        pipeline_kwargs.update(safety_checker=None, requires_safety_checker=False)
    pipeline = StableDiffusionPipeline.from_pretrained(args.model, **pipeline_kwargs)
    training_timesteps = int(pipeline.scheduler.config.num_train_timesteps)
    if training_timesteps % args.steps:
        raise ValueError(
            f"{training_timesteps} training timesteps are not divisible by {args.steps}"
        )
    stride = training_timesteps // args.steps
    expected_timesteps = [training_timesteps - 1 - stride * i for i in range(args.steps)]
    pipeline.scheduler = DDIMScheduler.from_config(
        pipeline.scheduler.config, steps_offset=stride - 1
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    set_attention_dropout(pipeline.unet, 0.0)
    pipeline.unet.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_by_query: dict[int, list[tuple[str, Path]]] = {
        int(query["query_index"]): [] for query in queries
    }
    output_records = []

    for query in tqdm(queries, desc="Base frames"):
        query_index = int(query["query_index"])
        query_dir = args.output_dir / f"query-{query_index:03d}"
        query_dir.mkdir(parents=True, exist_ok=True)
        image = render(
            pipeline,
            prompt=str(query["prompt"]),
            seed=int(query["seed"]),
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            expected_timesteps=expected_timesteps,
        )
        frame_path = query_dir / "frame-000-base.png"
        image.save(frame_path)
        frames_by_query[query_index].append(("Base / epoch 0", frame_path))

    checkpoint_rank = infer_lora_rank(checkpoints[0])
    rank = checkpoint_rank if args.rank is None else args.rank
    if rank != checkpoint_rank:
        raise ValueError(
            f"requested LoRA rank {rank}, but {checkpoints[0]} contains rank "
            f"{checkpoint_rank}"
        )
    print(f"[LoRA] inferred rank={rank} from {checkpoints[0].name}", flush=True)
    lora_layers = create_lora_layers(pipeline.unet, rank).to(
        device=device, dtype=dtype
    )
    for checkpoint_slot, checkpoint in enumerate(checkpoints, start=1):
        load_lora_state(lora_layers, checkpoint)
        epoch = checkpoint_epoch(checkpoint)
        for query in tqdm(
            queries,
            desc=f"{checkpoint.name} [{checkpoint_slot}/{len(checkpoints)}]",
        ):
            query_index = int(query["query_index"])
            query_dir = args.output_dir / f"query-{query_index:03d}"
            image = render(
                pipeline,
                prompt=str(query["prompt"]),
                seed=int(query["seed"]),
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                expected_timesteps=expected_timesteps,
            )
            frame_path = query_dir / f"frame-{checkpoint_slot:03d}-epoch-{epoch:04d}.png"
            image.save(frame_path)
            frames_by_query[query_index].append((f"LoRA / epoch {epoch}", frame_path))

    for query in queries:
        query_index = int(query["query_index"])
        query_dir = args.output_dir / f"query-{query_index:03d}"
        frames = frames_by_query[query_index]
        write_visualizations(
            query_dir,
            frames,
            str(query["prompt"]),
            int(query["seed"]),
            args.gif_frame_ms,
            not args.skip_gif,
        )
        output_records.append(
            {
                "query_index": query_index,
                "prompt": query["prompt"],
                "seed": int(query["seed"]),
                "frame_count": len(frames),
                "timeline": str((query_dir / "timeline.jpg").relative_to(args.output_dir)),
                "gif": (
                    str((query_dir / "trajectory.gif").relative_to(args.output_dir))
                    if not args.skip_gif
                    else None
                ),
            }
        )
    manifest = {
        "model": args.model,
        "query_manifest": str(args.query_manifest.resolve()),
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "rank": rank,
        "checkpoints": [checkpoint.name for checkpoint in checkpoints],
        "timesteps": expected_timesteps,
        "guidance_scale": args.guidance_scale,
        "queries": output_records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[done] queries={len(queries)} frames_per_query={len(checkpoints) + 1} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
