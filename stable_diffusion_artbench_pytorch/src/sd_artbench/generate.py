import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from .config import ArtBenchConfig, label_to_prompt


def parse_args() -> argparse.Namespace:
    root = ArtBenchConfig().index_root.parent.parent
    parser = argparse.ArgumentParser(description="Generate ArtBench-2 samples from a LoRA model.")
    parser.add_argument("--model", default="lambdalabs/miniSD-diffusers")
    parser.add_argument("--lora-dir", type=Path, default=root / "outputs" / "artbench2-lora" / "final")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "generated")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--images-per-class", type=int, default=1)
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help=(
            "Custom prompt to generate. Repeat --prompt for multiple prompts. "
            "When omitted, generate the two ArtBench-2 class prompts."
        ),
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument(
        "--replace-flagged",
        action="store_true",
        help=(
            "Replace a safety-flagged or all-black result with the next unused "
            "seed, preserving the requested number of valid query images."
        ),
    )
    parser.add_argument("--max-replacement-attempts", type=int, default=100)
    parser.add_argument(
        "--disable-safety-checker",
        action="store_true",
        help=(
            "Disable the pretrained pipeline safety checker explicitly. Formal "
            "query generation should normally use --replace-flagged instead."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_replacement_attempts < 1:
        raise ValueError("--max-replacement-attempts must be positive")
    requested_seeds = [
        int(value) for value in args.seeds.split(",") if value.strip()
    ]
    if not requested_seeds:
        raise ValueError("--seeds must contain at least one integer")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipeline_kwargs = {"torch_dtype": dtype}
    if args.disable_safety_checker:
        pipeline_kwargs.update(
            safety_checker=None,
            requires_safety_checker=False,
        )
    pipeline = StableDiffusionPipeline.from_pretrained(args.model, **pipeline_kwargs)
    training_timesteps = int(pipeline.scheduler.config.num_train_timesteps)
    if training_timesteps % args.steps:
        raise ValueError(
            f"{training_timesteps} training timesteps are not divisible by "
            f"{args.steps} inference steps"
        )
    timestep_stride = training_timesteps // args.steps
    # Match the CelebA DDIM sampler exactly: DDIM-50 evaluates the U-Net at
    # 999,979,...,19.  Stable-Diffusion scheduler configs commonly carry an
    # offset of 1, which would otherwise produce 981,961,...,1.
    pipeline.scheduler = DDIMScheduler.from_config(
        pipeline.scheduler.config,
        steps_offset=timestep_stride - 1,
    )
    pipeline.unet.load_attn_procs(args.lora_dir)
    pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.set_progress_bar_config(disable=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    prompt_specs: list[tuple[Optional[str], str]]
    if args.prompts:
        prompt_specs = [(None, prompt) for prompt in args.prompts]
    else:
        prompt_specs = [
            (label, label_to_prompt(label)) for label in ArtBenchConfig().classes
        ]

    next_replacement_seed = max(requested_seeds) + args.images_per_class
    used_prompt_seeds: set[tuple[int, int]] = set()
    for seed in requested_seeds:
        for prompt_index, (label, prompt) in enumerate(prompt_specs):
            for image_index in range(args.images_per_class):
                requested_seed = seed + image_index
                actual_seed = requested_seed
                replacement_attempts = 0
                while True:
                    generator = torch.Generator(device=pipeline.device).manual_seed(
                        actual_seed
                    )
                    output = pipeline(
                        prompt,
                        height=256,
                        width=256,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        generator=generator,
                    )
                    image = output.images[0]
                    detected = getattr(output, "nsfw_content_detected", None)
                    safety_flagged = (
                        detected is not None
                        and len(detected) > 0
                        and bool(detected[0])
                    )
                    extrema = image.convert("RGB").getextrema()
                    all_black = all(
                        channel_min == 0 and channel_max == 0
                        for channel_min, channel_max in extrema
                    )
                    if not safety_flagged and not all_black:
                        break
                    if not args.replace_flagged:
                        reason = "safety-flagged" if safety_flagged else "all-black"
                        raise RuntimeError(
                            f"generation produced a {reason} image for "
                            f"prompt={prompt!r}, seed={actual_seed}; rerun with "
                            "--replace-flagged"
                        )
                    replacement_attempts += 1
                    if replacement_attempts > args.max_replacement_attempts:
                        raise RuntimeError(
                            "exhausted replacement attempts for "
                            f"prompt={prompt!r}, requested_seed={requested_seed}"
                        )
                    while (prompt_index, next_replacement_seed) in used_prompt_seeds:
                        next_replacement_seed += 1
                    replacement_reason = (
                        "safety-flagged" if safety_flagged else "all-black"
                    )
                    print(
                        f"[replace] prompt={prompt!r} requested_seed={requested_seed} "
                        f"rejected_seed={actual_seed} reason={replacement_reason} "
                        f"next_seed={next_replacement_seed}",
                        flush=True,
                    )
                    actual_seed = next_replacement_seed
                    next_replacement_seed += 1
                expected_timesteps = [
                    training_timesteps - 1 - timestep_stride * index
                    for index in range(args.steps)
                ]
                actual_timesteps = [
                    int(value) for value in pipeline.scheduler.timesteps.tolist()
                ]
                if actual_timesteps != expected_timesteps:
                    raise RuntimeError(
                        "DDIM sampling schedule mismatch: "
                        f"actual={actual_timesteps}, expected={expected_timesteps}"
                    )
                name = label if label is not None else f"prompt-{prompt_index:02d}"
                filename = f"{name}__seed-{actual_seed}.png"
                image.save(args.output_dir / filename)
                used_prompt_seeds.add((prompt_index, actual_seed))
                records.append(
                    {
                        "file": filename,
                        "label": label,
                        "prompt": prompt,
                        "seed": actual_seed,
                        "requested_seed": requested_seed,
                        "replacement_attempts": replacement_attempts,
                    }
                )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

