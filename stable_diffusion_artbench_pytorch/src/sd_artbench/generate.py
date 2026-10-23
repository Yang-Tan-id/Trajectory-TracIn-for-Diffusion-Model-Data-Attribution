import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from .config import ArtBenchConfig, label_to_prompt
from .trajectory import ddim_stride_timesteps, save_query_trajectory


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
    expected_timesteps = ddim_stride_timesteps(
        args.steps,
        training_timesteps=int(pipeline.scheduler.config.num_train_timesteps),
    )
    # Diffusers' default Stable-Diffusion offset produces 981,...,1 for DDIM-50.
    # The existing JAX experiments use exactly 999,979,...,19 instead.
    step_stride = int(pipeline.scheduler.config.num_train_timesteps) // args.steps
    pipeline.scheduler = DDIMScheduler.from_config(
        pipeline.scheduler.config,
        steps_offset=step_stride - 1,
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

    for seed in [int(value) for value in args.seeds.split(",") if value.strip()]:
        for prompt_index, (label, prompt) in enumerate(prompt_specs):
            for image_index in range(args.images_per_class):
                actual_seed = seed + image_index
                generator = torch.Generator(device=pipeline.device).manual_seed(actual_seed)
                trajectory_latents = []
                trajectory_predictions = []
                trajectory_timesteps = []
                scheduler_step = pipeline.scheduler.step

                def recording_step(model_output, timestep, sample, *step_args, **step_kwargs):
                    trajectory_latents.append(sample[0].detach().to("cpu", torch.float16))
                    trajectory_predictions.append(
                        model_output[0].detach().to("cpu", torch.float16)
                    )
                    trajectory_timesteps.append(int(timestep.item()))
                    return scheduler_step(
                        model_output, timestep, sample, *step_args, **step_kwargs
                    )

                pipeline.scheduler.step = recording_step
                try:
                    image = pipeline(
                        prompt,
                        height=256,
                        width=256,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        generator=generator,
                    ).images[0]
                finally:
                    pipeline.scheduler.step = scheduler_step
                actual_timesteps = np.asarray(trajectory_timesteps, dtype=np.int64)
                if not np.array_equal(actual_timesteps, expected_timesteps):
                    raise RuntimeError(
                        "DDIM scheduler did not use the required fixed-stride schedule: "
                        f"actual={actual_timesteps.tolist()} expected="
                        f"{expected_timesteps.tolist()}"
                    )
                name = label if label is not None else f"prompt-{prompt_index:02d}"
                filename = f"{name}__seed-{actual_seed}.png"
                image.save(args.output_dir / filename)
                trajectory_filename = f"{name}__seed-{actual_seed}.trajectory.npz"
                save_query_trajectory(
                    args.output_dir / trajectory_filename,
                    latents=torch.stack(trajectory_latents).numpy(),
                    predicted_noise=torch.stack(trajectory_predictions).numpy(),
                    timesteps=actual_timesteps,
                )
                records.append(
                    {
                        "file": filename,
                        "trajectory_file": trajectory_filename,
                        "trajectory_timesteps": actual_timesteps.tolist(),
                        "trajectory_definition": "ddim_fixed_xt_cfg_predicted_noise",
                        "label": label,
                        "prompt": prompt,
                        "seed": actual_seed,
                        "guidance_scale": args.guidance_scale,
                        "inference_steps": args.steps,
                    }
                )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

