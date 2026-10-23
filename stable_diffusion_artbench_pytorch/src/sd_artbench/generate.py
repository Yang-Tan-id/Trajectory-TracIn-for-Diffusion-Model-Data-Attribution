import argparse
import json
from pathlib import Path

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
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.unet.load_attn_procs(args.lora_dir)
    pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.set_progress_bar_config(disable=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    classes = ArtBenchConfig().classes
    for seed in [int(value) for value in args.seeds.split(",") if value.strip()]:
        for label in classes:
            prompt = label_to_prompt(label)
            for image_index in range(args.images_per_class):
                actual_seed = seed + image_index
                generator = torch.Generator(device=pipeline.device).manual_seed(actual_seed)
                image = pipeline(
                    prompt,
                    height=256,
                    width=256,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                ).images[0]
                filename = f"{label}__seed-{actual_seed}.png"
                image.save(args.output_dir / filename)
                records.append(
                    {"file": filename, "label": label, "prompt": prompt, "seed": actual_seed}
                )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

