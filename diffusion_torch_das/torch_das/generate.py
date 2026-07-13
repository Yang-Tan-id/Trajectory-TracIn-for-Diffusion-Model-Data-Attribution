import argparse
from pathlib import Path

import torch
from diffusers import DDPMPipeline
from tqdm.auto import tqdm

from .utils import default_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images from a saved DDPM pipeline.")
    parser.add_argument("--model-dir", default="runs/smoke/ddpm")
    parser.add_argument("--output-dir", default="runs/smoke/gen")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipe = DDPMPipeline.from_pretrained(args.model_dir).to(device)

    made = 0
    for start in tqdm(range(0, args.num_images, args.batch_size), desc="generate"):
        bsz = min(args.batch_size, args.num_images - start)
        generators = [torch.Generator(device=device).manual_seed(args.seed * args.num_images + start + i) for i in range(bsz)]
        images = pipe(batch_size=bsz, generator=generators, num_inference_steps=args.num_inference_steps).images
        for image in images:
            image.save(output_dir / f"{made:05d}.png")
            made += 1
    print(f"saved {made} images to {output_dir}")


if __name__ == "__main__":
    main()
