import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import apply_index_subset, build_dataset
from .utils import default_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Compute DDPM denoising losses for a saved torch model.")
    parser.add_argument("--model-dir", default="runs/smoke/ddpm")
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--dataset-kind", default="synthetic", choices=["synthetic", "cifar2", "cifar10"])
    parser.add_argument("--dataset-type", default="train", choices=["train", "val", "test", "gen"])
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--output", default="runs/smoke/losses.pkl")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--center-crop", action="store_true")
    parser.add_argument("--synthetic-samples", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-timesteps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = default_device(args.device)
    split = "test" if args.dataset_type in {"val", "test"} else args.dataset_type
    dataset = build_dataset(args, split=split)
    dataset = apply_index_subset(dataset, args.index_path, args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = UNet2DModel.from_pretrained(Path(args.model_dir) / "unet").to(device).eval()
    scheduler = DDPMScheduler.from_pretrained(args.model_dir, subfolder="scheduler")
    selected = np.linspace(0, scheduler.config.num_train_timesteps - 1, args.num_timesteps, dtype=int)
    all_losses = []
    for batch in tqdm(dataloader, desc="eval"):
        clean = batch["input"].to(device)
        batch_losses = []
        for t in selected:
            timesteps = torch.full((clean.shape[0],), int(t), device=device, dtype=torch.long)
            set_seed(args.seed * 1000 + int(t))
            noise = torch.randn_like(clean)
            noisy = scheduler.add_noise(clean, noise, timesteps)
            with torch.no_grad():
                pred = model(noisy, timesteps).sample
                loss = F.mse_loss(pred.float(), noise.float(), reduction="none").mean(dim=(1, 2, 3))
            batch_losses.append(loss.cpu().numpy())
        all_losses.append(np.stack(batch_losses, axis=1))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    losses = np.concatenate(all_losses, axis=0)
    with output.open("wb") as f:
        pickle.dump(losses, f)
    print(f"saved losses {losses.shape} to {output}")


if __name__ == "__main__":
    main()
