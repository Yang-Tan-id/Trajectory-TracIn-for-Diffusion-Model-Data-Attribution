import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import apply_index_subset, build_dataset
from .utils import default_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Compute projected per-example gradients for DAS-style scoring.")
    parser.add_argument("--model-dir", default="runs/smoke/ddpm")
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--dataset-kind", default="synthetic", choices=["synthetic", "cifar2", "cifar10"])
    parser.add_argument("--dataset-type", default="train", choices=["train", "val", "test", "gen"])
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--output", default="runs/smoke/train_grads.npy")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--center-crop", action="store_true")
    parser.add_argument("--synthetic-samples", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1, help="Kept for DataLoader; gradients are per-example.")
    parser.add_argument("--num-timesteps", type=int, default=2)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def flat_grad(model):
    parts = []
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            parts.append(param.grad.detach().flatten())
    return torch.cat(parts)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = default_device(args.device)
    split = "test" if args.dataset_type in {"val", "test"} else args.dataset_type
    dataset = build_dataset(args, split=split)
    dataset = apply_index_subset(dataset, args.index_path, args.max_samples)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNet2DModel.from_pretrained(Path(args.model_dir) / "unet").to(device).eval()
    scheduler = DDPMScheduler.from_pretrained(args.model_dir, subfolder="scheduler")
    selected = np.linspace(0, scheduler.config.num_train_timesteps - 1, args.num_timesteps, dtype=int)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    generator = torch.Generator(device="cpu").manual_seed(0)
    projector = torch.randn(param_count, args.projection_dim, generator=generator, dtype=torch.float32) / np.sqrt(args.projection_dim)
    projected = np.memmap(args.output, dtype=np.float32, mode="w+", shape=(len(dataset), args.projection_dim))

    for row, batch in enumerate(tqdm(dataloader, desc="grad")):
        clean = batch["input"].to(device)
        emb = torch.zeros(param_count, device=device)
        for t in selected:
            timesteps = torch.full((1,), int(t), device=device, dtype=torch.long)
            set_seed(args.seed * 1000 + int(t))
            noise = torch.randn_like(clean)
            noisy = scheduler.add_noise(clean, noise, timesteps)
            pred = model(noisy, timesteps).sample
            loss = F.mse_loss(pred.float(), noise.float())
            model.zero_grad(set_to_none=True)
            loss.backward()
            grad = flat_grad(model)
            emb += grad / (torch.linalg.norm(grad) + 1e-8)
        emb = (emb / len(selected)).cpu()
        projected[row] = (emb @ projector).numpy()
    projected.flush()
    print(f"saved projected gradients {(len(dataset), args.projection_dim)} to {args.output}")


if __name__ == "__main__":
    main()
