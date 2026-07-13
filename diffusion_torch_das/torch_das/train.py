import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import apply_index_subset, build_dataset
from .utils import default_device, load_unet_from_config, save_args, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a torch diffusers DDPM on CIFAR2 or synthetic data.")
    parser.add_argument("--dataset", default="synthetic", help="synthetic, HF dataset name, HF disk path, or raw cifar-10-batches-py dir")
    parser.add_argument("--dataset-kind", default="synthetic", choices=["synthetic", "cifar2", "cifar10"], help="Controls class filtering")
    parser.add_argument("--index-path", default=None, help="Optional pickle indices into the selected dataset")
    parser.add_argument("--config", default="configs/tiny_unet.json")
    parser.add_argument("--output-dir", default="runs/smoke/ddpm")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--center-crop", action="store_true")
    parser.add_argument("--random-flip", action="store_true")
    parser.add_argument("--synthetic-samples", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--checkpointing-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_args(output_dir / "train_args.json", args)

    dataset = build_dataset(args, split="train", random_flip=args.random_flip)
    dataset = apply_index_subset(dataset, args.index_path, args.max_train_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = load_unet_from_config(args.config).to(device)
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule="linear",
        prediction_type="epsilon",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.95, 0.999), weight_decay=args.weight_decay)
    total_steps = max(1, len(dataloader) * args.num_epochs)
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=math.ceil(total_steps * 0.1), num_training_steps=total_steps)

    global_step = 0
    for epoch in range(args.num_epochs):
        progress = tqdm(dataloader, desc=f"epoch {epoch}")
        model.train()
        for batch in progress:
            clean = batch["input"].to(device)
            noise = torch.randn_like(clean)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (clean.shape[0],), device=device).long()
            noisy = scheduler.add_noise(clean, noise, timesteps)
            pred = model(noisy, timesteps).sample
            loss = F.mse_loss(pred, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}", step=global_step)
            if args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                DDPMPipeline(unet=model, scheduler=scheduler).save_pretrained(output_dir / f"checkpoint-{global_step}")

    DDPMPipeline(unet=model, scheduler=scheduler).save_pretrained(output_dir)
    print(f"saved model to {output_dir}")


if __name__ == "__main__":
    main()
