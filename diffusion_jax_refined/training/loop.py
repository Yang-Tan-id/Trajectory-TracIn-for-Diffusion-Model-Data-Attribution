from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from diffusers import DDPMPipeline
from tqdm.auto import tqdm


def train_epoch(
    model,
    train_dataloader,
    noise_scheduler,
    optimizer,
    lr_scheduler,
    accelerator,
    args,
    epoch: int,
    global_step: int,
    logger,
) -> int:
    model.train()
    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    progress_bar = tqdm(total=steps_per_epoch, disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in enumerate(train_dataloader):
        clean_images = batch["input"]
        noise = torch.randn(clean_images.shape, device=clean_images.device)
        bsz = clean_images.shape[0]
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=clean_images.device,
        ).long()
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        if step == 0:
            logger.info(f"Noisy images dtype: {noisy_images.dtype}")

        with accelerator.accumulate(model):
            model_output = model(noisy_images, timesteps).sample
            loss = F.mse_loss(model_output, noise)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                save_path = os.path.join(args.save_path, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

        logs = {
            "loss": loss.detach().item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "step": global_step,
        }
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

    progress_bar.close()
    return global_step


def generate_and_log_samples(model, noise_scheduler, accelerator, args, epoch: int, global_step: int) -> None:
    if not accelerator.is_main_process:
        return
    if epoch % args.save_images_epochs != 0 and epoch != args.num_epochs - 1:
        return
    unet = accelerator.unwrap_model(model)
    unet.eval()
    pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
    generator = torch.Generator(device=pipeline.device).manual_seed(42)
    images = pipeline(
        generator=generator,
        batch_size=16,
        num_inference_steps=1000,
        output_type="numpy",
    ).images
    images_processed = (images * 255).round().astype("uint8")
    accelerator.get_tracker("wandb").log(
        {"test_samples": [wandb.Image(img) for img in images_processed], "epoch": epoch},
        step=global_step,
    )


def save_model(model, noise_scheduler, accelerator, args, logger) -> None:
    if not accelerator.is_main_process:
        return
    unet = accelerator.unwrap_model(model)
    unet.eval()
    pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
    pipeline.save_pretrained(args.save_path)
    logger.info(f"Model saved to {args.save_path}")

