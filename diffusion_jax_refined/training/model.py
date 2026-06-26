from __future__ import annotations

import math

import torch
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler


def create_model(config_path: str, logger):
    logger.info(f"Creating UNet2DModel from {config_path}")
    config = UNet2DModel.load_config(config_path)
    config["resnet_time_scale_shift"] = "scale_shift"
    return UNet2DModel.from_config(config)


def set_dropout(model, logger, p: float = 0.1):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = p
            logger.info(f"Set dropout for {name}: {module.p}")
    return model


def setup_training_components(model, args, train_dataloader, logger):
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.95, 0.999),
        weight_decay=args.adam_weight_decay,
        eps=1e-8,
    )
    total_training_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = math.ceil(total_training_steps * 0.1)
    if args.checkpointing_steps == -1:
        args.checkpointing_steps = math.ceil(total_training_steps * 0.01)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=total_training_steps,
    )
    logger.info(f"Total training steps: {total_training_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Checkpointing every {args.checkpointing_steps} steps")
    return noise_scheduler, optimizer, lr_scheduler

