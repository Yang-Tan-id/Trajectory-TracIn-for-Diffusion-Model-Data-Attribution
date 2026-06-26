from __future__ import annotations

import argparse
import logging
import math
import os

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers.utils import check_min_version, is_wandb_available

from data import create_dataloader, load_train_dataset, select_dataset_from_index
from loop import save_model, train_epoch
from model import create_model, setup_training_components
from utils import set_all_seeds


check_min_version("0.16.0")


def parse_args():
    parser = argparse.ArgumentParser(description="Unconditional DDPM subset training")
    parser.add_argument("--dataset_name_or_path", type=str, default="cifar10")
    parser.add_argument("--model_config_name_or_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--save_images_epochs", type=int, default=20)
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--logger", type=str, default="wandb")
    parser.add_argument("--mixed_precision", type=str, default="no")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=ProjectConfiguration(total_limit=None),
    )
    logger = get_logger(__name__, log_level="INFO")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"Training arguments: {args}")
    logger.info(accelerator.state, main_process_only=False)
    if not is_wandb_available():
        raise ImportError("Install wandb or set --logger to a supported non-wandb tracker.")
    if accelerator.is_main_process:
        os.makedirs(args.save_path, exist_ok=True)

    model = create_model(args.model_config_name_or_path, logger)
    dataset = load_train_dataset(args, logger)
    dataset = select_dataset_from_index(dataset, args.index_path, logger)
    train_dataloader = create_dataloader(dataset, args, logger)
    noise_scheduler, optimizer, lr_scheduler = setup_training_components(
        model,
        args,
        train_dataloader,
        logger,
    )
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(args.wandb_name, config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    max_train_steps = args.num_epochs * math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    for epoch in range(args.num_epochs):
        global_step = train_epoch(
            model,
            train_dataloader,
            noise_scheduler,
            optimizer,
            lr_scheduler,
            accelerator,
            args,
            epoch,
            global_step,
            logger,
        )
        accelerator.wait_for_everyone()
        if epoch == args.num_epochs - 1:
            save_model(model, noise_scheduler, accelerator, args, logger)

    accelerator.end_training()


if __name__ == "__main__":
    main()

