from __future__ import annotations

import os
import pickle

import torch
from datasets import load_dataset, load_from_disk
from torchvision import transforms


def load_train_dataset(args, logger):
    if os.path.exists(args.dataset_name_or_path):
        logger.info(f"Loading local dataset from {args.dataset_name_or_path}")
        return load_from_disk(os.path.join(args.dataset_name_or_path, "train"))
    logger.info(f"Loading dataset by name: {args.dataset_name_or_path}")
    return load_dataset(args.dataset_name_or_path, split="train")


def select_dataset_from_index(dataset, index_path: str, logger):
    logger.info(f"Loading subset indices from {index_path}")
    with open(index_path, "rb") as handle:
        sub_idx = pickle.load(handle)
    selected = dataset.select(sub_idx)
    logger.info(f"Dataset size after filtering: {len(selected)}")
    return selected


def create_dataloader(dataset, args, logger):
    augmentations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["img"]]
        return {"input": images}

    dataset.set_transform(transform_images)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    logger.info("Created dataloader")
    return dataloader

