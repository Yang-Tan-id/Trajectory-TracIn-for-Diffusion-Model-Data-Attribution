import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms


CIFAR10_LABELS = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9,
}
CIFAR2_CLASS_NAMES = ("automobile", "horse")


class ImageTensorDataset(Dataset):
    def __init__(self, images, labels=None, resolution=32, center_crop=True, random_flip=False):
        self.images = images
        self.labels = labels if labels is not None else [None] * len(images)
        ops = [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
        ]
        if random_flip:
            ops.append(transforms.RandomHorizontalFlip())
        ops.extend([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return {"input": self.transform(image.convert("RGB")), "label": self.labels[idx], "index": idx}


def synthetic_dataset(num_samples=32, resolution=32, **kwargs):
    rng = np.random.default_rng(0)
    images = []
    labels = []
    for idx in range(num_samples):
        label = idx % 2
        base = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        if label == 0:
            base[:, :, 0] = 220
            base[8:24, 6:26, 1] = 80
        else:
            base[:, :, 2] = 220
            base[6:26, 14:18, 1] = 180
        noise = rng.integers(0, 30, size=base.shape, dtype=np.uint8)
        images.append(np.clip(base + noise, 0, 255))
        labels.append(label)
    return ImageTensorDataset(images, labels, resolution=resolution, **kwargs)


def _load_cifar_batch(path: Path):
    with path.open("rb") as f:
        batch = pickle.load(f, encoding="latin1")
    data = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = batch.get("labels", batch.get("fine_labels"))
    return data, labels


def local_cifar_dataset(cifar_dir, split="train", classes=None, **kwargs):
    cifar_dir = Path(cifar_dir)
    if split == "train":
        batch_paths = [cifar_dir / f"data_batch_{i}" for i in range(1, 6)]
    elif split in {"test", "val"}:
        batch_paths = [cifar_dir / "test_batch"]
    else:
        raise ValueError(f"Unsupported split: {split}")

    wanted = None if classes is None else {CIFAR10_LABELS[name] for name in classes}
    images, labels = [], []
    for batch_path in batch_paths:
        if not batch_path.exists():
            raise FileNotFoundError(f"Missing CIFAR batch: {batch_path}")
        data, batch_labels = _load_cifar_batch(batch_path)
        for image, label in zip(data, batch_labels):
            if wanted is None or int(label) in wanted:
                images.append(image)
                labels.append(int(label))
    return ImageTensorDataset(images, labels, **kwargs)


def hf_dataset(name_or_path, split="train", classes=None, **kwargs):
    from datasets import load_dataset, load_from_disk

    path = Path(name_or_path)
    ds = load_from_disk(str(path / split)) if path.exists() else load_dataset(name_or_path, split=split)
    label_col = "label"
    image_col = "img" if "img" in ds.column_names else "image"
    if classes is not None:
        wanted = {CIFAR10_LABELS[name] for name in classes}
        ds = ds.filter(lambda x: int(x[label_col]) in wanted)
    return ImageTensorDataset([row[image_col] for row in ds], [int(row[label_col]) for row in ds], **kwargs)


def load_image_folder(folder, **kwargs):
    paths = sorted(Path(folder).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No .png files found in {folder}")
    images = [Image.open(path).convert("RGB") for path in paths]
    return ImageTensorDataset(images, [0] * len(images), **kwargs)


def apply_index_subset(dataset, index_path=None, max_samples=None):
    indices = list(range(len(dataset)))
    if index_path:
        with Path(index_path).open("rb") as f:
            indices = list(pickle.load(f))
    if max_samples is not None:
        indices = indices[:max_samples]
    return Subset(dataset, indices)


def build_dataset(args, split="train", random_flip=False):
    common = {
        "resolution": args.resolution,
        "center_crop": args.center_crop,
        "random_flip": random_flip,
    }
    dataset_kind = getattr(args, "dataset_kind", "synthetic")
    dataset_path = Path(args.dataset)
    if split == "gen" and dataset_path.exists() and dataset_path.is_dir():
        return load_image_folder(dataset_path, **common)
    if args.dataset == "synthetic" or dataset_kind == "synthetic":
        return synthetic_dataset(num_samples=args.synthetic_samples, **common)
    classes = CIFAR2_CLASS_NAMES if dataset_kind == "cifar2" else None
    if dataset_path.exists() and (dataset_path / "data_batch_1").exists():
        return local_cifar_dataset(dataset_path, split=split, classes=classes, **common)
    return hf_dataset(args.dataset, split=split, classes=classes, **common)
