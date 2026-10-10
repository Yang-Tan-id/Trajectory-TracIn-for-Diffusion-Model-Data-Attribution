#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np


def read_partition(path: Path) -> dict[str, int]:
    partitions: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(f"Malformed partition row {line_number}: {line.rstrip()!r}")
            filename, split_text = fields
            split = int(split_text)
            if split not in (0, 1, 2):
                raise ValueError(f"Invalid split {split} on row {line_number}")
            if filename in partitions:
                raise ValueError(f"Duplicate partition filename: {filename}")
            partitions[filename] = split
    if not partitions:
        raise ValueError(f"No partition rows found in {path}")
    return partitions


def select_training_filenames(
    partitions: dict[str, int], seed: int, train_size: int
) -> np.ndarray:
    training = np.asarray(
        sorted(filename for filename, split in partitions.items() if split == 0),
        dtype="U32",
    )
    if train_size <= 0:
        raise ValueError("train_size must be positive")
    if train_size > len(training):
        raise ValueError(
            f"Requested {train_size:,} images but the partition has only {len(training):,} training rows"
        )
    chosen = np.random.default_rng(seed).choice(len(training), size=train_size, replace=False)
    return training[chosen]


def center_crop_resize(image, crop_size: int, image_size: int):
    from PIL import Image

    image = image.convert("RGB")
    width, height = image.size
    if width < crop_size or height < crop_size:
        raise ValueError(
            f"Image is {width}x{height}, smaller than requested {crop_size}x{crop_size} crop"
        )
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    return image.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)


def _zip_members_by_basename(archive: zipfile.ZipFile) -> dict[str, str]:
    members: dict[str, str] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        basename = PurePosixPath(info.filename).name
        if basename.lower().endswith((".jpg", ".jpeg", ".png")):
            if basename in members:
                raise ValueError(f"Archive contains duplicate image basename: {basename}")
            members[basename] = info.filename
    return members


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the fixed 10k CelebA training subset and 5k attribution mask."
    )
    parser.add_argument("--images-zip", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset" / "celeba" / "10000",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--attribution-size", type=int, default=5_000)
    parser.add_argument("--crop-size", type=int, default=140)
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args()

    if args.attribution_size <= 0 or args.attribution_size > args.train_size:
        parser.error("--attribution-size must be in [1, train-size]")
    if args.crop_size <= 0 or args.image_size <= 0:
        parser.error("--crop-size and --image-size must be positive")

    try:
        from PIL import Image
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise SystemExit("Pillow and tqdm are required to prepare CelebA") from exc

    partitions = read_partition(args.partition)
    filenames = select_training_filenames(partitions, args.seed, args.train_size)
    images = np.empty((args.train_size, args.image_size, args.image_size, 3), dtype=np.uint8)

    with zipfile.ZipFile(args.images_zip, "r") as archive:
        members = _zip_members_by_basename(archive)
        missing = [name for name in filenames if name not in members]
        if missing:
            raise FileNotFoundError(
                f"Archive is missing {len(missing)} selected files; first missing: {missing[0]}"
            )
        for index, filename in enumerate(
            tqdm(filenames, desc="Crop 140 and resize 64", unit="image")
        ):
            with archive.open(members[str(filename)], "r") as handle:
                with Image.open(io.BytesIO(handle.read())) as image:
                    processed = center_crop_resize(image, args.crop_size, args.image_size)
                    images[index] = np.asarray(processed, dtype=np.uint8)

    attribution_indices = np.random.default_rng(args.seed).choice(
        args.train_size, size=args.attribution_size, replace=False
    ).astype(np.int64)
    attribution_mask = np.zeros(args.train_size, dtype=np.bool_)
    attribution_mask[attribution_indices] = True

    args.out_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_root / "dataset.npz",
        images=images,
        labels=np.ones((args.train_size, 1), dtype=np.uint8),
        source_filenames=filenames,
        label_names=np.asarray(["face"]),
    )
    np.save(args.out_root / "celeba_10k_source_filenames.npy", filenames)
    np.save(args.out_root / "attribution_5k_indices.npy", attribution_indices)
    np.save(args.out_root / "attribution_5k_mask.npy", attribution_mask)

    metadata = {
        "format_version": 1,
        "source_images_zip": str(args.images_zip.resolve()),
        "source_partition": str(args.partition.resolve()),
        "source_split": "train (partition 0)",
        "selection_seed": args.seed,
        "size": args.train_size,
        "crop": f"center {args.crop_size}x{args.crop_size}",
        "resize": f"{args.image_size}x{args.image_size} bicubic",
        "conditioning": "unconditional",
        "attribution_subset_seed": args.seed,
        "attribution_subset_size": args.attribution_size,
    }
    (args.out_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved {args.train_size:,} images to {args.out_root / 'dataset.npz'}")
    print(
        f"saved {args.attribution_size:,}-point attribution indices and mask to {args.out_root}"
    )


if __name__ == "__main__":
    main()
