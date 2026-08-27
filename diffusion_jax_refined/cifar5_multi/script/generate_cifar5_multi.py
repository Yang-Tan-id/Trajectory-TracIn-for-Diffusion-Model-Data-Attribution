#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


LABELS = ("bird", "horse", "automobile", "dog", "cat")
POSITIONS = {
    "upper_left": (0, 0),
    "upper_right": (0, 32),
    "lower_left": (32, 0),
    "lower_right": (32, 32),
}


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f, encoding="bytes")


def load_cifar10_by_label(root: Path) -> tuple[list[str], dict[int, np.ndarray], dict[int, np.ndarray]]:
    meta = _load_pickle(root / "batches.meta")
    label_names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in meta[b"label_names"]]
    images_by_label: dict[int, list[np.ndarray]] = {}
    indices_by_label: dict[int, list[int]] = {}
    global_index = 0
    for batch_id in range(1, 6):
        batch = _load_pickle(root / f"data_batch_{batch_id}")
        flat = np.asarray(batch[b"data"], dtype=np.uint8)
        labels = np.asarray(batch[b"labels"], dtype=np.int32)
        images = np.transpose(flat.reshape(-1, 3, 32, 32), (0, 2, 3, 1))
        for img, label in zip(images, labels):
            label = int(label)
            images_by_label.setdefault(label, []).append(img)
            indices_by_label.setdefault(label, []).append(global_index)
            global_index += 1
    return (
        label_names,
        {k: np.stack(v, axis=0) for k, v in images_by_label.items()},
        {k: np.asarray(v, dtype=np.int64) for k, v in indices_by_label.items()},
    )


def generate_dataset(source_root: Path, out_root: Path, size: int, seed: int) -> None:
    label_names, images_by_label, indices_by_label = load_cifar10_by_label(source_root)
    source_label_ids = [label_names.index(name) for name in LABELS]
    rng = np.random.default_rng(seed)

    images = np.zeros((size, 64, 64, 3), dtype=np.uint8)
    labels = np.zeros((size, len(LABELS)), dtype=np.uint8)
    label_ids = np.full((size, 3), -1, dtype=np.int16)
    position_ids = np.full((size, 3), -1, dtype=np.int8)
    source_indices = np.full((size, 3), -1, dtype=np.int64)
    position_names = tuple(POSITIONS)

    for i in range(size):
        chosen_positions = rng.choice(4, size=3, replace=False)
        chosen_local_labels = rng.choice(len(LABELS), size=3, replace=False)
        labels[i, chosen_local_labels] = 1
        label_ids[i] = chosen_local_labels.astype(np.int16)
        position_ids[i] = chosen_positions.astype(np.int8)

        for slot, (pos_id, local_label_id) in enumerate(zip(chosen_positions, chosen_local_labels)):
            source_label_id = source_label_ids[int(local_label_id)]
            pool = images_by_label[source_label_id]
            pick = int(rng.integers(0, len(pool)))
            r0, c0 = POSITIONS[position_names[int(pos_id)]]
            images[i, r0 : r0 + 32, c0 : c0 + 32, :] = pool[pick]
            source_indices[i, slot] = int(indices_by_label[source_label_id][pick])

    out_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_root / "dataset.npz",
        images=images,
        labels=labels,
        label_ids=label_ids,
        position_ids=position_ids,
        source_indices=source_indices,
        label_names=np.asarray(LABELS),
        position_names=np.asarray(position_names),
    )
    metadata = {
        "format_version": 1,
        "size": int(size),
        "seed": int(seed),
        "image_size": 64,
        "quadrant_size": 32,
        "labels": list(LABELS),
        "positions": list(position_names),
        "source_root": str(source_root.resolve()),
        "arrays": {
            "images": "uint8 [N,64,64,3]",
            "labels": "uint8 multi-hot [N,5]",
            "label_ids": "int16 [N,3], local ids into labels",
            "position_ids": "int8 [N,3], ids into positions",
            "source_indices": "int64 [N,3], original CIFAR10 train index",
        },
    }
    (out_root / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"saved {size} cifar5_multi samples to {out_root / 'dataset.npz'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 64x64 CIFAR5 multi-object composite dataset.")
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset" / "cifar10" / "cifar-10-batches-py",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Defaults to diffusion_jax_refined/dataset/cifar5_multi/<size>.",
    )
    args = parser.parse_args()
    out_root = args.out_root
    if out_root is None:
        out_root = Path(__file__).resolve().parents[2] / "dataset" / "cifar5_multi" / str(args.size)
    generate_dataset(args.source_root, out_root, args.size, args.seed)


if __name__ == "__main__":
    main()
