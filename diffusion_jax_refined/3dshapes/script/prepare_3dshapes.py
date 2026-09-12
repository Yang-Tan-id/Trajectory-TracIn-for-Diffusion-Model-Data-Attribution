#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SHAPE_NAMES = ("cube", "cylinder", "sphere", "capsule")
GROUP_COLUMNS = (0, 1, 2, 4)  # floor, wall, object, shape


def _factor_ids(labels: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    ids = np.empty(labels.shape, dtype=np.int16)
    values = []
    for column in range(labels.shape[1]):
        unique, inverse = np.unique(labels[:, column], return_inverse=True)
        values.append(unique)
        ids[:, column] = inverse.astype(np.int16)
    return ids, values


def select_balanced_indices(
    labels: np.ndarray,
    seed: int,
    samples_per_group: int,
    *,
    show_progress: bool = False,
) -> np.ndarray:
    factor_ids, values = _factor_ids(labels)
    expected_cardinalities = (10, 10, 10, 8, 4, 15)
    actual = tuple(len(v) for v in values)
    if actual != expected_cardinalities:
        raise ValueError(f"Unexpected factor cardinalities {actual}; expected {expected_cardinalities}")
    group_id = np.ravel_multi_index(
        tuple(factor_ids[:, col] for col in GROUP_COLUMNS),
        dims=(10, 10, 10, 4),
    )
    order = np.argsort(group_id, kind="stable")
    counts = np.bincount(group_id, minlength=4000)
    if not np.all(counts == 120):
        bad = np.flatnonzero(counts != 120)
        raise ValueError(f"Expected 120 examples in every group; bad groups: {bad[:10].tolist()}")
    offsets = np.concatenate(([0], np.cumsum(counts)))
    rng = np.random.default_rng(seed)
    selected = []
    groups = range(4000)
    if show_progress:
        from tqdm.auto import tqdm

        groups = tqdm(groups, total=4000, desc="Selecting balanced groups", unit="group")
    for group in groups:
        candidates = order[offsets[group] : offsets[group + 1]]
        selected.extend(rng.choice(candidates, size=samples_per_group, replace=False).tolist())
    return np.asarray(selected, dtype=np.int64)


def build_conditions(factor_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Token order is an implementation vocabulary only. Prompt order is erased
    # by multi-hot encoding before the condition embedding is evaluated.
    names = (
        tuple(f"shape_{name}" for name in SHAPE_NAMES)
        + tuple(f"object_hue_{i}" for i in range(10))
        + tuple(f"wall_hue_{i}" for i in range(10))
        + tuple(f"floor_hue_{i}" for i in range(10))
    )
    labels = np.zeros((len(factor_ids), len(names)), dtype=np.uint8)
    rows = np.arange(len(factor_ids))
    labels[rows, factor_ids[:, 4]] = 1
    labels[rows, 4 + factor_ids[:, 2]] = 1
    labels[rows, 14 + factor_ids[:, 1]] = 1
    labels[rows, 24 + factor_ids[:, 0]] = 1
    return labels, np.asarray(names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the balanced 20k 3D Shapes experiment dataset.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the official 3dshapes.h5")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset" / "3dshapes" / "20000",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-group", type=int, default=5)
    parser.add_argument("--attribution-size", type=int, default=5000)
    parser.add_argument(
        "--read-chunk-size",
        type=int,
        default=256,
        help="Number of selected images to read from HDF5 per progress update.",
    )
    args = parser.parse_args()

    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("h5py is required; install/update the repository conda environment first") from exc

    if args.read_chunk_size <= 0:
        parser.error("--read-chunk-size must be positive")

    from tqdm.auto import tqdm

    print(f"[1/4] Reading labels from {args.input}", flush=True)
    with h5py.File(args.input, "r") as source:
        label_source = source["labels"]
        labels_all = np.empty(label_source.shape, dtype=np.float64)
        label_chunk_size = max(1, args.read_chunk_size * 64)
        for start in tqdm(
            range(0, len(label_source), label_chunk_size),
            total=(len(label_source) + label_chunk_size - 1) // label_chunk_size,
            desc="Reading labels",
            unit="chunk",
        ):
            end = min(start + label_chunk_size, len(label_source))
            labels_all[start:end] = label_source[start:end]

        print("[2/4] Selecting 5 samples from each of 4,000 groups", flush=True)
        selected = select_balanced_indices(
            labels_all,
            args.seed,
            args.samples_per_group,
            show_progress=True,
        )

        # h5py requires increasing fancy indices. Read in sorted chunks, then
        # place each chunk back in deterministic group/sample order.
        print(f"[3/4] Reading {len(selected):,} selected RGB images", flush=True)
        sort_order = np.argsort(selected)
        sorted_indices = selected[sort_order]
        image_shape = (len(selected),) + tuple(source["images"].shape[1:])
        images = np.empty(image_shape, dtype=np.uint8)
        for start in tqdm(
            range(0, len(selected), args.read_chunk_size),
            total=(len(selected) + args.read_chunk_size - 1) // args.read_chunk_size,
            desc="Reading selected images",
            unit="chunk",
        ):
            end = min(start + args.read_chunk_size, len(selected))
            images[sort_order[start:end]] = source["images"][sorted_indices[start:end]]

    raw_labels = labels_all[selected]
    factor_ids, factor_values = _factor_ids(raw_labels)
    conditions, label_names = build_conditions(factor_ids)
    if len(selected) != 4000 * args.samples_per_group:
        raise AssertionError("balanced selection has the wrong size")
    if args.attribution_size > len(selected):
        parser.error("--attribution-size exceeds the selected dataset size")
    attribution_indices = np.random.default_rng(args.seed).choice(
        len(selected), size=args.attribution_size, replace=False
    ).astype(np.int64)

    args.out_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[4/4] Compressing and writing dataset.npz to {args.out_root} "
        "(this final step may take several minutes)",
        flush=True,
    )
    np.savez_compressed(
        args.out_root / "dataset.npz",
        images=images,
        labels=conditions,
        raw_labels=raw_labels,
        factor_ids=factor_ids,
        source_indices=selected,
        label_names=label_names,
    )
    np.save(args.out_root / "3dshapes_20k_source_indices.npy", selected)
    np.save(args.out_root / "attribution_5k_indices.npy", attribution_indices)
    metadata = {
        "format_version": 1,
        "source": str(args.input.resolve()),
        "selection_seed": args.seed,
        "samples_per_group": args.samples_per_group,
        "num_groups": 4000,
        "size": len(selected),
        "attribution_subset_seed": args.seed,
        "attribution_subset_size": len(attribution_indices),
        "conditioning": "unordered 34-way multi-hot set",
        "condition_labels": label_names.tolist(),
        "factor_order": ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"],
        "factor_values": [values.tolist() for values in factor_values],
    }
    (args.out_root / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"saved {len(selected)} balanced samples to {args.out_root / 'dataset.npz'}")
    print(f"saved random attribution subset ({len(attribution_indices)}) to {args.out_root / 'attribution_5k_indices.npy'}")


if __name__ == "__main__":
    main()
