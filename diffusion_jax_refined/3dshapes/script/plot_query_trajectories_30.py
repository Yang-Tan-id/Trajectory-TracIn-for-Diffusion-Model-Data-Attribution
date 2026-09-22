#!/usr/bin/env python3
"""Plot 30 evenly spaced saved DDIM trajectory states for each query."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]


def prompt_tag(prompt: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(prompt).replace(",", "__"))
    return re.sub(r"_+", "_", text).strip("_")[:160] or "empty"


def display_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"expected a singleton batch dimension, got {image.shape}")
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"expected HWC image, got {image.shape}")
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("trajectory image contains no finite pixels")
    if float(finite.min()) < -0.01:
        image = (image + 1.0) / 2.0
    elif float(finite.max()) > 1.01:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def find_sample_dir(
    sample_root: Path,
    prompt: str,
    initial_seed: int,
    checkpoint_stem: str,
) -> Path:
    query_root = sample_root / "cifar" / f"prompt_{prompt_tag(prompt)}"
    exact = (
        query_root
        / f"model_prompted_solo__ckpt_{checkpoint_stem}"
        / f"seed_{initial_seed:06d}"
    )
    if (exact / "trajectory_xt.npy").is_file():
        return exact
    candidates = sorted(
        path.parent
        for path in query_root.glob(
            f"model_*/seed_{initial_seed:06d}/trajectory_xt.npy"
        )
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"trajectory not found; expected {exact}")
    raise FileNotFoundError(
        f"trajectory path is ambiguous for seed {initial_seed}: {candidates}"
    )


def select_positions(length: int, count: int) -> np.ndarray:
    if length < count:
        raise ValueError(f"trajectory has {length} states, fewer than requested {count}")
    positions = np.rint(np.linspace(0, length - 1, count)).astype(np.int64)
    if len(np.unique(positions)) != count:
        raise ValueError(f"could not select {count} unique positions from length {length}")
    return positions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num-states", type=int, default=30)
    parser.add_argument("--columns", type=int, default=10)
    parser.add_argument(
        "--query-file",
        type=Path,
        default=SHAPES_ROOT / "queries_seed_0_9.json",
    )
    parser.add_argument("--sample-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    if args.num_states <= 0 or args.columns <= 0:
        parser.error("--num-states and --columns must be positive")
    if args.num_states % args.columns:
        parser.error("--num-states must be divisible by --columns")

    records = json.loads(args.query_file.read_text())["queries"][:10]
    result_root = SHAPES_ROOT / "result" / args.experiment
    sample_root = args.sample_root or result_root / "sample_ddim_eta0_1000"
    output_dir = args.output_dir or result_root / "eval" / "query_trajectory_30"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = f"seed_{args.train_seed}_epoch_{args.epochs:04d}"
    rows = args.num_states // args.columns

    for query_id, record in enumerate(records):
        prompt = str(record["prompt"])
        initial_seed = int(record["initial_seed"])
        sample_dir = find_sample_dir(
            sample_root, prompt, initial_seed, checkpoint_stem
        )
        trajectory = np.load(sample_dir / "trajectory_xt.npy", mmap_mode="r")
        timesteps = np.asarray(np.load(sample_dir / "trajectory_t.npy"), dtype=np.int64)
        if trajectory.ndim != 5:
            raise ValueError(
                f"expected trajectory [K,B,H,W,C], got {trajectory.shape}: {sample_dir}"
            )
        if len(trajectory) != len(timesteps):
            raise ValueError(
                f"trajectory/timestep mismatch: {len(trajectory)} vs {len(timesteps)}"
            )
        positions = select_positions(len(trajectory), args.num_states)

        figure, axes = plt.subplots(
            rows,
            args.columns,
            figsize=(1.65 * args.columns, 1.85 * rows),
            squeeze=False,
        )
        for axis, position in zip(axes.flat, positions):
            axis.imshow(display_image(trajectory[int(position)]))
            axis.set_title(
                f"t={int(timesteps[position])}\npos={int(position)}",
                fontsize=8,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
        figure.suptitle(
            f"Q{query_id} · seed {initial_seed} · {prompt}",
            fontsize=12,
            fontweight="bold",
        )
        figure.subplots_adjust(
            left=0.01, right=0.99, bottom=0.02, top=0.90, wspace=0.08, hspace=0.32
        )
        output = output_dir / f"Q{query_id}_trajectory_{args.num_states}.png"
        figure.savefig(output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        print(f"[saved] {output}")


if __name__ == "__main__":
    main()
