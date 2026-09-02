#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TARGETS = [
    "endpoint_counterfactual",
    "noise_trajectory",
    "traj_counterfactual",
    "simple_loss",
]


def query_tag(query: str) -> str:
    return query.replace(",", "_").replace(" ", "_")


def lambda_from_parts(parts: tuple[str, ...]) -> str:
    for part in parts:
        if part.startswith("lambda_"):
            return part.replace("lambda_", "")
    return ""


def load_lds_rows(eval_root: Path, namespace: str) -> list[tuple[int, str, str, str, float]]:
    rows = []
    for path in eval_root.glob(f"**/initial_seed_10*/{namespace}/**/das*/**/lds_summary.json"):
        target = path.parent.name
        if target not in TARGETS:
            continue
        with path.open() as handle:
            data = json.load(handle)
        parts = path.parts
        query = next((part for part in parts if part.startswith("query_")), "?")
        seed_text = next(
            (part for part in parts if part.startswith("initial_seed_")),
            "initial_seed_-1",
        )
        lam = lambda_from_parts(parts)
        if not lam:
            for source in data.get("score_sources", []):
                lam = lambda_from_parts(Path(source.get("result_dir", "")).parts)
                if lam:
                    break
        rows.append((int(seed_text.replace("initial_seed_", "")), query, target, lam or "?", float(data["lds_spearman"])))
    return rows


def choose_fixed_lambdas(rows: list[tuple[int, str, str, str, float]]) -> dict[str, str]:
    by_target_lambda = defaultdict(list)
    for _seed, _query, target, lam, lds in rows:
        by_target_lambda[(target, lam)].append(lds)
    fixed = {}
    for target in TARGETS:
        candidates = []
        for (candidate_target, lam), vals in by_target_lambda.items():
            if candidate_target == target and vals:
                candidates.append((sum(vals) / len(vals), len(vals), lam))
        fixed[target] = max(candidates)[2] if candidates else "?"
    return fixed


def image_from_array(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array)
    while image.ndim > 3:
        image = image[0]
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    image = image[..., :3].astype(np.float32)
    if image.min() < 0.0:
        image = (image + 1.0) / 2.0
    elif image.max() > 1.5:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def load_trajectory_frames(root: Path, query: str, seed: int, steps: int) -> list[np.ndarray]:
    tag = query.removeprefix("query_")
    path = (
        root
        / "sample"
        / "cifar"
        / f"prompt_{tag}"
        / "model_prompted_solo__ckpt_seed_42_epoch_0200"
        / f"seed_{seed:06d}"
        / "trajectory_xt.npy"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    trajectory = np.load(path)
    n = trajectory.shape[0]
    indices = np.linspace(0, n - 1, steps, dtype=int)
    return [image_from_array(trajectory[index]) for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=Path("diffusion_jax_refined/cifar5_multi/result/cifar5_multi_exp1"))
    parser.add_argument("--namespace", default="ema_r20_das_10x10_residtrain")
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("diffusion_jax_refined/cifar5_multi/result/cifar5_multi_exp1/random20_das_trajectory_panel.png"))
    args = parser.parse_args()

    rows = load_lds_rows(args.experiment_root / "eval", args.namespace)
    fixed_lambdas = choose_fixed_lambdas(rows)
    lds_by_key = {(seed, query, target, lam): lds for seed, query, target, lam, lds in rows}

    completed = []
    for seed, query in sorted({(seed, query) for seed, query, _target, _lam, _lds in rows}):
        available = {
            target
            for s, q, target, lam, _lds in rows
            if s == seed and q == query and lam == fixed_lambdas.get(target)
        }
        if len(available) == len(TARGETS):
            completed.append((seed, query))
    selected = completed[: args.num_queries]
    if not selected:
        raise RuntimeError("No completed queries found for fixed target lambdas.")

    fig_width = 18
    fig_height = max(2.1, 1.15 * len(selected) + 1.0)
    fig, axes = plt.subplots(
        len(selected),
        args.steps + 1,
        figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": [1] * args.steps + [3.8], "wspace": 0.04, "hspace": 0.14},
    )
    if len(selected) == 1:
        axes = axes[None, :]

    for row_id, (seed, query) in enumerate(selected):
        frames = load_trajectory_frames(args.experiment_root, query, seed, args.steps)
        for col, frame in enumerate(frames):
            ax = axes[row_id, col]
            ax.imshow(frame)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_id == 0:
                ax.set_title(f"{col + 1}", fontsize=7)
            for spine in ax.spines.values():
                spine.set_linewidth(0.25)

        ax = axes[row_id, -1]
        ax.axis("off")
        label = f"seed {seed}\n{query.removeprefix('query_')}"
        lines = [label, ""]
        for target in TARGETS:
            lam = fixed_lambdas.get(target, "?")
            lds = lds_by_key.get((seed, query, target, lam), float("nan"))
            short = {
                "endpoint_counterfactual": "endpoint",
                "noise_trajectory": "noise",
                "traj_counterfactual": "traj",
                "simple_loss": "simple",
            }[target]
            lines.append(f"{short:<8s} λ={lam:<5s} LDS={lds: .3f}")
        ax.text(0.0, 0.5, "\n".join(lines), ha="left", va="center", fontsize=8, family="monospace")

    fig.suptitle(
        f"CIFAR5 random prompted DAS 10x10 | fixed lambda by target | n={len(selected)}",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
