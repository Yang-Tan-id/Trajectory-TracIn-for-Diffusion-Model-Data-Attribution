#!/usr/bin/env python3
"""Render the same-query endpoints from four CelebA top-500 removal models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CELEBA_ROOT = SCRIPT_DIR.parent
REFINED_ROOT = CELEBA_ROOT.parent
LEGACY_ROOT = REFINED_ROOT / "legacy_jax"
for path in (CELEBA_ROOT, REFINED_ROOT, LEGACY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_config_attr40 import DATA_ROOT, _prompt_path_tag
from LDS.DM_cifar_lds import CifarTargetEvaluator


METHODS = (
    ("endpoint_pollute_adamw_timestamp_raw_pos", "Pollute raw"),
    ("endpoint_pollute_adamw_timestamp_both_l2_pos", "Pollute both-L2"),
    ("endpoint_tracin_adamw_linear_raw_neg", "Endpoint-TracIn raw"),
    ("retrac_adamw_query_train_l2_neg", "ReTrac both-L2"),
)


def load_query(path: Path, query_id: int) -> dict:
    payload = json.loads(path.read_text())
    records = payload["queries"] if isinstance(payload, dict) else payload
    for position, record in enumerate(records):
        record_id = int(record.get("query_id", record.get("id", position)))
        if record_id == query_id:
            return record
    raise KeyError(f"query {query_id} is absent from {path}")


def query_seed(record: dict) -> int:
    if "initial_seed" in record:
        return int(record["initial_seed"])
    return int(record["seed"])


def find_metrics(removal_root: Path, method: str, query_id: int) -> Path:
    matches = sorted(
        (removal_root / method).glob(
            f"query_{query_id:03d}_seed_*/counterfactual_metrics.json"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one metric for {method} Q{query_id}, found {len(matches)}"
        )
    return matches[0]


def display_image(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (
        1,
        3,
        4,
    ):
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3:
        raise ValueError(f"expected image-shaped endpoint, found {image.shape}")
    image = image.astype(np.float32)
    if float(np.nanmin(image)) < -0.05:
        image = (image + 1.0) / 2.0
    elif float(np.nanmax(image)) > 1.5:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-id", type=int, required=True)
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--prefer-device", default="gpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    record = load_query(args.query_file, args.query_id)
    seed = query_seed(record)
    prompt = str(record["prompt"])
    result_root = CELEBA_ROOT / "result" / args.experiment
    removal_root = result_root / "endpoint_top500_removal_q0_9"
    base_checkpoint = (
        result_root
        / "model"
        / "prompted_jax"
        / f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt"
    )
    reference_root = (
        result_root
        / "sample_ddim_eta0_50"
        / "cifar"
        / f"prompt_{_prompt_path_tag(prompt)}"
        / f"model_prompted_solo__ckpt_{base_checkpoint.stem}"
    )
    output_dir = args.output_dir or (
        result_root / "analysis" / "endpoint_top500_removal_generated"
    )
    query_output = output_dir / f"query_{args.query_id:03d}_seed_{seed:03d}"
    query_output.mkdir(parents=True, exist_ok=True)

    evaluator = CifarTargetEvaluator(
        code_file=str(LEGACY_ROOT / "DM__training_CELEBA_pixel.py"),
        base_checkpoint=str(base_checkpoint),
        prompt=prompt,
        prefer_device=args.prefer_device,
        data_root=DATA_ROOT,
        target_function="endpoint_counterfactual",
        sample_root=str(reference_root),
        sample_seed=seed,
        sample_index=0,
        max_trajectory_steps=None,
        trajectory_reduction="mean",
        trajectory_projection=None,
        simple_loss_timesteps=[0],
        simple_loss_noise_seeds=None,
        simple_loss_num_mc=1,
        simple_loss_mc_seed=0,
        trajectory_sampler="ddim_eta0",
    )

    reference = np.asarray(evaluator.xt_ref[-1], dtype=np.float32)
    np.save(query_output / "baseline_endpoint_t000.npy", reference)
    panels = [("Baseline", reference, None, None)]
    endpoint_timestep = evaluator.t_seq[-1:]

    for method, title in METHODS:
        metrics_path = find_metrics(removal_root, method, args.query_id)
        metrics = json.loads(metrics_path.read_text())
        checkpoint = Path(metrics["removal_checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        adapter = evaluator._make_adapter(str(checkpoint))
        generated = evaluator._sample_model_space_trajectory(
            adapter,
            seed=seed,
            timesteps_to_save=endpoint_timestep,
        )[-1]
        generated = np.asarray(generated, dtype=np.float32)
        np.save(query_output / f"{method}_endpoint_t000.npy", generated)
        panels.append(
            (
                title,
                generated,
                float(metrics["endpoint_difference"]),
                float(metrics["trajectory_difference"]),
            )
        )
        print(f"[sampled] Q{args.query_id} {title} checkpoint={checkpoint}", flush=True)

    figure, axes = plt.subplots(1, len(panels), figsize=(3.25 * len(panels), 3.9))
    for axis, (title, image, endpoint, trajectory) in zip(axes, panels, strict=True):
        axis.imshow(display_image(image))
        if endpoint is None:
            subtitle = "reference"
        else:
            subtitle = f"endpoint={endpoint:.3e}\ntrajectory={trajectory:.3e}"
        axis.set_title(f"{title}\n{subtitle}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.suptitle(f"CelebA Q{args.query_id} · seed {seed} · top-500 removal\n{prompt}")
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure_path = query_output / f"q{args.query_id:03d}_top500_removal_endpoints.png"
    figure.savefig(figure_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"[saved] {figure_path}")
    print(f"[saved endpoints] {query_output}")


if __name__ == "__main__":
    main()
