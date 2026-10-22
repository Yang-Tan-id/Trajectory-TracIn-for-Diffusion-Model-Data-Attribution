#!/usr/bin/env python3
"""Compare original and four full-10k top-1000 CelebA removal models."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from functools import lru_cache
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
    ("endpoint_pollute_adamw_timestamp_query_l2_full10k_pos", "Pollute Q-L2"),
    ("endpoint_pollute_adamw_timestamp_both_l2_full10k_pos", "Pollute Both-L2"),
    ("endpoint_tracin_adamw_linear_query_l2_full10k_neg", "TracIn Q-L2"),
    ("endpoint_tracin_adamw_linear_both_l2_full10k_neg", "TracIn Both-L2"),
)


def parse_ids(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def load_queries(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text())
    records = payload["queries"] if isinstance(payload, dict) else payload
    return {
        int(record.get("query_id", record.get("id", position))): record
        for position, record in enumerate(records)
    }


def query_seed(record: dict) -> int:
    return int(record.get("initial_seed", record.get("seed")))


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


@lru_cache(maxsize=2)
def load_clip(
    model_name: str,
    device: str,
    cache_dir_text: str,
    local_files_only: bool,
):
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as error:
        raise RuntimeError(
            "CLIP evaluation requires torch, Pillow, and transformers in PYTHON_BIN"
        ) from error
    cache_dir = cache_dir_text or None
    processor = CLIPProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model = CLIPModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()
    return torch, processor, model


def evaluate_clip(
    *,
    images: list[np.ndarray],
    prompt: str,
    model_name: str,
    device: str,
    cache_dir: Path | None,
    local_files_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    torch, processor, model = load_clip(
        model_name,
        device,
        str(cache_dir) if cache_dir is not None else "",
        local_files_only,
    )
    pil_images = [
        Image.fromarray(np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8))
        for image in images
    ]
    image_inputs = processor(images=pil_images, return_tensors="pt")
    text_inputs = processor(text=[prompt], return_tensors="pt", padding=True)
    image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        text_features = model.get_text_features(**text_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        original_similarity = image_features @ image_features[0:1].T
        prompt_similarity = image_features @ text_features.T
    return (
        original_similarity[:, 0].detach().cpu().numpy().astype(np.float64),
        prompt_similarity[:, 0].detach().cpu().numpy().astype(np.float64),
    )


def render_query(
    *,
    record: dict,
    query_id: int,
    result_root: Path,
    removal_root: Path,
    output_dir: Path,
    base_checkpoint: Path,
    prefer_device: str,
    dpi: int,
    clip_model: str,
    clip_device: str,
    clip_cache_dir: Path | None,
    clip_local_files_only: bool,
) -> Path:
    seed = query_seed(record)
    prompt = str(record["prompt"])
    reference_root = (
        result_root
        / "sample_ddim_eta0_50"
        / "cifar"
        / f"prompt_{_prompt_path_tag(prompt)}"
        / f"model_prompted_solo__ckpt_{base_checkpoint.stem}"
    )
    query_output = output_dir / f"query_{query_id:03d}_seed_{seed:03d}"
    query_output.mkdir(parents=True, exist_ok=True)

    evaluator = CifarTargetEvaluator(
        code_file=str(LEGACY_ROOT / "DM__training_CELEBA_pixel.py"),
        base_checkpoint=str(base_checkpoint),
        prompt=prompt,
        prefer_device=prefer_device,
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
    reference_raw = np.asarray(evaluator.xt_ref[-1], dtype=np.float32)
    reference = display_image(reference_raw)
    np.save(query_output / "original_endpoint_t000.npy", reference_raw)
    panels: list[tuple[str, np.ndarray, float | None, float | None]] = [
        ("Original", reference, None, None)
    ]
    endpoint_timestep = evaluator.t_seq[-1:]

    for method, title in METHODS:
        metrics_path = find_metrics(removal_root, method, query_id)
        metrics = json.loads(metrics_path.read_text())
        checkpoint = Path(metrics["removal_checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        adapter = evaluator._make_adapter(str(checkpoint))
        generated_raw = np.asarray(
            evaluator._sample_model_space_trajectory(
                adapter,
                seed=seed,
                timesteps_to_save=endpoint_timestep,
            )[-1],
            dtype=np.float32,
        )
        generated = display_image(generated_raw)
        np.save(query_output / f"{method}_endpoint_t000.npy", generated_raw)
        panels.append(
            (
                title,
                generated,
                float(metrics["endpoint_difference"]),
                float(metrics["trajectory_difference"]),
            )
        )
        print(f"[sampled] Q{query_id} {title} checkpoint={checkpoint}", flush=True)

    differences = [np.mean(np.abs(image - reference), axis=-1) for _, image, _, _ in panels[1:]]
    positive = np.concatenate([value.reshape(-1) for value in differences])
    vmax = max(float(np.percentile(positive, 99.0)), 1e-8)
    clip_original, clip_prompt = evaluate_clip(
        images=[image for _, image, _, _ in panels],
        prompt=prompt,
        model_name=clip_model,
        device=clip_device,
        cache_dir=clip_cache_dir,
        local_files_only=clip_local_files_only,
    )
    clip_records = []
    for column, (title, _, endpoint, trajectory) in enumerate(panels):
        clip_records.append(
            {
                "query_id": query_id,
                "seed": seed,
                "method": "original" if column == 0 else METHODS[column - 1][0],
                "title": title,
                "clip_original_cosine": float(clip_original[column]),
                "clip_original_distance": float(1.0 - clip_original[column]),
                "clip_prompt_cosine": float(clip_prompt[column]),
                "endpoint_difference": endpoint,
                "trajectory_difference": trajectory,
            }
        )
    (query_output / "clip_metrics.json").write_text(
        json.dumps(
            {
                "clip_model": clip_model,
                "prompt": prompt,
                "records": clip_records,
            },
            indent=2,
        )
        + "\n"
    )

    figure, axes = plt.subplots(2, len(panels), figsize=(3.15 * len(panels), 6.4))
    heatmap = None
    for column, (title, image, endpoint, trajectory) in enumerate(panels):
        axes[0, column].imshow(image)
        if endpoint is None:
            subtitle = "reference"
        else:
            subtitle = f"end={endpoint:.3e}  traj={trajectory:.3e}"
        axes[0, column].set_title(f"{title}\n{subtitle}", fontsize=9)
        axes[0, column].axis("off")
        if column == 0:
            axes[1, column].imshow(np.zeros(reference.shape[:2]), cmap="magma", vmin=0, vmax=vmax)
            axes[1, column].set_title(
                f"reference\nCLIP(prompt)={clip_prompt[column]:+.3f}", fontsize=9
            )
        else:
            difference = differences[column - 1]
            heatmap = axes[1, column].imshow(difference, cmap="magma", vmin=0, vmax=vmax)
            axes[1, column].set_title(
                f"CLIP(original)={clip_original[column]:+.3f}  "
                f"distance={1.0 - clip_original[column]:.3f}\n"
                f"CLIP(prompt)={clip_prompt[column]:+.3f}  "
                f"pixel MAE={float(difference.mean()):.4f}",
                fontsize=8,
            )
            np.save(query_output / f"{METHODS[column - 1][0]}_pixel_absdiff.npy", difference)
        axes[1, column].axis("off")
    if heatmap is not None:
        colorbar = figure.colorbar(
            heatmap,
            ax=list(axes[1, 1:]),
            fraction=0.025,
            pad=0.02,
        )
        colorbar.set_label("mean absolute RGB difference")
    figure.suptitle(
        f"CelebA Q{query_id} · seed {seed} · full-10K top-1000 removal\n{prompt}",
        fontsize=11,
    )
    figure.subplots_adjust(left=0.02, right=0.94, bottom=0.03, top=0.86, wspace=0.08, hspace=0.2)
    figure_path = query_output / f"q{query_id:03d}_full10k_top1000_four_models.png"
    figure.savefig(figure_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    del evaluator
    gc.collect()
    print(f"[saved] {figure_path}", flush=True)
    return figure_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--prefer-device", default="gpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-device", default="cuda")
    parser.add_argument("--clip-cache-dir", type=Path)
    parser.add_argument("--clip-local-files-only", action="store_true")
    args = parser.parse_args()

    queries = load_queries(args.query_file)
    result_root = CELEBA_ROOT / "result" / args.experiment
    removal_root = result_root / "endpoint_top1000_removal_q0_9"
    output_dir = args.output_dir or (
        result_root / "analysis" / "endpoint_full10k_top1000_four_models"
    )
    base_checkpoint = (
        result_root
        / "model"
        / "prompted_jax"
        / f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt"
    )
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)
    for query_id in parse_ids(args.query_ids):
        render_query(
            record=queries[query_id],
            query_id=query_id,
            result_root=result_root,
            removal_root=removal_root,
            output_dir=output_dir,
            base_checkpoint=base_checkpoint,
            prefer_device=args.prefer_device,
            dpi=args.dpi,
            clip_model=args.clip_model,
            clip_device=args.clip_device,
            clip_cache_dir=args.clip_cache_dir,
            clip_local_files_only=args.clip_local_files_only,
        )
    print(f"[done] figures={output_dir}", flush=True)


if __name__ == "__main__":
    main()
