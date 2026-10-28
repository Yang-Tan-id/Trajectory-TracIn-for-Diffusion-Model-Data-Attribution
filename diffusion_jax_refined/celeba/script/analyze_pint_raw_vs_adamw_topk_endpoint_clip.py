#!/usr/bin/env python3
"""Compare endpoint CLIP similarity of Raw- and AdamW-ranked PINT top-k data."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np


CELEBA_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = CELEBA_ROOT.parent
TRANSFORMS = ("raw_gradient", "adamw_full")
VARIANT_DIRECTORIES = {
    "raw": "score",
    "query_l2": "score_query_normalized",
    "train_l2": "score_train_l2_normalized",
    "query_train_l2": "score_query_train_l2_normalized",
}


def parse_ids(text: str) -> list[int]:
    values = [int(value) for value in text.replace(",", " ").split() if value]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("query IDs must be nonempty and unique")
    return values


def prompt_tag(prompt: str) -> str:
    import re

    text = str(prompt).strip().replace(",", "__")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")[:80] or "empty_prompt"


def display_image(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"expected one image, got {image.shape}")
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"expected HWC image, got {image.shape}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]
    if not finite.size:
        raise ValueError("image has no finite pixels")
    if float(finite.min()) < -0.01:
        image = (image + 1.0) / 2.0
    elif float(finite.max()) > 1.01:
        image = image / 255.0
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def endpoint_path(sample_root: Path, checkpoint_stem: str, prompt: str, seed: int) -> Path:
    root = (
        sample_root
        / "cifar"
        / f"prompt_{prompt_tag(prompt)}"
        / f"model_prompted_solo__ckpt_{checkpoint_stem}"
        / f"seed_{seed:06d}"
    )
    for name in ("decoded_final.npy", "final_state.npy"):
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"query endpoint absent: {root}")


def score_directory(
    result_root: Path,
    train_seed: int,
    prompt: str,
    seed: int,
    transform: str,
    reduction: str,
    variant: str,
) -> Path:
    namespace = (
        f"traj_tracin_celeba_endpoint_pollute_{transform}_"
        f"train1x50_query1x50_{reduction}_shape3d_semantics_full10000"
    )
    return (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
        / f"query_{prompt_tag(prompt)}"
        / f"initial_seed_{seed}"
        / namespace
        / VARIANT_DIRECTORIES[variant]
    )


def ranking_scores(scores: np.ndarray, transform: str, reduction: str) -> np.ndarray:
    if reduction == "timestamp_sum_squared":
        return scores
    if reduction == "linear":
        return -scores if transform == "raw_gradient" else scores
    raise ValueError(f"unknown reduction: {reduction}")


def select_topk(directory: Path, transform: str, reduction: str, topk: int) -> np.ndarray:
    scores = np.asarray(np.load(directory / "scores.npy"), dtype=np.float64).reshape(-1)
    indices = np.asarray(np.load(directory / "score_indices.npy"), dtype=np.int64).reshape(-1)
    expected = np.arange(10_000, dtype=np.int64)
    if scores.shape != expected.shape or not np.array_equal(indices, expected):
        raise ValueError(f"score bank is not the ordered full-10k universe: {directory}")
    oriented = ranking_scores(scores, transform, reduction)
    if not np.all(np.isfinite(oriented)):
        raise ValueError(f"non-finite ranking scores: {directory}")
    order = np.argsort(-oriented, kind="stable")[:topk]
    return indices[order]


def batched(values: Sequence[np.ndarray], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def clip_features(images: Sequence[np.ndarray], model_name: str, batch_size: int) -> np.ndarray:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CLIP evaluation requires a CUDA GPU")
    device = torch.device("cuda")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).eval().half().to(device)
    output = []
    for batch in batched(images, batch_size):
        pil = [Image.fromarray(display_image(image), mode="RGB") for image in batch]
        pixels = processor(images=pil, return_tensors="pt")["pixel_values"]
        with torch.inference_mode():
            features = model.get_image_features(pixel_values=pixels.to(device, dtype=torch.float16))
            features = torch.nn.functional.normalize(features.float(), dim=-1)
        output.append(features.cpu().numpy())
    return np.concatenate(output, axis=0)


def label_swap(diff: np.ndarray, permutations: int, seed: int, two_sided: bool) -> float:
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    while completed < permutations:
        size = min(20_000, permutations - completed)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(size, len(diff)))
        values = (signs * diff).mean(axis=1)
        if two_sided:
            extreme += int(np.count_nonzero(np.abs(values) >= abs(observed)))
        else:
            extreme += int(np.count_nonzero(values >= observed))
        completed += size
    return (extreme + 1) / (permutations + 1)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    data_root = REFINE_ROOT / "dataset" / "celeba" / "10000_attr40"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", type=Path, default=CELEBA_ROOT / "queries_attr40_val25_x4.json")
    parser.add_argument("--query-ids", type=parse_ids, default=list(range(72)))
    parser.add_argument("--dataset", type=Path, default=data_root / "dataset.npz")
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--reduction", choices=("timestamp_sum_squared", "linear"), default="timestamp_sum_squared")
    parser.add_argument("--variant", choices=tuple(VARIANT_DIRECTORIES), default="query_train_l2")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--permutations", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not 0 < args.topk <= 10_000:
        parser.error("--topk must be within [1, 10000]")

    payload = json.loads(args.query_file.read_text())
    records = payload["queries"] if isinstance(payload, dict) else payload
    if min(args.query_ids) < 0 or max(args.query_ids) >= len(records):
        parser.error(f"query IDs must be within [0, {len(records) - 1}]")
    result_root = CELEBA_ROOT / "result" / args.experiment
    sample_root = result_root / "sample_ddim_eta0_50"
    output_dir = args.output_dir or (
        result_root / "analysis" /
        f"pint_raw_vs_adamw_{args.reduction}_{args.variant}_top{args.topk}_endpoint_clip"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.dataset, allow_pickle=False) as dataset:
        train_images = np.asarray(dataset["images"])
    if len(train_images) != 10_000:
        raise ValueError(f"expected 10000 training images, got {len(train_images)}")

    checkpoint_stem = f"seed_{args.train_seed}_epoch_{args.epochs:04d}"
    endpoint_images = []
    selections: dict[tuple[int, str], np.ndarray] = {}
    for query_id in args.query_ids:
        record = records[query_id]
        prompt = str(record["prompt"])
        seed = int(record.get("initial_seed", record.get("seed")))
        endpoint_images.append(
            np.load(endpoint_path(sample_root, checkpoint_stem, prompt, seed), allow_pickle=False)
        )
        for transform in TRANSFORMS:
            directory = score_directory(
                result_root, args.train_seed, prompt, seed,
                transform, args.reduction, args.variant,
            )
            selections[(query_id, transform)] = select_topk(
                directory, transform, args.reduction, args.topk
            )
        print(f"[select] Q{query_id} raw+AdamW", flush=True)

    print(f"[clip] encoding {len(train_images)} training images", flush=True)
    train_features = clip_features(train_images, args.clip_model, args.batch_size)
    print(f"[clip] encoding {len(endpoint_images)} endpoints", flush=True)
    endpoint_features = clip_features(endpoint_images, args.clip_model, args.batch_size)

    rows: list[dict[str, object]] = []
    for position, query_id in enumerate(args.query_ids):
        endpoint = endpoint_features[position]
        for transform in TRANSFORMS:
            indices = selections[(query_id, transform)]
            similarities = train_features[indices] @ endpoint
            rows.append(
                {
                    "query_id": query_id,
                    "transform": transform,
                    "reduction": args.reduction,
                    "variant": args.variant,
                    "topk": args.topk,
                    "clip_cosine_mean": float(similarities.mean()),
                    "clip_cosine_median": float(np.median(similarities)),
                    "clip_cosine_std": float(similarities.std()),
                    "clip_cosine_min": float(similarities.min()),
                    "clip_cosine_max": float(similarities.max()),
                }
            )

    by_key = {(int(row["query_id"]), str(row["transform"])): row for row in rows}
    raw = np.asarray([by_key[(qid, "raw_gradient")]["clip_cosine_mean"] for qid in args.query_ids], dtype=np.float64)
    adamw = np.asarray([by_key[(qid, "adamw_full")]["clip_cosine_mean"] for qid in args.query_ids], dtype=np.float64)
    diff = raw - adamw
    lines = [
        "CELEBA PINT TOP-K TRAINING DATA — CLIP SIMILARITY TO FULL-DATA ENDPOINT",
        f"queries={len(args.query_ids)} topk={args.topk} reduction={args.reduction} variant={args.variant}",
        "Higher CLIP cosine means the selected training images are more semantically similar to the endpoint.",
        "",
        f"{'QUERY':<7} {'RAW-GRAD':>12} {'ADAMW':>12} {'RAW-ADAMW':>12}",
        "-" * 47,
    ]
    for query_id, left, right in zip(args.query_ids, raw, adamw, strict=True):
        lines.append(f"Q{query_id:<6d} {left:>12.6f} {right:>12.6f} {left-right:>+12.6f}")
    lines.extend(
        [
            "-" * 47,
            f"MEAN    {raw.mean():>12.6f} {adamw.mean():>12.6f} {diff.mean():>+12.6f}",
            f"MEDIAN  {np.median(raw):>12.6f} {np.median(adamw):>12.6f} {np.median(diff):>+12.6f}",
            f"STD     {raw.std():>12.6f} {adamw.std():>12.6f} {diff.std():>12.6f}",
            f"RAW WINS={int(np.count_nonzero(diff > 0))}/{len(diff)}; ties={int(np.count_nonzero(diff == 0))}",
            (
                f"PAIRED LABEL-SWAP raw-adamw: p1={label_swap(diff, args.permutations, 2026, False):.8g} "
                f"p2={label_swap(diff, args.permutations, 2026, True):.8g}"
            ),
        ]
    )
    report = "\n".join(lines) + "\n"
    write_csv(output_dir / "per_query_endpoint_clip.csv", rows)
    (output_dir / "endpoint_clip_report.txt").write_text(report)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "query_ids": args.query_ids,
                "transforms": list(TRANSFORMS),
                "reduction": args.reduction,
                "variant": args.variant,
                "topk": args.topk,
                "clip_model": args.clip_model,
                "definition": "CLIP cosine between each selected training image and the full-data query endpoint",
            },
            indent=2,
        ) + "\n"
    )
    print("\n" + report, end="")
    print(f"[saved] {output_dir / 'endpoint_clip_report.txt'}")
    print(f"[saved] {output_dir / 'per_query_endpoint_clip.csv'}")


if __name__ == "__main__":
    main()
