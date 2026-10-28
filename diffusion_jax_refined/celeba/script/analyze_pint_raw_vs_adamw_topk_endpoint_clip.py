#!/usr/bin/env python3
"""Compare endpoint CLIP similarity of Raw- and AdamW-ranked PINT top-k data."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time
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


def parse_variants(text: str) -> list[str]:
    values = [value.strip() for value in text.replace(" ", ",").split(",") if value.strip()]
    unknown = sorted(set(values) - set(VARIANT_DIRECTORIES))
    if not values or len(values) != len(set(values)) or unknown:
        raise argparse.ArgumentTypeError(
            f"variants must be unique members of {tuple(VARIANT_DIRECTORIES)}; unknown={unknown}"
        )
    return values


PROMPT_PATH_TAG_MAX_LENGTH = 150


def prompt_tag(prompt: str) -> str:
    import re

    text = str(prompt).strip().replace(",", "__")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return (
        re.sub(r"_+", "_", text).strip("_")[:PROMPT_PATH_TAG_MAX_LENGTH]
        or "empty_prompt"
    )


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
    method: str,
    transform: str,
    reduction: str,
    variant: str,
) -> Path:
    if method == "pint":
        namespace = (
            f"traj_tracin_celeba_endpoint_pollute_{transform}_"
            f"train1x50_query1x50_{reduction}_shape3d_semantics_full10000"
        )
    elif method == "tracin":
        if reduction != "linear":
            raise ValueError("TracIn endpoint scores require --reduction linear")
        namespace = (
            f"traj_tracin_celeba_endpoint_tracin_{transform}_"
            "train_mean50_query_mean50_linear_shape3d_semantics_full10000"
        )
    else:
        raise ValueError(f"unknown method: {method}")
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


def ranking_scores(
    scores: np.ndarray, method: str, transform: str, reduction: str
) -> np.ndarray:
    if method == "pint" and reduction == "timestamp_sum_squared":
        return scores
    if method == "pint" and reduction == "linear":
        return -scores if transform == "raw_gradient" else scores
    if method == "tracin" and reduction == "linear":
        return scores if transform == "raw_gradient" else -scores
    raise ValueError(
        f"unsupported method/transform/reduction: {method}/{transform}/{reduction}"
    )


def select_topk(
    directory: Path,
    method: str,
    transform: str,
    reduction: str,
    topk: int,
) -> np.ndarray:
    scores = np.asarray(np.load(directory / "scores.npy"), dtype=np.float64).reshape(-1)
    indices = np.asarray(np.load(directory / "score_indices.npy"), dtype=np.int64).reshape(-1)
    expected = np.arange(10_000, dtype=np.int64)
    if scores.shape != expected.shape or not np.array_equal(indices, expected):
        raise ValueError(f"score bank is not the ordered full-10k universe: {directory}")
    oriented = ranking_scores(scores, method, transform, reduction)
    if not np.all(np.isfinite(oriented)):
        raise ValueError(f"non-finite ranking scores: {directory}")
    order = np.argsort(-oriented, kind="stable")[:topk]
    return indices[order]


def batched(values: Sequence[np.ndarray], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def clip_features(
    images: Sequence[np.ndarray], model_name: str, batch_size: int, label: str
) -> np.ndarray:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CLIP evaluation requires a CUDA GPU")
    device = torch.device("cuda")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).eval().half().to(device)
    output = []
    total = len(images)
    started = time.monotonic()
    for batch_number, batch in enumerate(batched(images, batch_size), start=1):
        pil = [Image.fromarray(display_image(image), mode="RGB") for image in batch]
        pixels = processor(images=pil, return_tensors="pt")["pixel_values"]
        with torch.inference_mode():
            features = model.get_image_features(pixel_values=pixels.to(device, dtype=torch.float16))
            features = torch.nn.functional.normalize(features.float(), dim=-1)
        output.append(features.cpu().numpy())
        completed = min(batch_number * batch_size, total)
        elapsed = time.monotonic() - started
        rate = completed / elapsed if elapsed > 0 else 0.0
        eta = (total - completed) / rate if rate > 0 else float("inf")
        print(
            f"[clip progress] {label} images={completed}/{total} "
            f"percent={100.0 * completed / total:.2f} elapsed={elapsed:.1f}s "
            f"eta={eta:.1f}s",
            flush=True,
        )
    return np.concatenate(output, axis=0)


def label_swap(
    diff: np.ndarray,
    permutations: int,
    seed: int,
    two_sided: bool,
    label: str,
) -> float:
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
        if completed % 100_000 == 0 or completed == permutations:
            print(
                f"[label-swap progress] {label} "
                f"permutations={completed}/{permutations} "
                f"percent={100.0 * completed / permutations:.2f}",
                flush=True,
            )
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
    parser.add_argument("--method", choices=("pint", "tracin"), default="pint")
    parser.add_argument("--reduction", choices=("timestamp_sum_squared", "linear"), default="timestamp_sum_squared")
    parser.add_argument("--variant", type=parse_variants, default=["query_train_l2"])
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--permutations", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not 0 < args.topk <= 10_000:
        parser.error("--topk must be within [1, 10000]")
    if args.method == "tracin" and args.reduction != "linear":
        parser.error("--method tracin requires --reduction linear")

    payload = json.loads(args.query_file.read_text())
    records = payload["queries"] if isinstance(payload, dict) else payload
    if min(args.query_ids) < 0 or max(args.query_ids) >= len(records):
        parser.error(f"query IDs must be within [0, {len(records) - 1}]")
    result_root = CELEBA_ROOT / "result" / args.experiment
    sample_root = result_root / "sample_ddim_eta0_50"
    output_dir = args.output_dir or (
        result_root / "analysis" /
        f"{args.method}_raw_vs_adamw_{args.reduction}_"
        f"{'_'.join(args.variant)}_top{args.topk}_endpoint_clip"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.dataset, allow_pickle=False) as dataset:
        train_images = np.asarray(dataset["images"])
    if len(train_images) != 10_000:
        raise ValueError(f"expected 10000 training images, got {len(train_images)}")

    checkpoint_stem = f"seed_{args.train_seed}_epoch_{args.epochs:04d}"
    endpoint_images = []
    selections: dict[tuple[int, str, str], np.ndarray] = {}
    for query_id in args.query_ids:
        record = records[query_id]
        prompt = str(record["prompt"])
        seed = int(record.get("initial_seed", record.get("seed")))
        endpoint_images.append(
            np.load(endpoint_path(sample_root, checkpoint_stem, prompt, seed), allow_pickle=False)
        )
        for variant in args.variant:
            for transform in TRANSFORMS:
                directory = score_directory(
                    result_root, args.train_seed, prompt, seed,
                    args.method, transform, args.reduction, variant,
                )
                selections[(query_id, variant, transform)] = select_topk(
                    directory, args.method, transform, args.reduction, args.topk
                )
        print(
            f"[select] Q{query_id} raw+AdamW variants={','.join(args.variant)}",
            flush=True,
        )

    print(f"[clip] encoding {len(train_images)} training images", flush=True)
    train_features = clip_features(
        train_images, args.clip_model, args.batch_size, "training"
    )
    print(f"[clip] encoding {len(endpoint_images)} endpoints", flush=True)
    endpoint_features = clip_features(
        endpoint_images, args.clip_model, args.batch_size, "endpoints"
    )

    rows: list[dict[str, object]] = []
    for position, query_id in enumerate(args.query_ids):
        endpoint = endpoint_features[position]
        for variant in args.variant:
            for transform in TRANSFORMS:
                indices = selections[(query_id, variant, transform)]
                similarities = train_features[indices] @ endpoint
                rows.append(
                    {
                        "query_id": query_id,
                        "method": args.method,
                        "transform": transform,
                        "reduction": args.reduction,
                        "variant": variant,
                        "topk": args.topk,
                        "clip_cosine_mean": float(similarities.mean()),
                        "clip_cosine_median": float(np.median(similarities)),
                        "clip_cosine_std": float(similarities.std()),
                        "clip_cosine_min": float(similarities.min()),
                        "clip_cosine_max": float(similarities.max()),
                    }
                )

    by_key = {
        (int(row["query_id"]), str(row["variant"]), str(row["transform"])): row
        for row in rows
    }
    lines = [
        f"CELEBA {args.method.upper()} TOP-K TRAINING DATA — CLIP SIMILARITY TO FULL-DATA ENDPOINT",
        (
            f"queries={len(args.query_ids)} topk={args.topk} "
            f"reduction={args.reduction} variants={','.join(args.variant)}"
        ),
        "Higher CLIP cosine means the selected training images are more semantically similar to the endpoint.",
    ]
    for variant_index, variant in enumerate(args.variant):
        raw = np.asarray(
            [by_key[(qid, variant, "raw_gradient")]["clip_cosine_mean"] for qid in args.query_ids],
            dtype=np.float64,
        )
        adamw = np.asarray(
            [by_key[(qid, variant, "adamw_full")]["clip_cosine_mean"] for qid in args.query_ids],
            dtype=np.float64,
        )
        diff = raw - adamw
        lines.extend(
            [
                "",
                f"VARIANT: {variant}",
                f"{'QUERY':<7} {'RAW-GRAD':>12} {'ADAMW':>12} {'RAW-ADAMW':>12}",
                "-" * 47,
            ]
        )
        for query_id, left, right in zip(args.query_ids, raw, adamw, strict=True):
            lines.append(f"Q{query_id:<6d} {left:>12.6f} {right:>12.6f} {left-right:>+12.6f}")
        p1 = label_swap(
            diff, args.permutations, 2026 + variant_index, False,
            f"{variant}/one-sided",
        )
        p2 = label_swap(
            diff, args.permutations, 2026 + variant_index, True,
            f"{variant}/two-sided",
        )
        lines.extend(
            [
                "-" * 47,
                f"MEAN    {raw.mean():>12.6f} {adamw.mean():>12.6f} {diff.mean():>+12.6f}",
                f"MEDIAN  {np.median(raw):>12.6f} {np.median(adamw):>12.6f} {np.median(diff):>+12.6f}",
                f"STD     {raw.std():>12.6f} {adamw.std():>12.6f} {diff.std():>12.6f}",
                f"RAW WINS={int(np.count_nonzero(diff > 0))}/{len(diff)}; ties={int(np.count_nonzero(diff == 0))}",
                f"PAIRED LABEL-SWAP raw-adamw: p1={p1:.8g} p2={p2:.8g}",
            ]
        )
    report = "\n".join(lines) + "\n"
    write_csv(output_dir / "per_query_endpoint_clip.csv", rows)
    (output_dir / "endpoint_clip_report.txt").write_text(report)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "query_ids": args.query_ids,
                "method": args.method,
                "transforms": list(TRANSFORMS),
                "reduction": args.reduction,
                "variants": args.variant,
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
