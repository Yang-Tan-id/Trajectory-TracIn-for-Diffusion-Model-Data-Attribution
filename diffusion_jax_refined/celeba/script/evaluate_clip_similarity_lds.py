#!/usr/bin/env python3
"""Evaluate CelebA LDS using endpoint-to-training-image CLIP similarity scores."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Sequence

import numpy as np


CELEBA_ROOT = Path(__file__).resolve().parents[1]
REFINED_ROOT = CELEBA_ROOT.parent
REPO_ROOT = REFINED_ROOT.parent
for root in (CELEBA_ROOT, REFINED_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from dataset_config_attr40 import _prompt_path_tag
from diffusion_jax_refined.celeba.script.run_endpoint_cached_lds import (
    build_membership,
    load_targets,
    spearman,
)


TARGET_ORDER = (
    "endpoint_contarfactual",
    "traj_contarfactual",
    "noise_trajectory",
    "simple_loss_50x1",
    "simple_loss",
)


def parse_ids(text: str) -> list[int]:
    values: list[int] = []
    for token in text.replace(",", " ").split():
        if "-" in token:
            left, right = token.split("-", 1)
            values.extend(range(int(left), int(right) + 1))
        else:
            values.append(int(token))
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("query IDs must be nonempty and unique")
    return values


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


def endpoint_path(
    sample_root: Path, checkpoint_stem: str, prompt: str, seed: int
) -> Path:
    root = (
        sample_root
        / "cifar"
        / f"prompt_{_prompt_path_tag(prompt)}"
        / f"model_prompted_solo__ckpt_{checkpoint_stem}"
        / f"seed_{seed:06d}"
    )
    for filename in ("decoded_final.npy", "final_state.npy"):
        candidate = root / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"query endpoint absent: {root}")


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
        raise RuntimeError("CLIP feature extraction requires a CUDA GPU")
    device = torch.device("cuda")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).eval().half().to(device)
    output: list[np.ndarray] = []
    started = time.monotonic()
    for batch_number, batch in enumerate(batched(images, batch_size), start=1):
        pil_images = [Image.fromarray(display_image(image), mode="RGB") for image in batch]
        pixels = processor(images=pil_images, return_tensors="pt")["pixel_values"]
        with torch.inference_mode():
            features = model.get_image_features(
                pixel_values=pixels.to(device, dtype=torch.float16)
            )
            features = torch.nn.functional.normalize(features.float(), dim=-1)
        output.append(features.cpu().numpy())
        completed = min(batch_number * batch_size, len(images))
        elapsed = time.monotonic() - started
        rate = completed / elapsed if elapsed else 0.0
        eta = (len(images) - completed) / rate if rate else float("inf")
        print(
            f"[clip] {label} {completed}/{len(images)} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )
    return np.concatenate(output, axis=0).astype(np.float32)


def load_or_compute_features(
    cache_path: Path,
    train_images: np.ndarray,
    endpoint_images: Sequence[np.ndarray],
    query_ids: Sequence[int],
    model_name: str,
    batch_size: int,
    recompute: bool,
) -> tuple[np.ndarray, np.ndarray]:
    expected_ids = np.asarray(query_ids, dtype=np.int64)
    if cache_path.is_file() and not recompute:
        with np.load(cache_path, allow_pickle=False) as cache:
            if not np.array_equal(cache["query_ids"], expected_ids):
                raise ValueError(f"cached query IDs differ: {cache_path}")
            if str(cache["clip_model"].item()) != model_name:
                raise ValueError(f"cached CLIP model differs: {cache_path}")
            train = np.asarray(cache["train_clip"], dtype=np.float32)
            endpoint = np.asarray(cache["endpoint_clip"], dtype=np.float32)
        print(f"[cache] reused CLIP features: {cache_path}", flush=True)
        return train, endpoint

    train = clip_features(train_images, model_name, batch_size, "training images")
    endpoint = clip_features(endpoint_images, model_name, batch_size, "query outputs")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            query_ids=expected_ids,
            clip_model=np.asarray(model_name),
            train_clip=train,
            endpoint_clip=endpoint,
        )
    os.replace(temporary, cache_path)
    print(f"[cache] saved CLIP features: {cache_path}", flush=True)
    return train, endpoint


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--query-file",
        type=Path,
        default=CELEBA_ROOT / "queries_attr40_val25_x4.json",
    )
    parser.add_argument("--query-ids", type=parse_ids, default=list(range(60)))
    parser.add_argument("--true-f-csv", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--recompute-features", action="store_true")
    args = parser.parse_args()

    result_root = CELEBA_ROOT / "result" / args.experiment
    args.dataset = args.dataset or (
        REFINED_ROOT / "dataset" / "celeba" / "10000_attr40" / "dataset.npz"
    )
    args.true_f_csv = args.true_f_csv or (
        result_root
        / "query100_lds_csv_ddim50"
        / "celeba_attr40_query100_lds_true_f_long.csv"
    )
    args.output_dir = args.output_dir or (
        result_root / "analysis" / "clip_similarity_score_lds_q0_q59"
    )
    args.feature_cache = args.feature_cache or (
        args.output_dir / "clip_features.npz"
    )

    payload = json.loads(args.query_file.read_text(encoding="utf-8"))
    records = payload["queries"] if isinstance(payload, dict) else payload
    records_by_id = {
        int(record.get("query_id", position)): record
        for position, record in enumerate(records)
    }
    absent = sorted(set(args.query_ids) - set(records_by_id))
    if absent:
        parser.error(f"query IDs absent from manifest: {absent}")

    with np.load(args.dataset, allow_pickle=False) as dataset:
        train_images = np.asarray(dataset["images"])
    if len(train_images) != 10_000:
        raise ValueError(f"expected 10000 training images, found {len(train_images)}")

    checkpoint_stem = f"seed_{args.train_seed}_epoch_{args.epochs:04d}"
    sample_root = result_root / "sample_ddim_eta0_50"
    endpoint_images = []
    for query_id in args.query_ids:
        record = records_by_id[query_id]
        endpoint_images.append(
            np.load(
                endpoint_path(
                    sample_root,
                    checkpoint_stem,
                    str(record["prompt"]),
                    int(record.get("initial_seed", record.get("seed"))),
                ),
                allow_pickle=False,
            )
        )

    train_features, endpoint_features = load_or_compute_features(
        args.feature_cache,
        train_images,
        endpoint_images,
        args.query_ids,
        args.clip_model,
        args.batch_size,
        args.recompute_features,
    )
    if train_features.shape[0] != 10_000 or endpoint_features.shape[0] != len(args.query_ids):
        raise ValueError("CLIP feature count mismatch")

    targets_by_key, subset_dirs = load_targets(args.true_f_csv)
    membership = build_membership(
        np.arange(10_000, dtype=np.int64),
        subset_dirs,
        result_root / "lds_model",
    )
    rows: list[dict[str, object]] = []
    for position, query_id in enumerate(args.query_ids):
        # Both feature matrices are L2 normalized, so this is cosine similarity.
        scores = train_features @ endpoint_features[position]
        prediction = -(membership @ scores.astype(np.float64))
        target_names = {
            target for qid, target in targets_by_key if qid == query_id
        }
        ordered_targets = [name for name in TARGET_ORDER if name in target_names]
        ordered_targets.extend(sorted(target_names - set(ordered_targets)))
        for target in ordered_targets:
            target_rows = targets_by_key[(query_id, target)]
            if len(target_rows) != 192:
                raise ValueError(f"Q{query_id} {target}: {len(target_rows)}/192 true-f rows")
            true_f = np.asarray(
                [float(row["true_f"]) for row in target_rows], dtype=np.float64
            )
            lds = spearman(prediction, true_f)
            rows.append(
                {
                    "query_id": query_id,
                    "target_function": target,
                    "lds": float(lds),
                    "lds_percent": float(100.0 * lds),
                    "score_definition": "CLIP image cosine(training_image, query_output)",
                    "prediction_subset": "kept",
                    "prediction_sign": -1.0,
                    "num_subset_models": 192,
                    "clip_model": args.clip_model,
                }
            )
            print(f"[LDS] Q{query_id} {target}={100.0 * lds:+.3f}%", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "per_query.csv"
    write_csv(csv_path, rows)

    lines = [
        "CELEBA CLIP-SIMILARITY SCORE LDS",
        "=" * 80,
        f"queries={len(args.query_ids)} range={min(args.query_ids)}..{max(args.query_ids)}",
        "score=CLIP cosine similarity between each training image and the query output",
        "prediction=-sum of CLIP scores over kept subset; subset models=192",
        f"CLIP model={args.clip_model}",
        "",
        f"{'TARGET':<32} {'MEAN':>10} {'MEDIAN':>10} {'STD':>10} {'QUERIES':>8}",
        "-" * 76,
    ]
    all_targets = []
    for name in TARGET_ORDER:
        if any(str(row["target_function"]) == name for row in rows):
            all_targets.append(name)
    all_targets.extend(
        sorted({str(row["target_function"]) for row in rows} - set(all_targets))
    )
    for target in all_targets:
        values = [
            float(row["lds_percent"])
            for row in rows
            if row["target_function"] == target
        ]
        lines.append(
            f"{target:<32} {statistics.mean(values):+9.3f}% "
            f"{statistics.median(values):+9.3f}% "
            f"{statistics.pstdev(values):9.3f}% {len(values):8d}"
        )
    report_path = args.output_dir / "report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    print(f"[saved] {csv_path}", flush=True)
    print(f"[saved] {report_path}", flush=True)


if __name__ == "__main__":
    main()
