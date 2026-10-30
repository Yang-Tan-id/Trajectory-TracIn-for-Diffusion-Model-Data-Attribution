#!/usr/bin/env python3
"""Compare CelebA Raw-gradient and AdamW Top-K endpoint similarity."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_five_method_cross_query_top1000_overlap import (
    METHODS,
    METHOD_PAIRS,
    load_queries,
    query_score_root,
    select_topk,
)
from analyze_pint_raw_vs_adamw_topk_endpoint_clip import (
    display_image,
    endpoint_path,
    label_swap,
)


CELEBA_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = CELEBA_ROOT.parent
METRICS = (
    ("clip_cosine", "CLIP cosine", +1),
    ("dino_cosine", "DINO cosine", +1),
    ("pixel_rmse", "Pixel RMSE", -1),
    ("lpips", "LPIPS", -1),
)


def parse_ids(text: str) -> list[int]:
    values = [int(value) for value in text.replace(",", " ").split() if value]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("query IDs must be nonempty and unique")
    return values


def batched(values: Sequence[Any], size: int):
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
            features = model.get_image_features(
                pixel_values=pixels.to(device, dtype=torch.float16)
            )
            features = torch.nn.functional.normalize(features.float(), dim=-1)
        output.append(features.cpu().numpy())
        completed = min(batch_number * batch_size, total)
        elapsed = time.monotonic() - started
        rate = completed / elapsed if elapsed else 0.0
        eta = (total - completed) / rate if rate else float("inf")
        print(
            f"[clip progress] {label} images={completed}/{total} "
            f"percent={100.0 * completed / total:.2f} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )
    return np.concatenate(output, axis=0)


def dino_features(
    images: Sequence[np.ndarray],
    repository: str,
    model_name: str,
    batch_size: int,
    label: str,
) -> np.ndarray:
    import torch
    from PIL import Image
    from torchvision import transforms

    if not torch.cuda.is_available():
        raise RuntimeError("DINO evaluation requires a CUDA GPU")
    device = torch.device("cuda")
    model = torch.hub.load(repository, model_name).eval().to(device)
    preprocess = transforms.Compose(
        [
            transforms.Resize(
                256, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    output = []
    total = len(images)
    started = time.monotonic()
    for batch_number, batch in enumerate(batched(images, batch_size), start=1):
        values = torch.stack(
            [
                preprocess(Image.fromarray(display_image(image), mode="RGB"))
                for image in batch
            ]
        ).to(device)
        with torch.inference_mode():
            features = model(values).float()
            features = torch.nn.functional.normalize(features, dim=-1)
        output.append(features.cpu().numpy())
        completed = min(batch_number * batch_size, total)
        elapsed = time.monotonic() - started
        rate = completed / elapsed if elapsed else 0.0
        eta = (total - completed) / rate if rate else float("inf")
        print(
            f"[dino progress] {label} images={completed}/{total} "
            f"percent={100.0 * completed / total:.2f} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )
    return np.concatenate(output, axis=0)


def load_or_compute_features(
    output_dir: Path,
    train_images: np.ndarray,
    endpoint_images: Sequence[np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_path = output_dir / "semantic_features.npz"
    if cache_path.is_file() and not args.recompute_features:
        with np.load(cache_path, allow_pickle=False) as cache:
            if not np.array_equal(cache["query_ids"], args.query_ids):
                raise ValueError("cached query IDs differ")
            if str(cache["clip_model"].item()) != args.clip_model:
                raise ValueError("cached CLIP model differs")
            if str(cache["dino_model"].item()) != args.dino_model:
                raise ValueError("cached DINO model differs")
            result = tuple(
                np.asarray(cache[name], dtype=np.float32)
                for name in (
                    "train_clip",
                    "endpoint_clip",
                    "train_dino",
                    "endpoint_dino",
                )
            )
        print(f"[cache] reused {cache_path}", flush=True)
        return result  # type: ignore[return-value]

    train_clip = clip_features(
        train_images, args.clip_model, args.batch_size, "training"
    )
    endpoint_clip = clip_features(
        endpoint_images, args.clip_model, args.batch_size, "endpoints"
    )
    train_dino = dino_features(
        train_images,
        args.dino_repository,
        args.dino_model,
        args.batch_size,
        "training",
    )
    endpoint_dino = dino_features(
        endpoint_images,
        args.dino_repository,
        args.dino_model,
        args.batch_size,
        "endpoints",
    )
    temporary = cache_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            query_ids=np.asarray(args.query_ids, dtype=np.int64),
            clip_model=np.asarray(args.clip_model),
            dino_model=np.asarray(args.dino_model),
            train_clip=train_clip,
            endpoint_clip=endpoint_clip,
            train_dino=train_dino,
            endpoint_dino=endpoint_dino,
        )
    temporary.replace(cache_path)
    print(f"[cache] saved {cache_path}", flush=True)
    return train_clip, endpoint_clip, train_dino, endpoint_dino


def pixel_lpips_for_query(
    endpoint_image: np.ndarray,
    selected_indices: Sequence[int],
    train_images: np.ndarray,
    *,
    batch_size: int,
    lpips_model: Any,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torchvision import transforms

    device = torch.device("cuda")
    to_tensor = transforms.ToTensor()
    reference = to_tensor(display_image(endpoint_image)).to(device)
    pixel_values: list[np.ndarray] = []
    lpips_values: list[np.ndarray] = []
    completed = 0
    started = time.monotonic()
    with torch.inference_mode():
        for batch_number, index_batch in enumerate(
            batched(selected_indices, batch_size), start=1
        ):
            values = torch.stack(
                [to_tensor(display_image(train_images[int(index)])) for index in index_batch]
            ).to(device)
            repeated = reference.unsqueeze(0).expand(values.shape[0], -1, -1, -1)
            rmse = torch.sqrt(torch.mean((values - repeated) ** 2, dim=(1, 2, 3)))
            perceptual = lpips_model(values * 2.0 - 1.0, repeated * 2.0 - 1.0)
            pixel_values.append(rmse.float().cpu().numpy())
            lpips_values.append(perceptual.flatten().float().cpu().numpy())
            completed += len(index_batch)
            if batch_number % 10 == 0 or completed == len(selected_indices):
                elapsed = time.monotonic() - started
                rate = completed / elapsed if elapsed else 0.0
                eta = (
                    (len(selected_indices) - completed) / rate
                    if rate
                    else float("inf")
                )
                print(
                    f"[pixel/lpips progress] {label} pairs={completed}/"
                    f"{len(selected_indices)} percent="
                    f"{100.0 * completed / len(selected_indices):.2f} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
    return np.concatenate(pixel_values), np.concatenate(lpips_values)


def oriented_improvement(
    raw: np.ndarray, adamw: np.ndarray, direction: int
) -> np.ndarray:
    if direction not in (-1, +1):
        raise ValueError("metric direction must be -1 or +1")
    return direction * (adamw - raw)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    import lpips
    import torch

    data_root = REFINE_ROOT / "dataset" / "celeba" / "10000_attr40"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-file",
        type=Path,
        default=CELEBA_ROOT / "queries_attr40_val25_x4.json",
    )
    parser.add_argument("--query-ids", type=parse_ids, default=list(range(60)))
    parser.add_argument("--dataset", type=Path, default=data_root / "dataset.npz")
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--dino-repository", default="facebookresearch/dino:main")
    parser.add_argument("--dino-model", default="dino_vits16")
    parser.add_argument(
        "--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--permutations", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--recompute-features", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("perceptual evaluation requires a CUDA GPU")
    if not 0 < args.topk <= 10_000:
        parser.error("--topk must be within [1, 10000]")
    if args.batch_size < 1 or args.permutations < 1:
        parser.error("batch size and permutations must be positive")

    records = load_queries(args.query_file)
    missing = sorted(set(args.query_ids) - set(records))
    if missing:
        parser.error(f"query IDs missing from query file: {missing}")
    result_root = CELEBA_ROOT / "result" / args.experiment
    sample_root = result_root / "sample_ddim_eta0_50"
    output_dir = args.output_dir or (
        result_root
        / "analysis"
        / f"five_method_raw_vs_adamw_top{args.topk}_endpoint_perceptual_"
        f"q{min(args.query_ids)}_q{max(args.query_ids)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.dataset, allow_pickle=False) as dataset:
        train_images = np.asarray(dataset["images"])
    if len(train_images) != 10_000:
        raise ValueError(f"expected 10000 training images, got {len(train_images)}")

    checkpoint_stem = f"seed_{args.train_seed}_epoch_{args.epochs:04d}"
    endpoint_images: list[np.ndarray] = []
    selections: dict[tuple[str, int], set[int]] = {}
    for query_position, query_id in enumerate(args.query_ids):
        record = records[query_id]
        prompt = str(record["prompt"])
        seed = int(record.get("initial_seed", record.get("seed")))
        endpoint_images.append(
            np.load(
                endpoint_path(sample_root, checkpoint_stem, prompt, seed),
                allow_pickle=False,
            )
        )
        score_root = query_score_root(result_root, args.train_seed, record)
        for label, namespace, score_directory, ranking_sign in METHODS:
            _universe, selected = select_topk(
                score_root / f"traj_tracin_{namespace}" / score_directory,
                topk=args.topk,
                ranking_sign=ranking_sign,
            )
            selections[(label, query_id)] = selected
        print(
            f"[select] {query_position + 1}/{len(args.query_ids)} Q{query_id} "
            f"configurations={len(METHODS)}",
            flush=True,
        )

    train_clip, endpoint_clip, train_dino, endpoint_dino = load_or_compute_features(
        output_dir, train_images, endpoint_images, args
    )
    lpips_model = lpips.LPIPS(net=args.lpips_net).eval().to(torch.device("cuda"))

    rows: list[dict[str, object]] = []
    for query_position, query_id in enumerate(args.query_ids):
        union = sorted(
            set().union(*(selections[(label, query_id)] for label, *_ in METHODS))
        )
        cache_path = output_dir / f"pixel_lpips_q{query_id:03d}.npz"
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cache:
                cached_indices = np.asarray(cache["indices"], dtype=np.int64)
                if not np.array_equal(cached_indices, union):
                    raise ValueError(f"{cache_path}: selected index cache differs")
                pixel = np.asarray(cache["pixel_rmse"], dtype=np.float32)
                perceptual = np.asarray(cache["lpips"], dtype=np.float32)
            print(f"[query cache] Q{query_id} reused pairs={len(union)}", flush=True)
        else:
            pixel, perceptual = pixel_lpips_for_query(
                endpoint_images[query_position],
                union,
                train_images,
                batch_size=args.batch_size,
                lpips_model=lpips_model,
                label=f"Q{query_id}",
            )
            temporary = cache_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    indices=union,
                    pixel_rmse=pixel,
                    lpips=perceptual,
                )
            temporary.replace(cache_path)
            print(
                f"[query progress] {query_position + 1}/{len(args.query_ids)} "
                f"Q{query_id} unique_pairs={len(union)}",
                flush=True,
            )
        local_position = {index: position for position, index in enumerate(union)}
        for label, *_unused in METHODS:
            indices = sorted(selections[(label, query_id)])
            slots = np.asarray(indices, dtype=np.int64)
            local_slots = np.asarray([local_position[index] for index in indices])
            clip_values = train_clip[slots] @ endpoint_clip[query_position]
            dino_values = train_dino[slots] @ endpoint_dino[query_position]
            rows.append(
                {
                    "query_id": query_id,
                    "configuration": label,
                    "topk": args.topk,
                    "clip_cosine": float(clip_values.mean()),
                    "dino_cosine": float(dino_values.mean()),
                    "pixel_rmse": float(pixel[local_slots].mean()),
                    "lpips": float(perceptual[local_slots].mean()),
                }
            )

    lookup = {
        (str(row["configuration"]), int(row["query_id"])): row for row in rows
    }
    summary_rows: list[dict[str, object]] = []
    lines = [
        "CELEBA FIVE-METHOD RAW-GRADIENT VS ADAMW TOP-K ENDPOINT PERCEPTUAL SIMILARITY",
        f"queries={len(args.query_ids)} topk={args.topk}",
        "CLIP/DINO: higher is better. Pixel-RMSE/LPIPS: lower is better.",
    ]
    for metric_index, (metric, metric_label, direction) in enumerate(METRICS):
        lines.extend(
            [
                "",
                metric_label.upper(),
                f"{'METHOD':30s} {'RAW-GRAD':>11s} {'ADAMW':>11s} "
                f"{'IMPROVE':>11s} {'ADAMW WINS':>12s} {'P1':>11s} {'P2':>11s}",
                "-" * 103,
            ]
        )
        for method_index, (method, raw_label, adamw_label) in enumerate(METHOD_PAIRS):
            raw = np.asarray(
                [lookup[(raw_label, query_id)][metric] for query_id in args.query_ids],
                dtype=np.float64,
            )
            adamw = np.asarray(
                [lookup[(adamw_label, query_id)][metric] for query_id in args.query_ids],
                dtype=np.float64,
            )
            improvement = oriented_improvement(raw, adamw, direction)
            p1 = label_swap(
                improvement,
                args.permutations,
                2026 + 100 * metric_index + method_index,
                False,
                f"{metric}/{method}/one-sided",
            )
            p2 = label_swap(
                improvement,
                args.permutations,
                3026 + 100 * metric_index + method_index,
                True,
                f"{metric}/{method}/two-sided",
            )
            summary_rows.append(
                {
                    "metric": metric,
                    "direction": "higher" if direction > 0 else "lower",
                    "method": method,
                    "queries": len(args.query_ids),
                    "raw_gradient_mean": float(raw.mean()),
                    "adamw_mean": float(adamw.mean()),
                    "oriented_adamw_improvement": float(improvement.mean()),
                    "adamw_wins": int(np.count_nonzero(improvement > 0)),
                    "raw_wins": int(np.count_nonzero(improvement < 0)),
                    "ties": int(np.count_nonzero(improvement == 0)),
                    "paired_label_swap_p1": p1,
                    "paired_label_swap_p2": p2,
                }
            )
            lines.append(
                f"{method:30s} {raw.mean():11.6f} {adamw.mean():11.6f} "
                f"{improvement.mean():+11.6f} "
                f"{np.count_nonzero(improvement > 0):5d}/{len(improvement):<6d} "
                f"{p1:11.6g} {p2:11.6g}"
            )

    report = "\n".join(lines) + "\n"
    print("\n" + report, end="")
    write_csv(output_dir / "per_query.csv", rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "report.txt").write_text(report, encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "query_ids": args.query_ids,
                "topk": args.topk,
                "clip_model": args.clip_model,
                "dino_repository": args.dino_repository,
                "dino_model": args.dino_model,
                "lpips_network": args.lpips_net,
                "definition": "mean training-image similarity/distance to the same-query full-data endpoint",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[saved] {output_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
