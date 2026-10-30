"""Compare Raw-gradient and AdamW extreme-K endpoint perceptual similarity.

For every query and attribution configuration, this script compares each of
the selected Top-K and/or Bottom-K training images with the full-data endpoint
for that query.  CLIP and DINO are reported as cosine similarities (larger is
better); pixel RMSE and LPIPS are reported as distances (smaller is better).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .analyze_five_method_cross_query_topk_overlap import (
    METHODS,
    TRANSFORMS,
    load_artifact,
    score_bank,
    select_topk,
)
from .analyze_five_method_raw_vs_adamw_endpoint_clip import (
    clip_features,
    label_swap,
)
from .config import ArtBenchConfig
from .data import load_metadata


METRICS = (
    ("clip_cosine", "CLIP cosine", +1),
    ("dino_cosine", "DINO cosine", +1),
    ("pixel_rmse", "Pixel RMSE", -1),
    ("lpips", "LPIPS", -1),
)


def parse_ints(value: str) -> list[int]:
    values = [int(item) for item in value.replace(",", " ").split() if item]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("query IDs must be nonempty and unique")
    return values


def batched(values: Sequence[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def dino_features(
    paths: Sequence[Path],
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
    outputs = []
    total = len(paths)
    started = time.monotonic()
    for batch_number, batch_paths in enumerate(batched(paths, batch_size), start=1):
        images = []
        for path in batch_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as source:
                images.append(preprocess(source.convert("RGB")))
        values = torch.stack(images).to(device)
        with torch.inference_mode():
            features = model(values).float()
            features = torch.nn.functional.normalize(features, dim=-1)
        outputs.append(features.cpu().numpy())
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
    return np.concatenate(outputs, axis=0)


def load_or_compute_features(
    cache_path: Path,
    candidate_indices: np.ndarray,
    query_indices: Sequence[int],
    train_paths: Sequence[Path],
    endpoint_paths: Sequence[Path],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cache_path.is_file() and not args.recompute_features:
        with np.load(cache_path, allow_pickle=False) as cache:
            if not np.array_equal(cache["candidate_indices"], candidate_indices):
                raise ValueError("cached candidate indices differ")
            if not np.array_equal(cache["query_indices"], args.query_indices):
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

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    train_clip = clip_features(
        train_paths, args.clip_model, args.batch_size, "training"
    )
    endpoint_clip = clip_features(
        endpoint_paths, args.clip_model, args.batch_size, "endpoints"
    )
    train_dino = dino_features(
        train_paths,
        args.dino_repository,
        args.dino_model,
        args.batch_size,
        "training",
    )
    endpoint_dino = dino_features(
        endpoint_paths,
        args.dino_repository,
        args.dino_model,
        args.batch_size,
        "endpoints",
    )
    temporary = cache_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            candidate_indices=candidate_indices,
            query_indices=np.asarray(query_indices, dtype=np.int64),
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
    endpoint_path: Path,
    selected_indices: Sequence[int],
    path_by_index: dict[int, Path],
    *,
    image_size: int,
    batch_size: int,
    lpips_model: Any,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from PIL import Image
    from torchvision import transforms

    device = torch.device("cuda")
    preprocess = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
        ]
    )
    with Image.open(endpoint_path) as source:
        reference = preprocess(source.convert("RGB")).to(device)
    pixel_values: list[np.ndarray] = []
    lpips_values: list[np.ndarray] = []
    completed = 0
    started = time.monotonic()
    with torch.inference_mode():
        for batch_number, index_batch in enumerate(
            batched(selected_indices, batch_size), start=1
        ):
            candidates = []
            for index in index_batch:
                path = path_by_index[int(index)]
                with Image.open(path) as source:
                    candidates.append(preprocess(source.convert("RGB")))
            values = torch.stack(candidates).to(device)
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def oriented_improvement(raw: np.ndarray, adamw: np.ndarray, direction: int) -> np.ndarray:
    """Return positive values when AdamW is better than Raw-gradient."""
    if direction not in (-1, +1):
        raise ValueError("metric direction must be -1 or +1")
    return direction * (adamw - raw)


def select_extreme_k(
    scores: np.ndarray,
    indices: np.ndarray,
    *,
    k: int,
    ranking_sign: int,
    extreme: str,
) -> set[int]:
    """Select Top-K or Bottom-K after orienting larger as more influential."""
    if extreme not in ("top", "bottom"):
        raise ValueError("extreme must be 'top' or 'bottom'")
    oriented_sign = ranking_sign if extreme == "top" else -ranking_sign
    return select_topk(scores, indices, topk=k, ranking_sign=oriented_sign)


def parse_args() -> argparse.Namespace:
    config = ArtBenchConfig()
    root = config.index_root.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint-root",
        type=Path,
        default=root / "outputs/attribution/endpoint-q60-t50-full-linear-hooks",
    )
    parser.add_argument(
        "--retrac-path",
        type=Path,
        default=root / "outputs/attribution/retrac-q60-t50-full-linear-hooks/retrac.npz",
    )
    parser.add_argument(
        "--endpoint-image-root",
        type=Path,
        default=root / "outputs/topk-removal-five-method-metrics-q0-59/combined",
    )
    parser.add_argument("--query-indices", type=parse_ints, default=list(range(60)))
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument(
        "--selection",
        choices=("top", "bottom", "both"),
        default="top",
        help="evaluate the most influential, least influential, or both sets",
    )
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--dino-repository", default="facebookresearch/dino:main")
    parser.add_argument("--dino-model", default="dino_vits16")
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--permutations", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        help="optional semantic_features.npz from an earlier compatible run",
    )
    parser.add_argument("--recompute-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import lpips
    import torch

    config = ArtBenchConfig()
    project = config.index_root.parent.parent
    if not torch.cuda.is_available():
        raise RuntimeError("perceptual evaluation requires a CUDA GPU")
    if not 0 < args.topk < config.train_size:
        raise ValueError(f"--topk must be in [1, {config.train_size - 1}]")
    if args.batch_size < 1 or args.image_size < 1 or args.permutations < 1:
        raise ValueError("sizes and permutations must be positive")
    selection_tag = "top_bottom" if args.selection == "both" else args.selection
    output_dir = args.output_dir or (
        project
        / "outputs/analysis"
        / f"five_method_raw_vs_adamw_{selection_tag}{args.topk}_endpoint_perceptual_"
        f"q{min(args.query_indices)}_q{max(args.query_indices)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cache = args.feature_cache or output_dir / "semantic_features.npz"
    extremes = ("top", "bottom") if args.selection == "both" else (args.selection,)

    artifacts = {
        "pollute": load_artifact(args.endpoint_root / "endpoint_pollute_timestamp_sum_squared.npz"),
        "tracin": load_artifact(args.endpoint_root / "endpoint_tracin.npz"),
        "retrac": load_artifact(args.retrac_path),
    }
    universes = {
        name: set(map(int, artifact["indices"]))
        for name, artifact in artifacts.items()
    }
    if len({frozenset(values) for values in universes.values()}) != 1:
        raise ValueError("the three attribution artifacts use different candidates")
    candidate_indices = np.asarray(artifacts["pollute"]["indices"], dtype=np.int64)
    if len(candidate_indices) != config.train_size:
        raise ValueError(f"expected {config.train_size} candidates")

    metadata = load_metadata(config)
    path_by_index = {
        int(index): Path(metadata.loc[int(index), "image_path"])
        for index in candidate_indices
    }
    train_paths = [path_by_index[int(index)] for index in candidate_indices]
    endpoint_paths = [
        args.endpoint_image_root / f"query-{query:03d}" / "full-data.png"
        for query in args.query_indices
    ]
    for path in endpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    selected: dict[tuple[str, str, str, int], set[int]] = {}
    for method, artifact_name, variant, raw_sign, adamw_sign in METHODS:
        artifact = artifacts[artifact_name]
        for transform, sign in zip(TRANSFORMS, (raw_sign, adamw_sign), strict=True):
            bank = score_bank(artifact, transform=transform, variant=variant)
            for query in args.query_indices:
                for extreme in extremes:
                    selected[(extreme, method, transform, query)] = select_extreme_k(
                        bank[query],
                        artifact["indices"],
                        k=args.topk,
                        ranking_sign=sign,
                        extreme=extreme,
                    )

    train_clip, endpoint_clip, train_dino, endpoint_dino = load_or_compute_features(
        feature_cache,
        candidate_indices,
        args.query_indices,
        train_paths,
        endpoint_paths,
        args,
    )
    feature_position = {
        int(index): position for position, index in enumerate(candidate_indices)
    }
    lpips_model = lpips.LPIPS(net=args.lpips_net).eval().to(torch.device("cuda"))

    rows: list[dict[str, object]] = []
    total_queries = len(args.query_indices)
    for query_position, query in enumerate(args.query_indices):
        union = sorted(
            set().union(
                *(
                    selected[(extreme, method, transform, query)]
                    for extreme in extremes
                    for method, *_ in METHODS
                    for transform in TRANSFORMS
                )
            )
        )
        cache_path = output_dir / f"pixel_lpips_{selection_tag}_q{query:03d}.npz"
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cache:
                cached_indices = np.asarray(cache["indices"], dtype=np.int64)
                if not np.array_equal(cached_indices, union):
                    raise ValueError(f"{cache_path}: selected index cache differs")
                pixel = np.asarray(cache["pixel_rmse"], dtype=np.float32)
                perceptual = np.asarray(cache["lpips"], dtype=np.float32)
            print(f"[query cache] Q{query} reused pairs={len(union)}", flush=True)
        else:
            pixel, perceptual = pixel_lpips_for_query(
                endpoint_paths[query_position],
                union,
                path_by_index,
                image_size=args.image_size,
                batch_size=args.batch_size,
                lpips_model=lpips_model,
                label=f"Q{query}",
            )
            temporary = cache_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, indices=union, pixel_rmse=pixel, lpips=perceptual)
            temporary.replace(cache_path)
            print(
                f"[query progress] {query_position + 1}/{total_queries} "
                f"Q{query} unique_pairs={len(union)}",
                flush=True,
            )
        local_position = {index: position for position, index in enumerate(union)}
        for extreme in extremes:
            for method, *_ in METHODS:
                for transform in TRANSFORMS:
                    indices = sorted(selected[(extreme, method, transform, query)])
                    global_slots = np.asarray([feature_position[index] for index in indices])
                    local_slots = np.asarray([local_position[index] for index in indices])
                    clip_values = train_clip[global_slots] @ endpoint_clip[query_position]
                    dino_values = train_dino[global_slots] @ endpoint_dino[query_position]
                    rows.append(
                        {
                            "selection": extreme,
                            "query_index": query,
                            "method": method,
                            "transform": transform,
                            "k": args.topk,
                            "clip_cosine": float(clip_values.mean()),
                            "dino_cosine": float(dino_values.mean()),
                            "pixel_rmse": float(pixel[local_slots].mean()),
                            "lpips": float(perceptual[local_slots].mean()),
                        }
                    )

    lookup = {
        (
            str(row["selection"]),
            str(row["method"]),
            int(row["query_index"]),
            str(row["transform"]),
        ): row
        for row in rows
    }
    summary_rows: list[dict[str, object]] = []
    lines = [
        "ARTBENCH FIVE-METHOD RAW-GRADIENT VS ADAMW EXTREME-K ENDPOINT PERCEPTUAL SIMILARITY",
        f"queries={len(args.query_indices)} k={args.topk} selection={args.selection}",
        "CLIP/DINO: higher is better. Pixel-RMSE/LPIPS: lower is better.",
    ]
    for extreme_index, extreme in enumerate(extremes):
        lines.extend(["", f"{'=' * 44} {extreme.upper()}-{args.topk} {'=' * 44}"])
        for metric_index, (metric, label, direction) in enumerate(METRICS):
            lines.extend(
                [
                    "",
                    label.upper(),
                    f"{'METHOD':30s} {'RAW-GRAD':>11s} {'ADAMW':>11s} {'IMPROVE':>11s} {'ADAMW WINS':>12s} {'P1':>11s} {'P2':>11s}",
                    "-" * 103,
                ]
            )
            for method_index, (method, *_unused) in enumerate(METHODS):
                raw = np.asarray(
                    [lookup[(extreme, method, query, "raw_gradient")][metric] for query in args.query_indices],
                    dtype=np.float64,
                )
                adamw = np.asarray(
                    [lookup[(extreme, method, query, "adamw_full")][metric] for query in args.query_indices],
                    dtype=np.float64,
                )
                improvement = oriented_improvement(raw, adamw, direction)
                seed_offset = 1000 * extreme_index + 100 * metric_index + method_index
                p1 = label_swap(
                    improvement,
                    permutations=args.permutations,
                    seed=2026 + seed_offset,
                    two_sided=False,
                )
                p2 = label_swap(
                    improvement,
                    permutations=args.permutations,
                    seed=3026 + seed_offset,
                    two_sided=True,
                )
                summary_rows.append(
                    {
                        "comparison": "adamw_vs_raw",
                        "selection": extreme,
                        "metric": metric,
                        "direction": "higher" if direction > 0 else "lower",
                        "method": method,
                        "transform": "raw_gradient_vs_adamw_full",
                        "queries": len(args.query_indices),
                        "left_mean": float(raw.mean()),
                        "right_mean": float(adamw.mean()),
                        "oriented_improvement": float(improvement.mean()),
                        "right_wins": int(np.count_nonzero(improvement > 0)),
                        "left_wins": int(np.count_nonzero(improvement < 0)),
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

    if args.selection == "both":
        lines.extend(["", "=" * 38 + " TOP VS BOTTOM " + "=" * 38])
        for metric_index, (metric, label, direction) in enumerate(METRICS):
            lines.extend(
                [
                    "",
                    label.upper(),
                    f"{'METHOD / TRANSFORM':46s} {'TOP':>11s} {'BOTTOM':>11s} {'TOP BETTER':>11s} {'TOP WINS':>10s} {'P1':>11s} {'P2':>11s}",
                    "-" * 119,
                ]
            )
            for method_index, (method, *_unused) in enumerate(METHODS):
                for transform_index, transform in enumerate(TRANSFORMS):
                    top = np.asarray(
                        [lookup[("top", method, query, transform)][metric] for query in args.query_indices],
                        dtype=np.float64,
                    )
                    bottom = np.asarray(
                        [lookup[("bottom", method, query, transform)][metric] for query in args.query_indices],
                        dtype=np.float64,
                    )
                    improvement = direction * (top - bottom)
                    seed_offset = 1000 * metric_index + 10 * method_index + transform_index
                    p1 = label_swap(
                        improvement,
                        permutations=args.permutations,
                        seed=5026 + seed_offset,
                        two_sided=False,
                    )
                    p2 = label_swap(
                        improvement,
                        permutations=args.permutations,
                        seed=6026 + seed_offset,
                        two_sided=True,
                    )
                    summary_rows.append(
                        {
                            "comparison": "top_vs_bottom",
                            "selection": "top_vs_bottom",
                            "metric": metric,
                            "direction": "higher" if direction > 0 else "lower",
                            "method": method,
                            "transform": transform,
                            "queries": len(args.query_indices),
                            "left_mean": float(bottom.mean()),
                            "right_mean": float(top.mean()),
                            "oriented_improvement": float(improvement.mean()),
                            "right_wins": int(np.count_nonzero(improvement > 0)),
                            "left_wins": int(np.count_nonzero(improvement < 0)),
                            "ties": int(np.count_nonzero(improvement == 0)),
                            "paired_label_swap_p1": p1,
                            "paired_label_swap_p2": p2,
                        }
                    )
                    display = f"{method} / {transform}"
                    lines.append(
                        f"{display:46s} {top.mean():11.6f} {bottom.mean():11.6f} "
                        f"{improvement.mean():+11.6f} "
                        f"{np.count_nonzero(improvement > 0):4d}/{len(improvement):<5d} "
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
                "query_indices": args.query_indices,
                "k": args.topk,
                "selection": args.selection,
                "feature_cache": str(feature_cache),
                "clip_model": args.clip_model,
                "dino_repository": args.dino_repository,
                "dino_model": args.dino_model,
                "lpips_network": args.lpips_net,
                "image_size_for_pixel_and_lpips": args.image_size,
                "definition": "mean selected training-image similarity/distance to the same-query full-data endpoint",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[saved] {output_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
