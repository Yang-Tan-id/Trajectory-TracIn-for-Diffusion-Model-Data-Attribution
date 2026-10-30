"""Compare ArtBench Raw-gradient and AdamW Top-K semantic similarity."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .analyze_five_method_cross_query_topk_overlap import (
    METHODS,
    TRANSFORMS,
    load_artifact,
    score_bank,
    select_topk,
)
from .config import ArtBenchConfig
from .data import load_metadata


def parse_ints(value: str) -> list[int]:
    values = [int(item) for item in value.replace(",", " ").split() if item]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("query IDs must be nonempty and unique")
    return values


def batched(values: Sequence[Path], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def clip_features(
    paths: Sequence[Path], model_name: str, batch_size: int, label: str
) -> np.ndarray:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CLIP evaluation requires a CUDA GPU")
    device = torch.device("cuda")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).eval().half().to(device)
    outputs = []
    total = len(paths)
    started = time.monotonic()
    for batch_number, batch_paths in enumerate(batched(paths, batch_size), start=1):
        images = []
        for path in batch_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        pixels = processor(images=images, return_tensors="pt")["pixel_values"]
        with torch.inference_mode():
            features = model.get_image_features(
                pixel_values=pixels.to(device, dtype=torch.float16)
            )
            features = torch.nn.functional.normalize(features.float(), dim=-1)
        outputs.append(features.cpu().numpy())
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
    return np.concatenate(outputs, axis=0)


def label_swap(
    diff: np.ndarray,
    *,
    permutations: int,
    seed: int,
    two_sided: bool,
) -> float:
    observed = float(diff.mean())
    generator = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    while completed < permutations:
        size = min(20_000, permutations - completed)
        signs = generator.choice(
            np.asarray([-1.0, 1.0]), size=(size, len(diff))
        )
        null_values = (signs * diff).mean(axis=1)
        if two_sided:
            extreme += int(
                np.count_nonzero(np.abs(null_values) >= abs(observed))
            )
        else:
            extreme += int(np.count_nonzero(null_values >= observed))
        completed += size
    return (extreme + 1) / (permutations + 1)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


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
        default=(
            root
            / "outputs/attribution/retrac-q60-t50-full-linear-hooks/retrac.npz"
        ),
    )
    parser.add_argument(
        "--endpoint-image-root",
        type=Path,
        default=(
            root
            / "outputs/topk-removal-five-method-metrics-q0-59/combined"
        ),
    )
    parser.add_argument(
        "--query-indices", type=parse_ints, default=list(range(60))
    )
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--permutations", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--recompute-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ArtBenchConfig()
    project = config.index_root.parent.parent
    if not 0 < args.topk < config.train_size:
        raise ValueError(f"--topk must be in [1, {config.train_size - 1}]")
    if args.batch_size < 1 or args.permutations < 1:
        raise ValueError("batch size and permutations must be positive")

    output_dir = args.output_dir or (
        project
        / "outputs/analysis"
        / f"five_method_raw_vs_adamw_top{args.topk}_endpoint_clip_q"
        f"{min(args.query_indices)}_q{max(args.query_indices)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "pollute": load_artifact(
            args.endpoint_root / "endpoint_pollute_timestamp_sum_squared.npz"
        ),
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
        raise ValueError(
            f"expected {config.train_size} candidates, got {len(candidate_indices)}"
        )

    metadata = load_metadata(config)
    missing_metadata = sorted(set(map(int, candidate_indices)) - set(metadata.index))
    if missing_metadata:
        raise ValueError(f"metadata indices absent: {missing_metadata[:10]}")
    train_paths = [
        Path(value)
        for value in metadata.loc[candidate_indices, "image_path"].tolist()
    ]
    endpoint_paths = [
        args.endpoint_image_root / f"query-{query:03d}" / "full-data.png"
        for query in args.query_indices
    ]
    for path in endpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    cache_path = output_dir / "clip_features.npz"
    train_features: np.ndarray
    endpoint_features: np.ndarray
    if cache_path.is_file() and not args.recompute_features:
        with np.load(cache_path, allow_pickle=False) as cache:
            cached_indices = np.asarray(cache["candidate_indices"], dtype=np.int64)
            cached_queries = np.asarray(cache["query_indices"], dtype=np.int64)
            cached_model = str(np.asarray(cache["clip_model"]).item())
            if not np.array_equal(cached_indices, candidate_indices):
                raise ValueError("cached candidate indices differ")
            if not np.array_equal(cached_queries, args.query_indices):
                raise ValueError("cached query IDs differ")
            if cached_model != args.clip_model:
                raise ValueError("cached CLIP model differs")
            train_features = np.asarray(cache["train_features"], dtype=np.float32)
            endpoint_features = np.asarray(
                cache["endpoint_features"], dtype=np.float32
            )
        print(f"[cache] reused {cache_path}", flush=True)
    else:
        train_features = clip_features(
            train_paths, args.clip_model, args.batch_size, "training"
        )
        endpoint_features = clip_features(
            endpoint_paths, args.clip_model, args.batch_size, "endpoints"
        )
        temporary = cache_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                candidate_indices=candidate_indices,
                query_indices=np.asarray(args.query_indices, dtype=np.int64),
                clip_model=np.asarray(args.clip_model),
                train_features=train_features,
                endpoint_features=endpoint_features,
            )
        temporary.replace(cache_path)
        print(f"[cache] saved {cache_path}", flush=True)

    feature_position = {
        int(metadata_index): position
        for position, metadata_index in enumerate(candidate_indices)
    }
    rows: list[dict[str, object]] = []
    for method_index, (method, artifact_name, variant, raw_sign, adamw_sign) in enumerate(METHODS):
        artifact = artifacts[artifact_name]
        for transform, ranking_sign in zip(
            TRANSFORMS, (raw_sign, adamw_sign), strict=True
        ):
            bank = score_bank(artifact, transform=transform, variant=variant)
            for endpoint_position, query in enumerate(args.query_indices):
                if not 0 <= query < bank.shape[0]:
                    raise IndexError(f"{method} lacks Q{query}")
                selected = select_topk(
                    bank[query],
                    artifact["indices"],
                    topk=args.topk,
                    ranking_sign=ranking_sign,
                )
                positions = np.asarray(
                    [feature_position[index] for index in sorted(selected)],
                    dtype=np.int64,
                )
                similarities = (
                    train_features[positions] @ endpoint_features[endpoint_position]
                )
                rows.append(
                    {
                        "query_index": query,
                        "method": method,
                        "transform": transform,
                        "topk": args.topk,
                        "clip_cosine_mean": float(similarities.mean()),
                        "clip_cosine_median": float(np.median(similarities)),
                        "clip_cosine_std": float(similarities.std()),
                    }
                )
        print(f"[scored] {method_index + 1}/{len(METHODS)} {method}", flush=True)

    lookup = {
        (str(row["method"]), int(row["query_index"]), str(row["transform"])): row
        for row in rows
    }
    summary_rows: list[dict[str, object]] = []
    lines = [
        "ARTBENCH FIVE-METHOD RAW-GRADIENT VS ADAMW TOP-K ENDPOINT CLIP SIMILARITY",
        f"queries={len(args.query_indices)} topk={args.topk} clip={args.clip_model}",
        "Higher cosine means selected training images are more semantically similar to the full-data endpoint.",
        "",
        (
            f"{'METHOD':30s} {'RAW-GRAD':>11s} {'ADAMW':>11s} "
            f"{'DELTA':>11s} {'ADAMW WINS':>12s} {'P1':>11s} {'P2':>11s}"
        ),
        "-" * 103,
    ]
    for method_index, (method, *_unused) in enumerate(METHODS):
        raw = np.asarray(
            [
                lookup[(method, query, "raw_gradient")]["clip_cosine_mean"]
                for query in args.query_indices
            ],
            dtype=np.float64,
        )
        adamw = np.asarray(
            [
                lookup[(method, query, "adamw_full")]["clip_cosine_mean"]
                for query in args.query_indices
            ],
            dtype=np.float64,
        )
        diff = adamw - raw
        p1 = label_swap(
            diff,
            permutations=args.permutations,
            seed=2026 + method_index,
            two_sided=False,
        )
        p2 = label_swap(
            diff,
            permutations=args.permutations,
            seed=3026 + method_index,
            two_sided=True,
        )
        summary = {
            "method": method,
            "queries": len(args.query_indices),
            "topk": args.topk,
            "raw_gradient_mean": float(raw.mean()),
            "adamw_mean": float(adamw.mean()),
            "adamw_minus_raw": float(diff.mean()),
            "adamw_wins": int(np.count_nonzero(diff > 0)),
            "raw_wins": int(np.count_nonzero(diff < 0)),
            "ties": int(np.count_nonzero(diff == 0)),
            "paired_label_swap_p1": p1,
            "paired_label_swap_p2": p2,
        }
        summary_rows.append(summary)
        lines.append(
            f"{method:30s} {raw.mean():11.6f} {adamw.mean():11.6f} "
            f"{diff.mean():+11.6f} "
            f"{summary['adamw_wins']:5d}/{len(diff):<6d} "
            f"{p1:11.6g} {p2:11.6g}"
        )

    report = "\n".join(lines) + "\n"
    print("\n" + report, end="")
    write_csv(output_dir / "per_query_endpoint_clip.csv", rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "report.txt").write_text(report, encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "query_indices": args.query_indices,
                "topk": args.topk,
                "clip_model": args.clip_model,
                "endpoint_image_root": str(args.endpoint_image_root.resolve()),
                "definition": (
                    "mean CLIP cosine between each selected training image and "
                    "the same-query full-data endpoint"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[saved] {output_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
