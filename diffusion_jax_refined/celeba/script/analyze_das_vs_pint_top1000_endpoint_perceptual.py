#!/usr/bin/env python3
"""Compare DAS and PINT Top-K training images with CelebA endpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_five_method_cross_query_top1000_overlap import (
    load_queries,
    query_score_root,
)
from analyze_five_method_raw_vs_adamw_endpoint_perceptual import (
    load_or_compute_features,
    pixel_lpips_for_query,
    write_csv,
)
from analyze_pint_raw_vs_adamw_topk_endpoint_clip import (
    endpoint_path,
    score_directory,
    select_topk,
)


CELEBA_ROOT = Path(__file__).resolve().parents[1]
REFINED_ROOT = CELEBA_ROOT.parent
CONFIGURATIONS = (
    "DAS EMA / lambda=200",
    "PINT AdamW / RAW",
    "PINT AdamW / BOTH-L2",
)
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


def das_topk(directory: Path, topk: int) -> np.ndarray:
    scores = np.asarray(np.load(directory / "scores.npy"), dtype=np.float64).reshape(-1)
    indices = np.asarray(
        np.load(directory / "score_indices.npy"), dtype=np.int64
    ).reshape(-1)
    expected = np.arange(10_000, dtype=np.int64)
    if scores.shape != expected.shape or not np.array_equal(indices, expected):
        raise ValueError(f"DAS score bank is not ordered full-10k: {directory}")
    if int(np.isfinite(scores).sum()) < topk:
        raise ValueError(f"DAS score bank has fewer than {topk} finite values")
    oriented = np.where(np.isfinite(scores), scores, -np.inf)
    return indices[np.argsort(-oriented, kind="stable")[:topk]]


def main() -> None:
    data_root = REFINED_ROOT / "dataset" / "celeba" / "10000_attr40"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-file",
        type=Path,
        default=CELEBA_ROOT / "queries_attr40_val25_x4.json",
    )
    parser.add_argument("--query-ids", type=parse_ids, default=list(range(100)))
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
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--recompute-features", action="store_true")
    args = parser.parse_args()

    import lpips
    import torch

    if not torch.cuda.is_available():
        parser.error("perceptual evaluation requires a CUDA GPU")
    if not 0 < args.topk <= 10_000:
        parser.error("--topk must be within [1, 10000]")

    records = load_queries(args.query_file)
    missing = sorted(set(args.query_ids) - set(records))
    if missing:
        parser.error(f"query IDs missing from query file: {missing}")

    result_root = CELEBA_ROOT / "result" / args.experiment
    sample_root = result_root / "sample_ddim_eta0_50"
    output_dir = args.output_dir or (
        result_root
        / "analysis"
        / f"das_vs_pint_top{args.topk}_endpoint_perceptual_q0_q99"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cache = args.feature_cache or output_dir / "semantic_features.npz"

    with np.load(args.dataset, allow_pickle=False) as dataset:
        train_images = np.asarray(dataset["images"])
    if len(train_images) != 10_000:
        raise ValueError(f"expected 10000 training images, got {len(train_images)}")

    checkpoint_stem = f"seed_{args.train_seed}_epoch_{args.epochs:04d}"
    endpoint_images: list[np.ndarray] = []
    selections: dict[tuple[int, str], np.ndarray] = {}
    for position, query_id in enumerate(args.query_ids, start=1):
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
        selections[(query_id, CONFIGURATIONS[0])] = das_topk(
            score_root / "das_50x1_mc1_full10k_ema" / "score" / "lambda_200",
            args.topk,
        )
        for label, variant in zip(CONFIGURATIONS[1:], ("raw", "query_train_l2")):
            directory = score_directory(
                result_root,
                args.train_seed,
                prompt,
                seed,
                "pint",
                "adamw_full",
                "timestamp_sum_squared",
                variant,
            )
            selections[(query_id, label)] = select_topk(
                directory,
                "pint",
                "adamw_full",
                "timestamp_sum_squared",
                args.topk,
            )
        print(f"[select] {position}/{len(args.query_ids)} Q{query_id}", flush=True)

    train_clip, endpoint_clip, train_dino, endpoint_dino = load_or_compute_features(
        feature_cache, train_images, endpoint_images, args
    )
    lpips_model = lpips.LPIPS(net=args.lpips_net).eval().to(torch.device("cuda"))

    rows: list[dict[str, object]] = []
    for position, query_id in enumerate(args.query_ids):
        union = sorted(
            set().union(
                *(set(map(int, selections[(query_id, label)])) for label in CONFIGURATIONS)
            )
        )
        cache_path = output_dir / f"pixel_lpips_q{query_id:03d}.npz"
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cache:
                if not np.array_equal(cache["indices"], union):
                    raise ValueError(f"{cache_path}: selected-index cache differs")
                pixel = np.asarray(cache["pixel_rmse"], dtype=np.float32)
                perceptual = np.asarray(cache["lpips"], dtype=np.float32)
            print(f"[query cache] Q{query_id} pairs={len(union)}", flush=True)
        else:
            pixel, perceptual = pixel_lpips_for_query(
                endpoint_images[position],
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
                    indices=np.asarray(union, dtype=np.int64),
                    pixel_rmse=pixel,
                    lpips=perceptual,
                )
            temporary.replace(cache_path)

        local = {index: slot for slot, index in enumerate(union)}
        for label in CONFIGURATIONS:
            indices = selections[(query_id, label)]
            local_slots = np.asarray([local[int(index)] for index in indices])
            clip_values = train_clip[indices] @ endpoint_clip[position]
            dino_values = train_dino[indices] @ endpoint_dino[position]
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
        print(f"[query] {position + 1}/{len(args.query_ids)} Q{query_id}", flush=True)

    lookup = {
        (int(row["query_id"]), str(row["configuration"])): row for row in rows
    }
    lines = [
        "CELEBA DAS VS PINT TOP-1000 TRAINING-IMAGE PERCEPTUAL SIMILARITY",
        f"queries={len(args.query_ids)} range={min(args.query_ids)}..{max(args.query_ids)}",
        "No removal and no retraining. Reference is the same-query full-data endpoint.",
        "CLIP/DINO: higher is more similar; Pixel-RMSE/LPIPS: lower is more similar.",
    ]
    for metric, display, direction in METRICS:
        lines.extend(
            [
                "",
                display.upper(),
                f"{'CONFIGURATION':32s} {'MEAN':>11s} {'MEDIAN':>11s} {'STD':>11s}",
                "-" * 69,
            ]
        )
        arrays: dict[str, np.ndarray] = {}
        for label in CONFIGURATIONS:
            values = np.asarray(
                [lookup[(query_id, label)][metric] for query_id in args.query_ids],
                dtype=np.float64,
            )
            arrays[label] = values
            lines.append(
                f"{label:32s} {values.mean():11.6f} "
                f"{np.median(values):11.6f} {values.std():11.6f}"
            )
        for label in CONFIGURATIONS[1:]:
            improvement = direction * (arrays[CONFIGURATIONS[0]] - arrays[label])
            lines.append(
                f"DAS vs {label}: DAS wins="
                f"{np.count_nonzero(improvement > 0)}/{len(improvement)}, "
                f"ties={np.count_nonzero(improvement == 0)}, "
                f"oriented mean difference={improvement.mean():+.6f}"
            )

    report = "\n".join(lines) + "\n"
    write_csv(output_dir / "per_query.csv", rows)
    (output_dir / "report.txt").write_text(report, encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "query_ids": args.query_ids,
                "topk": args.topk,
                "configurations": list(CONFIGURATIONS),
                "das_contract": "EMA 50x1 full10k lambda=200, larger score selected",
                "pint_contract": "AdamW timestamp-sum-squared full10k",
                "reference": "same-query full-data endpoint",
                "clip_model": args.clip_model,
                "dino_model": args.dino_model,
                "lpips_network": args.lpips_net,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n" + report, end="")
    print(f"[saved] {output_dir / 'report.txt'}")
    print(f"[saved] {output_dir / 'per_query.csv'}")


if __name__ == "__main__":
    main()
