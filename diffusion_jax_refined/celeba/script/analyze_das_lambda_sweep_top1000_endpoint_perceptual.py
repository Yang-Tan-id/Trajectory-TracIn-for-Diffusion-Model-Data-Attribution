#!/usr/bin/env python3
"""Evaluate CelebA DAS Top-K training-image similarity for every lambda."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_das_vs_pint_top1000_endpoint_perceptual import (
    CELEBA_ROOT,
    METRICS,
    REFINED_ROOT,
    das_topk,
    parse_ids,
)
from analyze_five_method_cross_query_top1000_overlap import (
    load_queries,
    query_score_root,
)
from analyze_five_method_raw_vs_adamw_endpoint_perceptual import (
    load_or_compute_features,
    pixel_lpips_for_query,
    write_csv,
)
from analyze_pint_raw_vs_adamw_topk_endpoint_clip import endpoint_path


DEFAULT_LAMBDAS = (
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    500.0,
    1000.0,
    2000.0,
    5000.0,
    10000.0,
)


def parse_floats(text: str) -> list[float]:
    values = [float(value) for value in text.replace(",", " ").split() if value]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("lambdas must be nonempty and unique")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("lambdas must be positive")
    return values


def lambda_tag(value: float) -> str:
    return ("%g" % float(value)).replace("+", "_").replace("-", "neg_").replace(".", "p")


def lambda_label(value: float) -> str:
    return f"DAS EMA / lambda={value:g}"


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
    parser.add_argument(
        "--namespace",
        default="das_50x1_mc1_full10k_ema_sherman_morrison",
    )
    parser.add_argument(
        "--lambdas",
        type=parse_floats,
        default=list(DEFAULT_LAMBDAS),
    )
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
        / f"das_lambda_sweep_top{args.topk}_endpoint_perceptual_q"
        f"{min(args.query_ids)}_q{max(args.query_ids)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cache = args.feature_cache or output_dir / "semantic_features.npz"

    with np.load(args.dataset, allow_pickle=False) as dataset:
        train_images = np.asarray(dataset["images"])
    if len(train_images) != 10_000:
        raise ValueError(f"expected 10000 training images, got {len(train_images)}")

    checkpoint_stem = f"seed_{args.train_seed}_epoch_{args.epochs:04d}"
    endpoint_images: list[np.ndarray] = []
    selections: dict[tuple[int, float], np.ndarray] = {}
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
        for damping in args.lambdas:
            directory = (
                score_root
                / args.namespace
                / "score"
                / f"lambda_{lambda_tag(damping)}"
            )
            selections[(query_id, damping)] = das_topk(directory, args.topk)
        print(
            f"[select] {position}/{len(args.query_ids)} Q{query_id} "
            f"lambdas={len(args.lambdas)}",
            flush=True,
        )

    train_clip, endpoint_clip, train_dino, endpoint_dino = load_or_compute_features(
        feature_cache, train_images, endpoint_images, args
    )
    lpips_model = lpips.LPIPS(net=args.lpips_net).eval().to(torch.device("cuda"))

    rows: list[dict[str, object]] = []
    for position, query_id in enumerate(args.query_ids):
        union = sorted(
            set().union(
                *(
                    set(map(int, selections[(query_id, damping)]))
                    for damping in args.lambdas
                )
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
        for damping in args.lambdas:
            indices = selections[(query_id, damping)]
            local_slots = np.asarray([local[int(index)] for index in indices])
            clip_values = train_clip[indices] @ endpoint_clip[position]
            dino_values = train_dino[indices] @ endpoint_dino[position]
            rows.append(
                {
                    "query_id": query_id,
                    "lambda": damping,
                    "configuration": lambda_label(damping),
                    "topk": args.topk,
                    "clip_cosine": float(clip_values.mean()),
                    "dino_cosine": float(dino_values.mean()),
                    "pixel_rmse": float(pixel[local_slots].mean()),
                    "lpips": float(perceptual[local_slots].mean()),
                    "selected_union_size": len(union),
                }
            )
        print(
            f"[query] {position + 1}/{len(args.query_ids)} Q{query_id} "
            f"union={len(union)}",
            flush=True,
        )

    lines = [
        "CELEBA DAS LAMBDA SWEEP TOP-1000 TRAINING-IMAGE PERCEPTUAL SIMILARITY",
        "=" * 88,
        f"queries={len(args.query_ids)} range={min(args.query_ids)}..{max(args.query_ids)}",
        f"lambdas={len(args.lambdas)}; topk={args.topk}; candidate universe=10000",
        "No removal and no retraining. Reference is the same-query full-data endpoint.",
        "CLIP/DINO: higher is more similar; Pixel-RMSE/LPIPS: lower is more similar.",
        f"score namespace={args.namespace}",
    ]
    summary_rows: list[dict[str, object]] = []
    for metric, display, direction in METRICS:
        lines.extend(
            [
                "",
                display.upper(),
                f"{'LAMBDA':>10s} {'MEAN':>12s} {'MEDIAN':>12s} {'STD':>12s}",
                "-" * 50,
            ]
        )
        means: list[tuple[float, float]] = []
        for damping in args.lambdas:
            values = np.asarray(
                [
                    float(row[metric])
                    for row in rows
                    if float(row["lambda"]) == damping
                ],
                dtype=np.float64,
            )
            mean = float(values.mean())
            means.append((mean, damping))
            summary_rows.append(
                {
                    "metric": metric,
                    "lambda": damping,
                    "queries": len(values),
                    "mean": mean,
                    "median": float(np.median(values)),
                    "std": float(values.std()),
                }
            )
            lines.append(
                f"{damping:10g} {mean:12.6f} "
                f"{np.median(values):12.6f} {values.std():12.6f}"
            )
        best_mean, best_lambda = max(means) if direction > 0 else min(means)
        lines.append(
            f"BEST MEAN SIMILARITY: lambda={best_lambda:g} value={best_mean:.6f}"
        )

    report = "\n".join(lines) + "\n"
    write_csv(output_dir / "per_query.csv", rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "report.txt").write_text(report, encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "query_ids": args.query_ids,
                "topk": args.topk,
                "lambdas": args.lambdas,
                "namespace": args.namespace,
                "reference": "same-query full-data endpoint",
                "clip_model": args.clip_model,
                "dino_model": args.dino_model,
                "lpips_network": args.lpips_net,
                "feature_cache": str(feature_cache),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n" + report, end="")
    print(f"[saved] {output_dir / 'report.txt'}")
    print(f"[saved] {output_dir / 'per_query.csv'}")
    print(f"[saved] {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
