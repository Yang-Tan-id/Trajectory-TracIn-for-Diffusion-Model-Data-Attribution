#!/usr/bin/env python3
"""Compare perceptual similarity of the largest and smallest CelebA DAS scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_das_lambda_sweep_top1000_endpoint_perceptual import lambda_tag
from analyze_das_vs_pint_top1000_endpoint_perceptual import (
    CELEBA_ROOT,
    METRICS,
    REFINED_ROOT,
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


def select_extremes(directory: Path, k: int) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(np.load(directory / "scores.npy"), dtype=np.float64).reshape(-1)
    indices = np.asarray(
        np.load(directory / "score_indices.npy"), dtype=np.int64
    ).reshape(-1)
    expected = np.arange(10_000, dtype=np.int64)
    if scores.shape != expected.shape or not np.array_equal(indices, expected):
        raise ValueError(f"DAS score bank is not ordered full-10k: {directory}")
    finite = np.isfinite(scores)
    if int(finite.sum()) < 2 * k:
        raise ValueError(f"DAS score bank has fewer than {2 * k} finite values")
    top_order = np.argsort(-np.where(finite, scores, -np.inf), kind="stable")[:k]
    bottom_order = np.argsort(np.where(finite, scores, np.inf), kind="stable")[:k]
    if set(map(int, top_order)) & set(map(int, bottom_order)):
        raise ValueError("Top-K and Bottom-K selections overlap")
    return indices[top_order], indices[bottom_order]


def main() -> None:
    data_root = REFINED_ROOT / "dataset" / "celeba" / "10000_attr40"
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
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--namespace", default="das_50x1_mc1_full10k_ema")
    parser.add_argument("--lambda-value", type=float, default=200.0)
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
    if not 0 < args.k <= 5_000:
        parser.error("--k must be within [1, 5000]")
    if args.lambda_value <= 0:
        parser.error("--lambda-value must be positive")

    records = load_queries(args.query_file)
    missing = sorted(set(args.query_ids) - set(records))
    if missing:
        parser.error(f"query IDs missing from query file: {missing}")

    result_root = CELEBA_ROOT / "result" / args.experiment
    sample_root = result_root / "sample_ddim_eta0_50"
    output_dir = args.output_dir or (
        result_root
        / "analysis"
        / f"das_lambda{lambda_tag(args.lambda_value)}_top_bottom{args.k}_"
        f"endpoint_perceptual_q{min(args.query_ids)}_q{max(args.query_ids)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_cache = (
        result_root
        / "analysis"
        / "das_lambda_sweep_top1000_endpoint_perceptual_q0_q59"
        / "semantic_features.npz"
    )
    feature_cache = args.feature_cache or (
        previous_cache if previous_cache.is_file() else output_dir / "semantic_features.npz"
    )

    with np.load(args.dataset, allow_pickle=False) as dataset:
        train_images = np.asarray(dataset["images"])
    if len(train_images) != 10_000:
        raise ValueError(f"expected 10000 training images, got {len(train_images)}")

    checkpoint_stem = f"seed_{args.train_seed}_epoch_{args.epochs:04d}"
    endpoint_images: list[np.ndarray] = []
    selections: dict[tuple[int, str], np.ndarray] = {}
    score_dir_by_query: dict[int, Path] = {}
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
        score_dir = (
            query_score_root(result_root, args.train_seed, record)
            / args.namespace
            / "score"
            / f"lambda_{lambda_tag(args.lambda_value)}"
        )
        score_dir_by_query[query_id] = score_dir
        top, bottom = select_extremes(score_dir, args.k)
        selections[(query_id, "top")] = top
        selections[(query_id, "bottom")] = bottom
        print(f"[select] {position}/{len(args.query_ids)} Q{query_id}", flush=True)

    train_clip, endpoint_clip, train_dino, endpoint_dino = load_or_compute_features(
        feature_cache, train_images, endpoint_images, args
    )
    lpips_model = lpips.LPIPS(net=args.lpips_net).eval().to(torch.device("cuda"))

    rows: list[dict[str, object]] = []
    for position, query_id in enumerate(args.query_ids):
        union = np.unique(
            np.concatenate(
                [selections[(query_id, "top")], selections[(query_id, "bottom")]]
            )
        )
        if len(union) != 2 * args.k:
            raise ValueError(f"Q{query_id}: expected {2 * args.k} unique extremes")
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
                    indices=union,
                    pixel_rmse=pixel,
                    lpips=perceptual,
                )
            temporary.replace(cache_path)

        local = {int(index): slot for slot, index in enumerate(union)}
        for group in ("top", "bottom"):
            indices = selections[(query_id, group)]
            slots = np.asarray([local[int(index)] for index in indices], dtype=np.int64)
            rows.append(
                {
                    "query_id": query_id,
                    "group": group,
                    "k": args.k,
                    "lambda": args.lambda_value,
                    "clip_cosine": float((train_clip[indices] @ endpoint_clip[position]).mean()),
                    "dino_cosine": float((train_dino[indices] @ endpoint_dino[position]).mean()),
                    "pixel_rmse": float(pixel[slots].mean()),
                    "lpips": float(perceptual[slots].mean()),
                    "minimum_selected_score": float(
                        np.load(score_dir_by_query[query_id] / "scores.npy")[indices].min()
                    ),
                    "maximum_selected_score": float(
                        np.load(score_dir_by_query[query_id] / "scores.npy")[indices].max()
                    ),
                }
            )
        print(f"[query] {position + 1}/{len(args.query_ids)} Q{query_id}", flush=True)

    per_query = {(int(row["query_id"]), str(row["group"])): row for row in rows}
    lines = [
        "CELEBA EMA-DAS TOP-1000 VS BOTTOM-1000 TRAINING-IMAGE PERCEPTUAL SANITY CHECK",
        "=" * 96,
        f"queries={len(args.query_ids)} range={min(args.query_ids)}..{max(args.query_ids)}",
        f"score={args.namespace}/lambda_{lambda_tag(args.lambda_value)}; k={args.k}",
        "Top means largest stored DAS scores; Bottom means smallest stored DAS scores.",
        "No removal and no retraining. Reference is the same-query full-data endpoint.",
        "CLIP/DINO: higher is more similar; Pixel-RMSE/LPIPS: lower is more similar.",
    ]
    summary_rows: list[dict[str, object]] = []
    for metric, display, direction in METRICS:
        top = np.asarray(
            [float(per_query[(query_id, "top")][metric]) for query_id in args.query_ids]
        )
        bottom = np.asarray(
            [float(per_query[(query_id, "bottom")][metric]) for query_id in args.query_ids]
        )
        delta = top - bottom
        top_better = int(np.count_nonzero(direction * delta > 0))
        bottom_better = int(np.count_nonzero(direction * delta < 0))
        lines.extend(
            [
                "",
                display.upper(),
                f"Top mean={top.mean():.6f} median={np.median(top):.6f} std={top.std():.6f}",
                f"Bottom mean={bottom.mean():.6f} median={np.median(bottom):.6f} std={bottom.std():.6f}",
                f"Top-Bottom mean={delta.mean():+.6f}; top more similar={top_better}/{len(top)}; "
                f"bottom more similar={bottom_better}/{len(top)}",
            ]
        )
        summary_rows.append(
            {
                "metric": metric,
                "queries": len(top),
                "top_mean": float(top.mean()),
                "top_median": float(np.median(top)),
                "top_std": float(top.std()),
                "bottom_mean": float(bottom.mean()),
                "bottom_median": float(np.median(bottom)),
                "bottom_std": float(bottom.std()),
                "top_minus_bottom_mean": float(delta.mean()),
                "top_more_similar_queries": top_better,
                "bottom_more_similar_queries": bottom_better,
            }
        )

    report = "\n".join(lines) + "\n"
    write_csv(output_dir / "per_query.csv", rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "report.txt").write_text(report, encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "query_ids": args.query_ids,
                "k": args.k,
                "namespace": args.namespace,
                "lambda": args.lambda_value,
                "selection": {"top": "descending", "bottom": "ascending"},
                "reference": "same-query full-data endpoint",
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
