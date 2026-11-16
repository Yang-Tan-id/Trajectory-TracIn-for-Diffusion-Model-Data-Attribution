"""Compare ArtBench perceptual similarity for largest and smallest DAS scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze_das_vs_pint_top1000_endpoint_perceptual import (
    atomic_csv,
    load_das,
    parse_ints,
)
from .analyze_five_method_raw_vs_adamw_endpoint_perceptual import (
    METRICS,
    load_or_compute_features,
    pixel_lpips_for_query,
)
from .config import ArtBenchConfig
from .data import load_metadata


def select_extremes(
    scores: np.ndarray, indices: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if scores.shape != indices.shape:
        raise ValueError("score/index shape mismatch")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("candidate indices contain duplicates")
    if not 0 < k <= len(scores) // 2:
        raise ValueError(f"k must be within [1, {len(scores) // 2}]")
    finite = np.isfinite(scores)
    if int(finite.sum()) < 2 * k:
        raise ValueError(f"fewer than {2 * k} finite scores")
    top_order = np.argsort(-np.where(finite, scores, -np.inf), kind="stable")[:k]
    bottom_order = np.argsort(np.where(finite, scores, np.inf), kind="stable")[:k]
    if set(map(int, top_order)) & set(map(int, bottom_order)):
        raise ValueError("Top-K and Bottom-K selections overlap")
    return indices[top_order], indices[bottom_order]


def parse_args() -> argparse.Namespace:
    config = ArtBenchConfig()
    project = config.index_root.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--das-score-path",
        type=Path,
        default=(
            project
            / "outputs/attribution/artbench_das_50x1_full5000_dual32768_sm"
            / "scores/lambda_1000/scores.npz"
        ),
    )
    parser.add_argument(
        "--endpoint-image-root",
        type=Path,
        default=project / "outputs/topk-removal-five-method-metrics-q0-59/combined",
    )
    parser.add_argument("--query-indices", type=parse_ints, default=list(range(60)))
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--dino-repository", default="facebookresearch/dino:main")
    parser.add_argument("--dino-model", default="dino_vits16")
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--recompute-features", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import lpips
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("perceptual evaluation requires a CUDA GPU")
    config = ArtBenchConfig()
    project = config.index_root.parent.parent
    if not args.query_indices or len(args.query_indices) != len(set(args.query_indices)):
        raise ValueError("query IDs must be nonempty and unique")
    if not 0 < args.k <= config.train_size // 2:
        raise ValueError(f"k must be within [1, {config.train_size // 2}]")

    scores, query_ids, candidate_indices = load_das(args.das_score_path)
    if len(candidate_indices) != config.train_size:
        raise ValueError(
            f"expected {config.train_size} DAS candidates, found {len(candidate_indices)}"
        )
    query_positions = {int(query): position for position, query in enumerate(query_ids)}
    missing = [query for query in args.query_indices if query not in query_positions]
    if missing:
        raise ValueError(f"DAS score artifact is missing queries {missing}")

    selections: dict[tuple[int, str], np.ndarray] = {}
    for query in args.query_indices:
        top, bottom = select_extremes(
            scores[query_positions[query]], candidate_indices, args.k
        )
        selections[(query, "top")] = top
        selections[(query, "bottom")] = bottom
        print(f"[select] Q{query} top={len(top)} bottom={len(bottom)}", flush=True)

    output_dir = args.output_dir or (
        project
        / "outputs/analysis"
        / f"das_lambda1000_top_bottom{args.k}_endpoint_perceptual_q"
        f"{min(args.query_indices)}_q{max(args.query_indices)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

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

    previous_cache = (
        project
        / "outputs/analysis/das_vs_pint_top1000_endpoint_perceptual_q0_q59"
        / "semantic_features.npz"
    )
    feature_cache = args.feature_cache or (
        previous_cache if previous_cache.is_file() else output_dir / "semantic_features.npz"
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
    for query_position, query in enumerate(args.query_indices):
        union = np.unique(
            np.concatenate(
                [selections[(query, "top")], selections[(query, "bottom")]]
            )
        )
        if len(union) != 2 * args.k:
            raise ValueError(f"Q{query}: expected {2 * args.k} unique extreme indices")
        cache_path = output_dir / f"pixel_lpips_q{query:03d}.npz"
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cache:
                if not np.array_equal(cache["indices"], union):
                    raise ValueError(f"{cache_path}: selection cache differs")
                pixel = np.asarray(cache["pixel_rmse"], dtype=np.float32)
                perceptual = np.asarray(cache["lpips"], dtype=np.float32)
            print(f"[query cache] Q{query} pairs={len(union)}", flush=True)
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
                np.savez_compressed(
                    handle,
                    indices=union,
                    pixel_rmse=pixel,
                    lpips=perceptual,
                )
            temporary.replace(cache_path)

        local_position = {int(index): position for position, index in enumerate(union)}
        score_row = scores[query_positions[query]]
        score_position = {
            int(index): position for position, index in enumerate(candidate_indices)
        }
        for group in ("top", "bottom"):
            selected = selections[(query, group)]
            global_slots = np.asarray(
                [feature_position[int(index)] for index in selected], dtype=np.int64
            )
            local_slots = np.asarray(
                [local_position[int(index)] for index in selected], dtype=np.int64
            )
            score_slots = np.asarray(
                [score_position[int(index)] for index in selected], dtype=np.int64
            )
            selected_scores = score_row[score_slots]
            rows.append(
                {
                    "query_index": query,
                    "group": group,
                    "k": args.k,
                    "clip_cosine": float(
                        (train_clip[global_slots] @ endpoint_clip[query_position]).mean()
                    ),
                    "dino_cosine": float(
                        (train_dino[global_slots] @ endpoint_dino[query_position]).mean()
                    ),
                    "pixel_rmse": float(pixel[local_slots].mean()),
                    "lpips": float(perceptual[local_slots].mean()),
                    "minimum_selected_score": float(selected_scores.min()),
                    "maximum_selected_score": float(selected_scores.max()),
                }
            )
        print(
            f"[query] {query_position + 1}/{len(args.query_indices)} Q{query} pairs={len(union)}",
            flush=True,
        )

    atomic_csv(output_dir / "per_query.csv", rows)
    per_query = {(int(row["query_index"]), str(row["group"])): row for row in rows}
    lines = [
        "ARTBENCH DAS LAMBDA=1000 TOP-1000 VS BOTTOM-1000 PERCEPTUAL SANITY CHECK",
        "=" * 92,
        f"queries={len(args.query_indices)} range={min(args.query_indices)}..{max(args.query_indices)}",
        f"candidates={len(candidate_indices)}; k={args.k}",
        f"score artifact={args.das_score_path}",
        "Top means largest stored DAS scores; Bottom means smallest stored DAS scores.",
        "No removal and no retraining. Reference is the same-query full-data output.",
        "CLIP/DINO: higher is more similar; Pixel-RMSE/LPIPS: lower is more similar.",
    ]
    summary_rows: list[dict[str, object]] = []
    for metric, display, direction in METRICS:
        top = np.asarray(
            [float(per_query[(query, "top")][metric]) for query in args.query_indices]
        )
        bottom = np.asarray(
            [float(per_query[(query, "bottom")][metric]) for query in args.query_indices]
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
    atomic_csv(output_dir / "summary.csv", summary_rows)
    report = "\n".join(lines) + "\n"
    (output_dir / "report.txt").write_text(report, encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "query_indices": args.query_indices,
                "k": args.k,
                "lambda": 1000,
                "score_path": str(args.das_score_path),
                "selection": {"top": "descending", "bottom": "ascending"},
                "reference": "same-query full-data output",
                "feature_cache": str(feature_cache),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n" + report, end="", flush=True)
    print(f"[saved] {output_dir / 'report.txt'}", flush=True)
    print(f"[saved] {output_dir / 'per_query.csv'}", flush=True)
    print(f"[saved] {output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
