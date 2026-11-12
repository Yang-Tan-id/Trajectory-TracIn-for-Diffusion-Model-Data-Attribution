"""Compare perceptual-nearest Top-1000 with six attribution Top-1000 sets."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

from .analyze_five_method_raw_vs_adamw_endpoint_perceptual import (
    load_or_compute_features,
    pixel_lpips_for_query,
)
from .config import ArtBenchConfig
from .data import load_metadata
from .plot_das_top10_with_five_methods import METHODS, selections


PERCEPTUAL_METRICS = (
    ("pixel_rmse", "Pixel-RMSE", -1),
    ("clip_cosine", "CLIP", +1),
    ("lpips", "LPIPS", -1),
    ("dino_cosine", "DINO", +1),
)


def parse_ints(value: str) -> list[int]:
    result = [int(item) for item in value.replace(",", " ").split() if item]
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("query IDs must be nonempty and unique")
    if any(query < 0 or query >= 60 for query in result):
        raise argparse.ArgumentTypeError("query IDs must be in Q0-Q59")
    return result


def parse_args() -> argparse.Namespace:
    config = ArtBenchConfig()
    project = config.index_root.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-indices", type=parse_ints, default=list(range(60)))
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument(
        "--endpoint-root",
        type=Path,
        default=project / "outputs/attribution/endpoint-q60-t50-full-linear-hooks",
    )
    parser.add_argument(
        "--retrac-path",
        type=Path,
        default=(
            project
            / "outputs/attribution/retrac-q60-t50-full-linear-hooks/retrac.npz"
        ),
    )
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
        "--base-index-path",
        type=Path,
        default=config.index_root / "idx-train.pkl",
    )
    parser.add_argument(
        "--endpoint-image-root",
        type=Path,
        default=project / "outputs/topk-removal-five-method-metrics-q0-59/combined",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project / "outputs/analysis/perceptual_top1000_method_overlap_q0_q59",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=(
            project
            / "outputs/analysis/das_vs_pint_top1000_endpoint_perceptual_q0_q59"
            / "semantic_features.npz"
        ),
    )
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--dino-repository", default="facebookresearch/dino:main")
    parser.add_argument("--dino-model", default="dino_vits16")
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--recompute-features", action="store_true")
    return parser.parse_args()


def nearest_indices(
    values: np.ndarray, candidate_indices: np.ndarray, *, top_k: int, direction: int
) -> np.ndarray:
    if direction not in (-1, +1):
        raise ValueError("direction must be -1 or +1")
    oriented = direction * np.asarray(values, dtype=np.float64)
    ranking = np.where(np.isfinite(oriented), oriented, -np.inf)
    order = np.argsort(-ranking, kind="stable")[:top_k]
    if len(order) != top_k or not np.all(np.isfinite(ranking[order])):
        raise ValueError(f"fewer than {top_k} finite perceptual values")
    return np.asarray(candidate_indices, dtype=np.int64)[order]


def overlap(left: np.ndarray, right: np.ndarray) -> tuple[int, float, float]:
    left_set = set(map(int, left))
    right_set = set(map(int, right))
    intersection = len(left_set & right_set)
    return (
        intersection,
        intersection / len(left_set),
        intersection / len(left_set | right_set),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    import lpips
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Pixel/LPIPS analysis requires a CUDA GPU")
    if not 0 < args.top_k < 5000:
        raise ValueError("top-k must be in [1, 4999]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    method_sets = selections(args)
    with args.base_index_path.open("rb") as handle:
        candidate_indices = np.asarray(pickle.load(handle), dtype=np.int64)
    if candidate_indices.shape != (5000,) or len(np.unique(candidate_indices)) != 5000:
        raise ValueError("base index file must contain 5,000 unique indices")
    metadata = load_metadata(ArtBenchConfig())
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

    train_clip, endpoint_clip, train_dino, endpoint_dino = load_or_compute_features(
        args.feature_cache,
        candidate_indices,
        args.query_indices,
        train_paths,
        endpoint_paths,
        args,
    )
    lpips_model = lpips.LPIPS(net=args.lpips_net).eval().to(torch.device("cuda"))
    nearest: dict[str, dict[int, np.ndarray]] = {
        metric: {} for metric, _label, _direction in PERCEPTUAL_METRICS
    }

    for query_position, query in enumerate(args.query_indices):
        cache_path = args.output_dir / f"pixel_lpips_all5000_q{query:03d}.npz"
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cache:
                if not np.array_equal(cache["candidate_indices"], candidate_indices):
                    raise ValueError(f"{cache_path}: candidate order differs")
                pixel = np.asarray(cache["pixel_rmse"], dtype=np.float32)
                perceptual = np.asarray(cache["lpips"], dtype=np.float32)
            print(f"[cache] reused Q{query} Pixel/LPIPS", flush=True)
        else:
            pixel, perceptual = pixel_lpips_for_query(
                endpoint_paths[query_position],
                candidate_indices,
                path_by_index,
                image_size=args.image_size,
                batch_size=args.batch_size,
                lpips_model=lpips_model,
                label=f"Q{query} all-5000",
            )
            temporary = cache_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    candidate_indices=candidate_indices,
                    pixel_rmse=pixel,
                    lpips=perceptual,
                )
            temporary.replace(cache_path)
            print(f"[cache] saved {cache_path}", flush=True)
        values = {
            "pixel_rmse": pixel,
            "clip_cosine": train_clip @ endpoint_clip[query_position],
            "lpips": perceptual,
            "dino_cosine": train_dino @ endpoint_dino[query_position],
        }
        for metric, _label, direction in PERCEPTUAL_METRICS:
            nearest[metric][query] = nearest_indices(
                values[metric], candidate_indices, top_k=args.top_k, direction=direction
            )
        print(f"[nearest] {query_position + 1}/{len(args.query_indices)} Q{query}", flush=True)

    rows: list[dict[str, object]] = []
    method_labels = [label for label, *_rest in METHODS]
    for query in args.query_indices:
        for metric, metric_label, _direction in PERCEPTUAL_METRICS:
            for method in method_labels:
                count, fraction, jaccard = overlap(
                    nearest[metric][query], method_sets[method][query]
                )
                rows.append(
                    {
                        "query_index": query,
                        "perceptual_metric": metric,
                        "perceptual_label": metric_label,
                        "method": method,
                        "top_k": args.top_k,
                        "overlap_count": count,
                        "overlap_fraction": fraction,
                        "jaccard": jaccard,
                    }
                )

    summary_rows: list[dict[str, object]] = []
    lines = [
        "ARTBENCH PERCEPTUAL-NEAREST TOP-1000 VS ATTRIBUTION TOP-1000 OVERLAP",
        "=" * 94,
        f"queries={len(args.query_indices)} candidates=5000 topk={args.top_k}",
        "Overlap is intersection/1000; independent-random expectation is 20%.",
    ]
    for metric, metric_label, _direction in PERCEPTUAL_METRICS:
        lines.extend(
            [
                "",
                metric_label.upper(),
                f"{'METHOD':18s} {'MEAN':>10s} {'MEDIAN':>10s} {'STD':>10s} {'MIN':>10s} {'MAX':>10s}",
                "-" * 74,
            ]
        )
        for method in method_labels:
            values = np.asarray(
                [
                    float(row["overlap_fraction"])
                    for row in rows
                    if row["perceptual_metric"] == metric and row["method"] == method
                ]
            )
            summary = {
                "perceptual_metric": metric,
                "perceptual_label": metric_label,
                "method": method,
                "queries": len(values),
                "overlap_mean": float(values.mean()),
                "overlap_median": float(np.median(values)),
                "overlap_std": float(values.std()),
                "overlap_min": float(values.min()),
                "overlap_max": float(values.max()),
            }
            summary_rows.append(summary)
            lines.append(
                f"{method:18s} {summary['overlap_mean']:10.2%} "
                f"{summary['overlap_median']:10.2%} {summary['overlap_std']:10.2%} "
                f"{summary['overlap_min']:10.2%} {summary['overlap_max']:10.2%}"
            )

    write_csv(args.output_dir / "per_query_overlap.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    np.savez_compressed(
        args.output_dir / "perceptual_top1000_indices.npz",
        query_ids=np.asarray(args.query_indices, dtype=np.int64),
        metric_names=np.asarray([item[0] for item in PERCEPTUAL_METRICS]),
        indices=np.asarray(
            [
                [nearest[metric][query] for query in args.query_indices]
                for metric, _label, _direction in PERCEPTUAL_METRICS
            ],
            dtype=np.int64,
        ),
    )
    report = "\n".join(lines) + "\n"
    (args.output_dir / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report, end="")
    print(f"[saved] {args.output_dir / 'report.txt'}", flush=True)


if __name__ == "__main__":
    main()
