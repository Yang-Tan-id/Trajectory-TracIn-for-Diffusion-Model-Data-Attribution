"""Compare ArtBench DAS and PINT Top-K selections to full-data endpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .analyze_five_method_cross_query_topk_overlap import (
    load_artifact,
    score_bank,
    select_topk,
)
from .analyze_five_method_raw_vs_adamw_endpoint_perceptual import (
    METRICS,
    load_or_compute_features,
    pixel_lpips_for_query,
)
from .config import ArtBenchConfig
from .data import load_metadata


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split() if item]


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.replace(",", " ").split() if item]


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def das_score_files(score_root: Path, requested: list[float] | None) -> list[tuple[float, Path]]:
    discovered: dict[float, Path] = {}
    for path in score_root.glob("lambda_*/scores.npz"):
        try:
            damping = float(path.parent.name.removeprefix("lambda_"))
        except ValueError:
            continue
        discovered[damping] = path
    values = sorted(discovered) if requested is None else requested
    missing = [value for value in values if value not in discovered]
    if missing:
        raise FileNotFoundError(f"missing DAS scores for lambdas {missing} in {score_root}")
    if not values:
        raise FileNotFoundError(f"no lambda_*/scores.npz under {score_root}")
    return [(value, discovered[value]) for value in values]


def load_das(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        scores = np.asarray(payload["scores"], dtype=np.float64)
        query_ids = np.asarray(payload["query_ids"], dtype=np.int64)
        indices = np.asarray(payload["train_metadata_indices"], dtype=np.int64)
    if scores.shape != (len(query_ids), len(indices)):
        raise ValueError(f"{path}: inconsistent score shape {scores.shape}")
    return scores, query_ids, indices


def summary(values: np.ndarray) -> tuple[float, float, float]:
    return float(values.mean()), float(np.median(values)), float(values.std())


def parse_args() -> argparse.Namespace:
    root = ArtBenchConfig().index_root.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--das-score-root",
        type=Path,
        default=root / "outputs/attribution/artbench_das_50x1_full5000_dual32768_sm/scores",
    )
    parser.add_argument(
        "--pint-artifact",
        type=Path,
        default=(
            root
            / "outputs/attribution/endpoint-q60-t50-full-linear-hooks"
            / "endpoint_pollute_timestamp_sum_squared.npz"
        ),
    )
    parser.add_argument(
        "--endpoint-image-root",
        type=Path,
        default=root / "outputs/topk-removal-five-method-metrics-q0-59/combined",
    )
    parser.add_argument("--query-indices", type=parse_ints, default=list(range(60)))
    parser.add_argument("--lambdas", type=parse_floats)
    parser.add_argument("--topk", type=int, default=1000)
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
    if len(args.query_indices) != len(set(args.query_indices)):
        raise ValueError("query IDs must be unique")
    if not 0 < args.topk < config.train_size:
        raise ValueError(f"topk must be in [1, {config.train_size - 1}]")
    output_dir = args.output_dir or (
        project / "outputs/analysis/das_vs_pint_top1000_endpoint_perceptual_q0_q59"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pint = load_artifact(args.pint_artifact)
    pint_bank = score_bank(pint, transform="adamw_full", variant="query_train_l2")
    candidate_indices = np.asarray(pint["indices"], dtype=np.int64)
    configurations: dict[str, dict[int, set[int]]] = {
        "PINT AdamW / BOTH-L2": {
            query: select_topk(
                pint_bank[query], candidate_indices, topk=args.topk, ranking_sign=+1
            )
            for query in args.query_indices
        }
    }
    for damping, path in das_score_files(args.das_score_root, args.lambdas):
        scores, query_ids, indices = load_das(path)
        if set(map(int, indices)) != set(map(int, candidate_indices)):
            raise ValueError(f"{path}: DAS and PINT candidate universes differ")
        score_positions = {int(index): position for position, index in enumerate(indices)}
        reorder = np.asarray([score_positions[int(index)] for index in candidate_indices])
        query_positions = {int(query): position for position, query in enumerate(query_ids)}
        missing = [query for query in args.query_indices if query not in query_positions]
        if missing:
            raise ValueError(f"{path}: missing queries {missing}")
        label = f"DAS lambda={damping:g}"
        configurations[label] = {
            query: select_topk(
                scores[query_positions[query], reorder],
                candidate_indices,
                topk=args.topk,
                ranking_sign=+1,
            )
            for query in args.query_indices
        }
        print(f"[selected] {label} queries={len(args.query_indices)}", flush=True)

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

    feature_cache = args.feature_cache or output_dir / "semantic_features.npz"
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
        union = sorted(
            set().union(*(selection[query] for selection in configurations.values()))
        )
        cache_path = output_dir / f"pixel_lpips_top{args.topk}_q{query:03d}.npz"
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cache:
                if not np.array_equal(cache["indices"], union):
                    raise ValueError(f"{cache_path}: selection cache differs")
                pixel = np.asarray(cache["pixel_rmse"], dtype=np.float32)
                perceptual = np.asarray(cache["lpips"], dtype=np.float32)
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
                    handle, indices=np.asarray(union), pixel_rmse=pixel, lpips=perceptual
                )
            temporary.replace(cache_path)
        local_position = {index: position for position, index in enumerate(union)}
        for label, selection in configurations.items():
            indices = sorted(selection[query])
            global_slots = np.asarray([feature_position[index] for index in indices])
            local_slots = np.asarray([local_position[index] for index in indices])
            clip = train_clip[global_slots] @ endpoint_clip[query_position]
            dino = train_dino[global_slots] @ endpoint_dino[query_position]
            rows.append(
                {
                    "query_index": query,
                    "configuration": label,
                    "topk": args.topk,
                    "clip_cosine": float(clip.mean()),
                    "dino_cosine": float(dino.mean()),
                    "pixel_rmse": float(pixel[local_slots].mean()),
                    "lpips": float(perceptual[local_slots].mean()),
                }
            )
        print(
            f"[query] {query_position + 1}/{len(args.query_indices)} Q{query} "
            f"unique_pairs={len(union)}",
            flush=True,
        )

    atomic_csv(output_dir / "per_query.csv", rows)
    summary_rows: list[dict[str, object]] = []
    lines = [
        "ARTBENCH DAS VS PINT TOP-1000 ENDPOINT PERCEPTUAL SIMILARITY",
        "=" * 96,
        f"queries={len(args.query_indices)} topk={args.topk}",
        "CLIP/DINO: higher is better. Pixel-RMSE/LPIPS: lower is better.",
    ]
    for metric, metric_label, _direction in METRICS:
        lines.extend(["", metric_label.upper(), f"{'CONFIGURATION':34s} {'MEAN':>12s} {'MEDIAN':>12s} {'STD':>12s}"])
        lines.append("-" * 74)
        for label in configurations:
            values = np.asarray(
                [float(row[metric]) for row in rows if row["configuration"] == label]
            )
            mean, median, std = summary(values)
            summary_rows.append(
                {
                    "metric": metric,
                    "configuration": label,
                    "queries": len(values),
                    "mean": mean,
                    "median": median,
                    "std": std,
                }
            )
            lines.append(f"{label:34s} {mean:12.6f} {median:12.6f} {std:12.6f}")
    atomic_csv(output_dir / "summary.csv", summary_rows)
    report = output_dir / "report.txt"
    report.write_text("\n".join(lines) + "\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "query_indices": args.query_indices,
                "topk": args.topk,
                "configurations": list(configurations),
                "feature_cache": str(feature_cache),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print("\n".join(lines), flush=True)
    print(f"[saved] {report}", flush=True)


if __name__ == "__main__":
    main()
