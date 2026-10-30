#!/usr/bin/env python3
"""Compare Shape3D Raw-gradient and AdamW Top-K endpoint similarity."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
CELEBA_SCRIPT = REFINE_ROOT / "celeba" / "script"
for path in (SHAPES_ROOT / "script", CELEBA_SCRIPT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_five_method_raw_vs_adamw_endpoint_perceptual import (
    METRICS,
    clip_features,
    dino_features,
    oriented_improvement,
    pixel_lpips_for_query,
)
from analyze_pint_raw_vs_adamw_topk_endpoint_clip import label_swap
from plot_endpoint_nine_row_top10 import (
    TRANSFORMS,
    banks_for_transform,
    load_queries,
)
from plot_endpoint_top6_datapoints import (
    display_image,
    endpoint_path,
    load_dataset,
    load_score_indices,
    load_top_scores,
    score_root,
)


METHOD_KEYS = (
    ("PINT / RAW", "pollute_raw"),
    ("PINT / BOTH-L2", "pollute_query_train_l2"),
    ("ENDPOINT TRACIN / RAW", "tracin_raw"),
    ("ENDPOINT TRACIN / BOTH-L2", "tracin_query_train_l2"),
    ("RETRAC / BOTH-L2", "retrac_query_train_l2"),
)
TRANSFORM_NAMES = ("raw_gradient", "adamw_full")


def parse_ids(text: str) -> list[int]:
    values = [int(value) for value in text.replace(",", " ").split() if value]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("query IDs must be nonempty and unique")
    return values


def score_bank_directory(
    query_root: Path, transform_name: str, method_key: str
) -> tuple[Path, int]:
    transform = TRANSFORMS[transform_name]
    matching = {
        bank.key: bank for bank in banks_for_transform(transform)
    }
    if method_key not in matching:
        raise KeyError(f"missing {method_key} for {transform_name}")
    bank = matching[method_key]
    return (
        query_root / f"traj_tracin_{bank.namespace}" / bank.score_directory,
        bank.ranking_sign,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def semantic_features(
    output_dir: Path,
    candidate_indices: np.ndarray,
    train_images: np.ndarray,
    endpoint_images: list[np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_path = output_dir / "semantic_features.npz"
    if cache_path.is_file() and not args.recompute_features:
        with np.load(cache_path, allow_pickle=False) as cache:
            if not np.array_equal(cache["candidate_indices"], candidate_indices):
                raise ValueError("cached candidate indices differ")
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

    candidate_images = train_images[candidate_indices]
    train_clip = clip_features(
        candidate_images, args.clip_model, args.batch_size, "training candidates"
    )
    endpoint_clip = clip_features(
        endpoint_images, args.clip_model, args.batch_size, "endpoints"
    )
    train_dino = dino_features(
        candidate_images,
        args.dino_repository,
        args.dino_model,
        args.batch_size,
        "training candidates",
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
            candidate_indices=candidate_indices,
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


def main() -> None:
    import lpips
    import torch

    default_data_root = Path(
        os.environ.get(
            "THREEDSHAPES_DATA_ROOT",
            str(REFINE_ROOT / "dataset" / "3dshapes" / "20000"),
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-file",
        type=Path,
        default=(
            SHAPES_ROOT
            / "queries_in_distribution_plus_zero_seed_100_219.json"
        ),
    )
    parser.add_argument("--query-ids", type=parse_ids, default=list(range(60)))
    parser.add_argument("--dataset", type=Path, default=default_data_root / "dataset.npz")
    parser.add_argument("--experiment", default="experiment1")
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
    parser.add_argument("--sample-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--recompute-features", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("perceptual evaluation requires a CUDA GPU")
    if not 0 < args.topk <= 5000:
        parser.error("--topk must be within [1, 5000]")

    queries = load_queries(args.query_file, args.query_ids)
    train_images, _labels = load_dataset(args.dataset)
    result_root = SHAPES_ROOT / "result" / args.experiment
    sample_root = args.sample_root or result_root / "sample_ddim_eta0_1000"
    output_dir = args.output_dir or (
        result_root
        / "analysis"
        / f"five_method_raw_vs_adamw_top{args.topk}_endpoint_perceptual_"
        f"q{min(args.query_ids)}_q{max(args.query_ids)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = f"seed_{args.train_seed}_epoch_{args.epochs:04d}"

    endpoint_images: list[np.ndarray] = []
    selections: dict[tuple[str, str, int], np.ndarray] = {}
    candidate_set: set[int] | None = None
    for query_position, query in enumerate(queries):
        endpoint_images.append(
            display_image(
                np.load(
                    endpoint_path(sample_root, query, checkpoint_stem),
                    allow_pickle=False,
                )
            )
        )
        query_root = score_root(result_root, args.train_seed, query)
        for method_label, method_key in METHOD_KEYS:
            for transform_name in TRANSFORM_NAMES:
                directory, ranking_sign = score_bank_directory(
                    query_root, transform_name, method_key
                )
                scores = np.asarray(
                    np.load(directory / "scores.npy"), dtype=np.float64
                ).reshape(-1)
                indices = load_score_indices(directory, len(scores))
                current_set = set(map(int, indices))
                if candidate_set is None:
                    candidate_set = current_set
                elif current_set != candidate_set:
                    raise ValueError(
                        f"Q{query.query_id} {method_label}/{transform_name}: "
                        "candidate universe differs"
                    )
                selected, _stored = load_top_scores(
                    directory, args.topk, len(train_images), ranking_sign
                )
                selections[(method_label, transform_name, query.query_id)] = selected
        print(
            f"[select] {query_position + 1}/{len(queries)} Q{query.query_id} "
            f"configurations={len(METHOD_KEYS) * len(TRANSFORM_NAMES)}",
            flush=True,
        )
    if candidate_set is None:
        raise RuntimeError("no candidate score bank loaded")
    candidate_indices = np.asarray(sorted(candidate_set), dtype=np.int64)
    candidate_position = {
        int(index): position for position, index in enumerate(candidate_indices)
    }
    print(f"[candidates] unique={len(candidate_indices)}", flush=True)

    train_clip, endpoint_clip, train_dino, endpoint_dino = semantic_features(
        output_dir, candidate_indices, train_images, endpoint_images, args
    )
    lpips_model = lpips.LPIPS(net=args.lpips_net).eval().to(torch.device("cuda"))

    rows: list[dict[str, object]] = []
    for query_position, query in enumerate(queries):
        union = sorted(
            set().union(
                *(
                    set(
                        map(
                            int,
                            selections[(method_label, transform_name, query.query_id)],
                        )
                    )
                    for method_label, _method_key in METHOD_KEYS
                    for transform_name in TRANSFORM_NAMES
                )
            )
        )
        cache_path = output_dir / f"pixel_lpips_q{query.query_id:03d}.npz"
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cache:
                if not np.array_equal(cache["indices"], union):
                    raise ValueError(f"{cache_path}: selected index cache differs")
                pixel = np.asarray(cache["pixel_rmse"], dtype=np.float32)
                perceptual = np.asarray(cache["lpips"], dtype=np.float32)
            print(
                f"[query cache] Q{query.query_id} reused pairs={len(union)}",
                flush=True,
            )
        else:
            pixel, perceptual = pixel_lpips_for_query(
                endpoint_images[query_position],
                union,
                train_images,
                batch_size=args.batch_size,
                lpips_model=lpips_model,
                label=f"Q{query.query_id}",
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
                f"[query progress] {query_position + 1}/{len(queries)} "
                f"Q{query.query_id} unique_pairs={len(union)}",
                flush=True,
            )
        local_position = {index: position for position, index in enumerate(union)}
        for method_label, _method_key in METHOD_KEYS:
            for transform_name in TRANSFORM_NAMES:
                indices = selections[(method_label, transform_name, query.query_id)]
                feature_slots = np.asarray(
                    [candidate_position[int(index)] for index in indices],
                    dtype=np.int64,
                )
                local_slots = np.asarray(
                    [local_position[int(index)] for index in indices],
                    dtype=np.int64,
                )
                clip_values = train_clip[feature_slots] @ endpoint_clip[query_position]
                dino_values = train_dino[feature_slots] @ endpoint_dino[query_position]
                rows.append(
                    {
                        "query_id": query.query_id,
                        "method": method_label,
                        "transform": transform_name,
                        "topk": args.topk,
                        "clip_cosine": float(clip_values.mean()),
                        "dino_cosine": float(dino_values.mean()),
                        "pixel_rmse": float(pixel[local_slots].mean()),
                        "lpips": float(perceptual[local_slots].mean()),
                    }
                )

    lookup = {
        (str(row["method"]), int(row["query_id"]), str(row["transform"])): row
        for row in rows
    }
    summary_rows: list[dict[str, object]] = []
    lines = [
        "SHAPE3D FIVE-METHOD RAW-GRADIENT VS ADAMW TOP-K ENDPOINT PERCEPTUAL SIMILARITY",
        f"queries={len(queries)} topk={args.topk} candidates={len(candidate_indices)}",
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
        for method_index, (method_label, _method_key) in enumerate(METHOD_KEYS):
            raw = np.asarray(
                [
                    lookup[(method_label, query.query_id, "raw_gradient")][metric]
                    for query in queries
                ],
                dtype=np.float64,
            )
            adamw = np.asarray(
                [
                    lookup[(method_label, query.query_id, "adamw_full")][metric]
                    for query in queries
                ],
                dtype=np.float64,
            )
            improvement = oriented_improvement(raw, adamw, direction)
            p1 = label_swap(
                improvement,
                args.permutations,
                2026 + 100 * metric_index + method_index,
                False,
                f"{metric}/{method_label}/one-sided",
            )
            p2 = label_swap(
                improvement,
                args.permutations,
                3026 + 100 * metric_index + method_index,
                True,
                f"{metric}/{method_label}/two-sided",
            )
            summary_rows.append(
                {
                    "metric": metric,
                    "direction": "higher" if direction > 0 else "lower",
                    "method": method_label,
                    "queries": len(queries),
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
                f"{method_label:30s} {raw.mean():11.6f} {adamw.mean():11.6f} "
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
                "candidate_count": len(candidate_indices),
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
