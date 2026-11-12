#!/usr/bin/env python3
"""Plot ArtBench Top-10 training images for five baselines plus DAS."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .analyze_five_method_cross_query_topk_overlap import load_artifact, score_bank
from .config import ArtBenchConfig
from .data import load_metadata


METHODS = (
    ("PINT", "pollute", "query_train_l2", +1),
    ("TracInCP", "tracin", "raw", -1),
    ("GAS", "tracin", "query_train_l2", -1),
    ("ReTrac", "retrac", None, -1),
    ("Random", "random", None, +1),
    ("DAS", "das", None, +1),
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
    parser.add_argument("--query-indices", type=parse_ints, default=list(range(10)))
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
        "--output-dir",
        type=Path,
        default=project / "outputs/analysis/top10_five_methods_plus_das_lambda1000",
    )
    parser.add_argument(
        "--endpoint-image-root",
        type=Path,
        default=(
            project / "outputs/topk-removal-five-method-metrics-q0-59/combined"
        ),
        help="Directory containing query-NNN/full-data.png outputs.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tile-size", type=int, default=160)
    parser.add_argument("--random-seed", type=int, default=2026)
    return parser.parse_args()


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_tile(path: Path, size: int) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size), "white")
    tile.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return tile


def top_indices(
    scores: np.ndarray, indices: np.ndarray, *, top_k: int, sign: int
) -> np.ndarray:
    oriented = sign * np.asarray(scores, dtype=np.float64)
    ranking = np.where(np.isfinite(oriented), oriented, -np.inf)
    order = np.argsort(-ranking, kind="stable")[:top_k]
    if len(order) != top_k or not np.all(np.isfinite(ranking[order])):
        raise ValueError(f"fewer than {top_k} finite scores")
    return np.asarray(indices, dtype=np.int64)[order]


def load_das(path: Path) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        scores = np.asarray(payload["scores"], dtype=np.float64)
        indices = np.asarray(payload["train_metadata_indices"], dtype=np.int64)
        query_ids = np.asarray(payload["query_ids"], dtype=np.int64)
    if scores.shape != (len(query_ids), len(indices)):
        raise ValueError(f"invalid DAS score shape: {scores.shape}")
    return scores, indices, {int(query): slot for slot, query in enumerate(query_ids)}


def selections(args: argparse.Namespace) -> dict[str, dict[int, np.ndarray]]:
    artifacts: dict[str, Any] = {
        "pollute": load_artifact(
            args.endpoint_root / "endpoint_pollute_timestamp_sum_squared.npz"
        ),
        "tracin": load_artifact(args.endpoint_root / "endpoint_tracin.npz"),
        "retrac": load_artifact(args.retrac_path),
    }
    with args.base_index_path.open("rb") as handle:
        base_indices = np.asarray(pickle.load(handle), dtype=np.int64)
    das_scores, das_indices, das_positions = load_das(args.das_score_path)
    universes = [set(map(int, artifact["indices"])) for artifact in artifacts.values()]
    universes.extend((set(map(int, base_indices)), set(map(int, das_indices))))
    if len({frozenset(universe) for universe in universes}) != 1:
        raise ValueError("method candidate universes differ")

    result: dict[str, dict[int, np.ndarray]] = {}
    for label, artifact_name, variant, sign in METHODS:
        result[label] = {}
        if artifact_name in artifacts:
            artifact = artifacts[artifact_name]
            bank = score_bank(artifact, transform="adamw_full", variant=variant)
            for query in args.query_indices:
                result[label][query] = top_indices(
                    bank[query], artifact["indices"], top_k=args.top_k, sign=sign
                )
        elif artifact_name == "das":
            for query in args.query_indices:
                if query not in das_positions:
                    raise ValueError(f"DAS scores do not contain Q{query}")
                result[label][query] = top_indices(
                    das_scores[das_positions[query]],
                    das_indices,
                    top_k=args.top_k,
                    sign=sign,
                )
        else:
            for query in args.query_indices:
                generator = np.random.default_rng(
                    np.random.SeedSequence([args.random_seed, query])
                )
                result[label][query] = top_indices(
                    generator.random(len(base_indices)),
                    base_indices,
                    top_k=args.top_k,
                    sign=sign,
                )
    return result


def render(
    output: Path,
    reference_output: Path,
    query: int,
    selected: dict[str, dict[int, np.ndarray]],
    paths: dict[int, Path],
    tile_size: int,
    top_k: int,
) -> None:
    gap = 6
    outer = 14
    label_width = 132
    output_gap = 40
    header_height = 42
    row_count = len(METHODS)
    width = (
        outer * 2
        + label_width
        + tile_size
        + output_gap
        + top_k * tile_size
        + (top_k - 1) * gap
    )
    height = outer * 2 + header_height + row_count * tile_size + (row_count - 1) * gap
    canvas = Image.new("RGB", (width, height), "#F7F7F7")
    draw = ImageDraw.Draw(canvas)
    row_font = font(17, bold=True)
    rank_font = font(14, bold=True)

    output_left = outer + label_width
    output_top = outer + header_height + (height - outer * 2 - header_height - tile_size) // 2
    output_label = "Output"
    output_bounds = draw.textbbox((0, 0), output_label, font=row_font)
    draw.text(
        (
            output_left + (tile_size - (output_bounds[2] - output_bounds[0])) / 2,
            outer + 8,
        ),
        output_label,
        font=row_font,
        fill="#202124",
    )
    canvas.paste(fit_tile(reference_output, tile_size), (output_left, output_top))
    draw.rectangle(
        (
            output_left,
            output_top,
            output_left + tile_size - 1,
            output_top + tile_size - 1,
        ),
        outline="#4C78A8",
        width=3,
    )

    grid_left = output_left + tile_size + output_gap

    for rank in range(top_k):
        left = grid_left + rank * (tile_size + gap)
        text = f"#{rank + 1}"
        bounds = draw.textbbox((0, 0), text, font=rank_font)
        draw.text(
            (left + (tile_size - (bounds[2] - bounds[0])) / 2, outer + 10),
            text,
            font=rank_font,
            fill="#333333",
        )
    for row, (label, *_unused) in enumerate(METHODS):
        top = outer + header_height + row * (tile_size + gap)
        bounds = draw.textbbox((0, 0), label, font=row_font)
        draw.text(
            (outer + 4, top + (tile_size - (bounds[3] - bounds[1])) / 2),
            label,
            font=row_font,
            fill="#202124",
        )
        for column, metadata_index in enumerate(selected[label][query]):
            left = grid_left + column * (tile_size + gap)
            canvas.paste(fit_tile(paths[int(metadata_index)], tile_size), (left, top))
            draw.rectangle(
                (left, top, left + tile_size - 1, top + tile_size - 1),
                outline="#D0D0D0",
                width=1,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)


def main() -> None:
    args = parse_args()
    if args.top_k < 1 or args.tile_size < 64:
        raise ValueError("top-k must be positive and tile-size must be at least 64")
    selected = selections(args)
    metadata = load_metadata(ArtBenchConfig())
    paths = {int(index): Path(row["image_path"]) for index, row in metadata.iterrows()}
    for query in args.query_indices:
        output = args.output_dir / f"artbench_q{query:02d}_top{args.top_k}_six_methods.png"
        reference_output = (
            args.endpoint_image_root / f"query-{query:03d}" / "full-data.png"
        )
        render(
            output,
            reference_output,
            query,
            selected,
            paths,
            args.tile_size,
            args.top_k,
        )
        print(f"[saved] Q{query} {output}", flush=True)
    print("[rows] PINT | TracInCP | GAS | ReTrac | Random | DAS", flush=True)


if __name__ == "__main__":
    main()
