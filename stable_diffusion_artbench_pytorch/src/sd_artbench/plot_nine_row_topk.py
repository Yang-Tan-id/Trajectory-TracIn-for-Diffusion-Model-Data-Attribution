"""Plot nine-row Top-K ArtBench attribution comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .config import ArtBenchConfig


VARIANT_LABELS = ("RAW", "QUERY-L2", "TRAIN-L2", "BOTH-L2")


@dataclass(frozen=True)
class ScoreBank:
    key: str
    label: str
    scores: np.ndarray
    largest: bool


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split() if item]


def parse_strings(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def parse_args() -> argparse.Namespace:
    defaults = ArtBenchConfig()
    root = defaults.index_root.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint-root",
        type=Path,
        default=root / "outputs" / "attribution" / "endpoint-q60-t50-full-linear-hooks",
    )
    parser.add_argument(
        "--retrac-path",
        type=Path,
        default=(
            root
            / "outputs"
            / "attribution"
            / "retrac-q60-t50-full-linear-hooks"
            / "retrac.npz"
        ),
    )
    parser.add_argument(
        "--query-manifest",
        type=Path,
        default=root / "outputs" / "artbench2-queries-60" / "manifest.json",
    )
    parser.add_argument("--metadata-csv", type=Path, default=defaults.metadata_csv)
    parser.add_argument("--image-root", type=Path, default=defaults.image_root)
    parser.add_argument("--query-indices", type=parse_ints, default=list(range(10)))
    parser.add_argument(
        "--transforms",
        type=parse_strings,
        default=["raw_gradient", "adamw_full"],
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "attribution" / "nine-row-top10-q0-9",
    )
    parser.add_argument("--tile-size", type=int, default=150)
    return parser.parse_args()


def load_score_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        result = {
            "scores": np.asarray(payload["scores"], dtype=np.float64),
            "indices": np.asarray(payload["train_metadata_indices"], dtype=np.int64),
            "transforms": [str(value) for value in payload["transform_names"]],
            "variants": (
                [str(value) for value in payload["variant_names"]]
                if "variant_names" in payload.files
                else None
            ),
            "manifest": json.loads(str(payload["manifest_json"].item())),
        }
    if not np.isfinite(result["scores"]).all():
        raise ValueError(f"non-finite scores in {path}")
    return result


def load_queries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("query manifest must be a non-empty JSON list")
    records = []
    for query_index, item in enumerate(payload):
        image_path = Path(item["file"])
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        record = dict(item)
        record["query_index"] = query_index
        record["image_path"] = image_path
        records.append(record)
    return records


def load_training_records(
    metadata_csv: Path, image_root: Path, source_indices: Sequence[int]
) -> list[dict[str, Any]]:
    frame = pd.read_csv(metadata_csv)
    records = []
    for train_position, source_index in enumerate(source_indices):
        row = frame.loc[int(source_index)]
        image_path = image_root / str(row["label"]) / str(row["name"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        records.append(
            {
                "train_position": train_position,
                "source_index": int(source_index),
                "label": str(row["label"]),
                "name": str(row["name"]),
                "image_path": image_path,
            }
        )
    return records


def topk(scores: np.ndarray, count: int, largest: bool) -> np.ndarray:
    if scores.ndim != 1:
        raise ValueError("one-query scores must be rank one")
    ranking_values = -scores if largest else scores
    candidates = np.argpartition(ranking_values, count - 1)[:count]
    order = np.argsort(ranking_values[candidates], kind="stable")
    return candidates[order]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def image_tile(path: Path, size: int) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.contain(source.convert("RGB"), (size, size))
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def shortened(text: str, maximum: int) -> str:
    return text if len(text) <= maximum else text[: maximum - 3] + "..."


def build_banks(
    endpoint_tracin: dict[str, Any],
    endpoint_pollute: dict[str, Any],
    retrac: dict[str, Any],
    transform_slot: int,
) -> list[ScoreBank]:
    banks = []
    for variant_slot, variant_label in enumerate(VARIANT_LABELS):
        banks.append(
            ScoreBank(
                key=f"endpoint_pollute_{variant_label.lower()}",
                label=f"Endpoint-Pollute | {variant_label} | largest",
                scores=endpoint_pollute["scores"][transform_slot, variant_slot],
                largest=True,
            )
        )
    for variant_slot, variant_label in enumerate(VARIANT_LABELS):
        banks.append(
            ScoreBank(
                key=f"endpoint_tracin_{variant_label.lower()}",
                label=f"Endpoint-TracIn | {variant_label} | smallest",
                scores=endpoint_tracin["scores"][transform_slot, variant_slot],
                largest=False,
            )
        )
    banks.append(
        ScoreBank(
            key="retrac_both_l2",
            label="ReTrac | BOTH-L2 | smallest",
            scores=retrac["scores"][transform_slot],
            largest=False,
        )
    )
    if len(banks) != 9:
        raise AssertionError("expected exactly nine score banks")
    return banks


def make_figure(
    *,
    query: dict[str, Any],
    transform_name: str,
    banks: Sequence[ScoreBank],
    training: Sequence[dict[str, Any]],
    top_k: int,
    tile_size: int,
    csv_rows: list[dict[str, Any]],
) -> Image.Image:
    label_width = 255
    caption_height = 48
    header_height = 76
    row_gap = 7
    cell_width = tile_size + 8
    row_height = tile_size + caption_height
    width = label_width + (top_k + 1) * cell_width + 8
    height = header_height + len(banks) * (row_height + row_gap) + 8
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(21, bold=True)
    label_font = font(13, bold=True)
    text_font = font(11)
    small_font = font(10)
    query_index = int(query["query_index"])
    draw.text(
        (10, 7),
        f"ArtBench attribution | Q{query_index:03d} | {transform_name}",
        fill="black",
        font=title_font,
    )
    draw.text(
        (10, 38),
        f"seed={query['seed']} | {shortened(str(query['prompt']), 120)}",
        fill="black",
        font=text_font,
    )
    query_tile = image_tile(Path(query["image_path"]), tile_size)
    for row, bank in enumerate(banks):
        y = header_height + row * (row_height + row_gap)
        group_color = (
            (239, 247, 255)
            if row < 4
            else (255, 246, 235)
            if row < 8
            else (239, 252, 239)
        )
        draw.rectangle((0, y, width, y + row_height), fill=group_color)
        draw.multiline_text(
            (10, y + 48),
            bank.label.replace(" | ", "\n"),
            fill="black",
            font=label_font,
            spacing=4,
        )
        query_x = label_width
        canvas.paste(query_tile, (query_x, y))
        draw.text((query_x + 4, y + tile_size + 5), "QUERY", fill="black", font=text_font)
        selected = topk(bank.scores[query_index], top_k, bank.largest)
        for rank, train_position in enumerate(selected, start=1):
            record = training[int(train_position)]
            x = label_width + rank * cell_width
            canvas.paste(image_tile(Path(record["image_path"]), tile_size), (x, y))
            score = float(bank.scores[query_index, train_position])
            draw.text(
                (x + 3, y + tile_size + 3),
                f"#{rank} pos={int(train_position)} src={record['source_index']}",
                fill="black",
                font=small_font,
            )
            draw.text(
                (x + 3, y + tile_size + 20),
                f"{record['label']} | {score:+.3e}",
                fill="black",
                font=small_font,
            )
            csv_rows.append(
                {
                    "query_index": query_index,
                    "prompt": query["prompt"],
                    "seed": query["seed"],
                    "transform": transform_name,
                    "method_variant": bank.key,
                    "ranking_direction": "largest" if bank.largest else "smallest",
                    "rank": rank,
                    "train_position": int(train_position),
                    "source_index": record["source_index"],
                    "train_label": record["label"],
                    "train_file": str(record["image_path"]),
                    "stored_score": score,
                }
            )
    return canvas


def validate_inputs(
    endpoint_tracin: dict[str, Any],
    endpoint_pollute: dict[str, Any],
    retrac: dict[str, Any],
) -> None:
    for candidate, name in (
        (endpoint_pollute, "Endpoint-Pollute"),
        (retrac, "ReTrac"),
    ):
        if not np.array_equal(endpoint_tracin["indices"], candidate["indices"]):
            raise ValueError(f"{name} training-index order differs from Endpoint-TracIn")
        if endpoint_tracin["transforms"] != candidate["transforms"]:
            raise ValueError(f"{name} transform order differs from Endpoint-TracIn")
    expected_variants = ["raw", "query_l2", "train_l2", "query_train_l2"]
    for candidate, name in (
        (endpoint_tracin, "Endpoint-TracIn"),
        (endpoint_pollute, "Endpoint-Pollute"),
    ):
        if candidate["variants"] != expected_variants:
            raise ValueError(f"{name} variants are {candidate['variants']}, expected {expected_variants}")
        if candidate["scores"].ndim != 4:
            raise ValueError(f"{name} scores must have shape [transform,variant,query,train]")
    if retrac["scores"].ndim != 3:
        raise ValueError("ReTrac scores must have shape [transform,query,train]")
    query_counts = {
        endpoint_tracin["scores"].shape[-2],
        endpoint_pollute["scores"].shape[-2],
        retrac["scores"].shape[-2],
    }
    if len(query_counts) != 1:
        raise ValueError(f"score files have different query counts: {sorted(query_counts)}")


def main() -> None:
    args = parse_args()
    if args.top_k < 1 or args.tile_size < 64:
        raise ValueError("--top-k must be positive and --tile-size must be at least 64")
    endpoint_tracin = load_score_file(args.endpoint_root / "endpoint_tracin.npz")
    endpoint_pollute = load_score_file(
        args.endpoint_root / "endpoint_pollute_timestamp_sum_squared.npz"
    )
    retrac = load_score_file(args.retrac_path)
    validate_inputs(endpoint_tracin, endpoint_pollute, retrac)
    queries = load_queries(args.query_manifest)
    query_count = endpoint_tracin["scores"].shape[-2]
    if len(queries) != query_count:
        raise ValueError(f"query manifest has {len(queries)} rows but scores have {query_count}")
    if not args.query_indices or min(args.query_indices) < 0 or max(args.query_indices) >= query_count:
        raise ValueError(f"query indices must be inside [0, {query_count - 1}]")
    if len(args.query_indices) != len(set(args.query_indices)):
        raise ValueError("query indices must be unique")
    unknown_transforms = sorted(
        set(args.transforms).difference(endpoint_tracin["transforms"])
    )
    if unknown_transforms:
        raise ValueError(f"unknown transforms: {unknown_transforms}")
    training = load_training_records(
        args.metadata_csv, args.image_root, endpoint_tracin["indices"]
    )
    if args.top_k > len(training):
        raise ValueError("--top-k exceeds the training-set size")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, Any]] = []
    figure_count = 0
    for query_index in args.query_indices:
        query = queries[query_index]
        for transform_name in args.transforms:
            transform_slot = endpoint_tracin["transforms"].index(transform_name)
            banks = build_banks(
                endpoint_tracin, endpoint_pollute, retrac, transform_slot
            )
            figure = make_figure(
                query=query,
                transform_name=transform_name,
                banks=banks,
                training=training,
                top_k=args.top_k,
                tile_size=args.tile_size,
                csv_rows=csv_rows,
            )
            output = (
                args.output_dir
                / f"Q{query_index:03d}_{transform_name}_top{args.top_k}_nine_rows.jpg"
            )
            figure.save(output, quality=94, subsampling=0)
            figure_count += 1
            print(f"[saved {figure_count}] {output}", flush=True)

    csv_path = args.output_dir / f"top{args.top_k}_rankings.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    manifest = {
        "query_indices": args.query_indices,
        "transforms": args.transforms,
        "top_k": args.top_k,
        "rows_per_figure": 9,
        "figure_count": figure_count,
        "endpoint_tracin": str((args.endpoint_root / "endpoint_tracin.npz").resolve()),
        "endpoint_pollute": str(
            (args.endpoint_root / "endpoint_pollute_timestamp_sum_squared.npz").resolve()
        ),
        "retrac": str(args.retrac_path.resolve()),
        "ranking": {
            "endpoint_pollute": "timestamp_sum_squared descending",
            "endpoint_tracin": "linear ascending",
            "retrac": "linear ascending",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[done] figures={figure_count} ranking_rows={len(csv_rows)} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
