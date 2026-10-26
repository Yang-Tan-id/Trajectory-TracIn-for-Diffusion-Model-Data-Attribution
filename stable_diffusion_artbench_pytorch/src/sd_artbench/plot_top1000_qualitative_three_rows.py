#!/usr/bin/env python3
"""Create a compact three-row ArtBench Top-K removal comparison grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLUMNS = (
    ("Output", "full-data.png"),
    ("PINT (Ours)", "endpoint_pollute_query_train_l2.png"),
    ("Endpoint TracIn", "endpoint_tracin_query_train_l2.png"),
    ("ReTrac", "retrac_query_train_l2.png"),
)


def parse_indices(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one query index")
    return result


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=project / "outputs/topk-removal-trajectory-mse-q0-9",
    )
    parser.add_argument(
        "--query-manifest",
        type=Path,
        default=project / "outputs/artbench2-queries-60/manifest.json",
    )
    parser.add_argument("--query-indices", type=parse_indices, default=[0, 1, 2])
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=project
        / "outputs/topk-removal-qualitative-paper/artbench_top1000_q0_q2.png",
    )
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


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    text_font: ImageFont.ImageFont,
    fill: str,
) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=text_font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        ((left + right - width) / 2, (top + bottom - height) / 2 - bounds[1]),
        text,
        font=text_font,
        fill=fill,
    )


def fit_tile(path: Path, size: int) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size), "white")
    tile.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return tile


def load_seeds(path: Path) -> dict[int, int]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {index: int(record["seed"]) for index, record in enumerate(payload)}


def main() -> None:
    args = parse_args()
    if len(args.query_indices) != 3:
        raise ValueError("the paper grid must contain exactly three query rows")
    if args.tile_size < 128:
        raise ValueError("tile-size must be at least 128")

    seeds = load_seeds(args.query_manifest)
    label_width = 112
    header_height = 64
    gap = 7
    outer = 14
    column_width = args.tile_size + gap
    row_height = args.tile_size + gap
    width = outer * 2 + label_width + len(COLUMNS) * column_width
    height = outer * 2 + header_height + len(args.query_indices) * row_height

    background = "#F7F7F7"
    output_fill = "#E8F0F7"
    output_border = "#4C78A8"
    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    header_font = font(17, bold=True)
    row_font = font(15, bold=True)
    seed_font = font(12)

    output_left = outer + label_width
    output_right = output_left + args.tile_size
    draw.rounded_rectangle(
        (output_left - 5, outer - 4, output_right + 5, height - outer + 1),
        radius=7,
        fill=output_fill,
        outline=output_border,
        width=4,
    )

    for column, (label, _) in enumerate(COLUMNS):
        left = outer + label_width + column * column_width
        centered_text(
            draw,
            (left, outer, left + args.tile_size, outer + header_height),
            label,
            text_font=header_font,
            fill=output_border if column == 0 else "#202124",
        )

    for row, query_index in enumerate(args.query_indices):
        query_dir = args.image_root / f"query-{query_index:03d}"
        top = outer + header_height + row * row_height
        label_top = top + args.tile_size // 2 - 25
        draw.text((outer + 6, label_top), f"Q{query_index:02d}", font=row_font, fill="#202124")
        if query_index in seeds:
            draw.text(
                (outer + 6, label_top + 24),
                f"seed={seeds[query_index]}",
                font=seed_font,
                fill="#5F6368",
            )
        for column, (_, filename) in enumerate(COLUMNS):
            left = outer + label_width + column * column_width
            tile = fit_tile(query_dir / filename, args.tile_size)
            canvas.paste(tile, (left, top))
            outline = output_border if column == 0 else "#D0D0D0"
            draw.rectangle(
                (left, top, left + args.tile_size - 1, top + args.tile_size - 1),
                outline=outline,
                width=3 if column == 0 else 1,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=95)
    print(f"[saved] {args.output}")
    print(f"[rows] {','.join(f'Q{value}' for value in args.query_indices)}")
    print("[columns] " + " | ".join(label for label, _ in COLUMNS))
    print("[highlight] Output")


if __name__ == "__main__":
    main()
