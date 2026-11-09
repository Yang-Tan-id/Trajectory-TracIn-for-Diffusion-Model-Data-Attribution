#!/usr/bin/env python3
"""Create the selected-query ArtBench five-method endpoint grid.

The grid intentionally omits query identifiers and sampling seeds.  Rows are
ordered by ``QUERY_INDICES``.  The full-data endpoint is shown first and is
visually separated from the five counterfactual methods.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


QUERY_INDICES = (6, 8, 12, 22, 39, 54)
COLUMNS = (
    ("Endpoint", ("full-data.png",)),
    ("PINT", ("endpoint_pollute_query_train_l2.png",)),
    ("TracInCP", ("endpoint_tracin_raw.png",)),
    ("GAS", ("endpoint_tracin_query_train_l2.png",)),
    ("ReTrac", ("retrac_query_train_l2.png",)),
    # The first name is the method identifier used by the Q0--Q59 run.  The
    # aliases keep the plotter compatible with older merged output folders.
    (
        "Random",
        (
            "random_full5000_seeded.png",
            "random.png",
            "random-removal.png",
            "random_removal.png",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=project / "outputs/topk-removal-five-method-metrics-q0-59/combined",
        help="Directory containing query-000 through query-059.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project
        / "outputs/topk-removal-five-method-metrics-q0-59"
        / "artbench_selected_q6_q8_q12_q22_q39_q54_five_method_endpoints.png",
    )
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument(
        "--gap",
        type=int,
        default=18,
        help="Horizontal and vertical gap between method tiles.",
    )
    parser.add_argument(
        "--endpoint-gap",
        type=int,
        default=44,
        help="Horizontal gap after Endpoint; must be at least twice --gap.",
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
) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=text_font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        ((left + right - width) / 2, (top + bottom - height) / 2 - bounds[1]),
        text,
        font=text_font,
        fill="#202124",
    )


def resolve_image(query_dir: Path, candidates: tuple[str, ...]) -> Path:
    for filename in candidates:
        path = query_dir / filename
        if path.is_file():
            return path
    expected = ", ".join(str(query_dir / name) for name in candidates)
    raise FileNotFoundError(f"none of the expected endpoint images exists: {expected}")


def fit_tile(path: Path, size: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size), "white")
    tile.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return tile


def main() -> None:
    args = parse_args()
    if args.tile_size < 128:
        raise ValueError("tile-size must be at least 128")
    if args.gap < 0:
        raise ValueError("gap must be nonnegative")
    if args.endpoint_gap < 2 * args.gap:
        raise ValueError("endpoint-gap must be at least twice gap")

    outer = 14
    header_height = 64
    row_height = args.tile_size + args.gap
    horizontal_gaps = args.endpoint_gap + args.gap * (len(COLUMNS) - 2)
    width = outer * 2 + len(COLUMNS) * args.tile_size + horizontal_gaps
    height = outer * 2 + header_height + len(QUERY_INDICES) * row_height - args.gap

    canvas = Image.new("RGB", (width, height), "#F7F7F7")
    draw = ImageDraw.Draw(canvas)
    header_font = font(18, bold=True)

    column_lefts = [outer]
    for column in range(1, len(COLUMNS)):
        preceding_gap = args.endpoint_gap if column == 1 else args.gap
        column_lefts.append(column_lefts[-1] + args.tile_size + preceding_gap)

    for left, (label, _filenames) in zip(column_lefts, COLUMNS):
        centered_text(
            draw,
            (left, outer, left + args.tile_size, outer + header_height),
            label,
            text_font=header_font,
        )

    for row, query_index in enumerate(QUERY_INDICES):
        query_dir = args.image_root / f"query-{query_index:03d}"
        top = outer + header_height + row * row_height
        for left, (_label, filenames) in zip(column_lefts, COLUMNS):
            path = resolve_image(query_dir, filenames)
            canvas.paste(fit_tile(path, args.tile_size), (left, top))
            draw.rectangle(
                (left, top, left + args.tile_size - 1, top + args.tile_size - 1),
                outline="#D0D0D0",
                width=1,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=95)
    print(f"[saved] {args.output}")
    print("[rows] " + ",".join(f"Q{query}" for query in QUERY_INDICES))
    print("[columns] " + " | ".join(label for label, _ in COLUMNS))
    print(f"[gaps] methods={args.gap}px endpoint={args.endpoint_gap}px")
    print("[row labels] omitted")
    print("[seeds] omitted")


if __name__ == "__main__":
    main()
