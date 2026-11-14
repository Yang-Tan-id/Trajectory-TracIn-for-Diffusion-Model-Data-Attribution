#!/usr/bin/env python3
"""Plot full-data, five-method, and DAS counterfactual ArtBench outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BASELINE_COLUMNS = (
    ("Output", ("full-data.png",)),
    ("PINT", ("endpoint_pollute_query_train_l2.png",)),
    ("TracInCP", ("endpoint_tracin_raw.png",)),
    ("GAS", ("endpoint_tracin_query_train_l2.png",)),
    ("ReTrac", ("retrac_query_train_l2.png",)),
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
DAS_COLUMN = ("DAS", ("das_lambda_1000.png",))


def parse_query_indices(value: str) -> tuple[int, ...]:
    indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not indices:
        raise argparse.ArgumentTypeError("expected at least one query index")
    if len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError("query indices must be unique")
    invalid = [index for index in indices if not 0 <= index <= 59]
    if invalid:
        raise argparse.ArgumentTypeError(f"query indices must be Q0--Q59: {invalid}")
    return indices


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=project / "outputs/topk-removal-five-method-metrics-q0-59/combined",
    )
    parser.add_argument(
        "--das-root",
        type=Path,
        default=(
            project
            / "outputs/topk-removal-das-lambda1000-q0-59"
            / "metrics/q0_q27"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            project
            / "outputs/topk-removal-das-lambda1000-q0-59"
            / "comparison/q0_q27"
            / "q0_q27_output_pint_tracincp_gas_retrac_random_das.png"
        ),
    )
    parser.add_argument(
        "--query-indices",
        type=parse_query_indices,
        default=tuple(range(28)),
        help="Comma-separated query IDs in row order (default: Q0--Q27).",
    )
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--gap", type=int, default=18)
    parser.add_argument(
        "--output-gap",
        type=int,
        default=44,
        help="Gap after Output; must be at least twice --gap.",
    )
    return parser.parse_args()


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def resolve_image(directory: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        "none of the expected images exists: "
        + ", ".join(str(directory / name) for name in names)
    )


def fit_tile(path: Path, size: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size), "white")
    tile.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return tile


def centered_header(
    draw: ImageDraw.ImageDraw,
    left: int,
    width: int,
    top: int,
    height: int,
    label: str,
    header_font: ImageFont.ImageFont,
) -> None:
    bounds = draw.textbbox((0, 0), label, font=header_font)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    draw.text(
        (
            left + (width - text_width) / 2,
            top + (height - text_height) / 2 - bounds[1],
        ),
        label,
        font=header_font,
        fill="#202124",
    )


def main() -> None:
    args = parse_args()
    if args.tile_size < 128:
        raise ValueError("tile-size must be at least 128")
    if args.gap < 0:
        raise ValueError("gap must be nonnegative")
    if args.output_gap < 2 * args.gap:
        raise ValueError("output-gap must be at least twice gap")

    columns = BASELINE_COLUMNS + (DAS_COLUMN,)
    outer = 14
    header_height = 64
    row_height = args.tile_size + args.gap
    width = (
        outer * 2
        + len(columns) * args.tile_size
        + args.output_gap
        + args.gap * (len(columns) - 2)
    )
    height = (
        outer * 2
        + header_height
        + len(args.query_indices) * row_height
        - args.gap
    )
    canvas = Image.new("RGB", (width, height), "#F7F7F7")
    draw = ImageDraw.Draw(canvas)
    header_font = font(18)

    column_lefts = [outer]
    for column in range(1, len(columns)):
        gap = args.output_gap if column == 1 else args.gap
        column_lefts.append(column_lefts[-1] + args.tile_size + gap)

    for left, (label, _names) in zip(column_lefts, columns):
        centered_header(
            draw, left, args.tile_size, outer, header_height, label, header_font
        )

    for row, query in enumerate(args.query_indices):
        baseline_dir = args.baseline_root / f"query-{query:03d}"
        das_dir = args.das_root / f"query-{query:03d}"
        top = outer + header_height + row * row_height
        for column, (left, (_label, names)) in enumerate(zip(column_lefts, columns)):
            directory = das_dir if column == len(columns) - 1 else baseline_dir
            path = resolve_image(directory, names)
            canvas.paste(fit_tile(path, args.tile_size), (left, top))
            draw.rectangle(
                (left, top, left + args.tile_size - 1, top + args.tile_size - 1),
                outline="#D0D0D0",
                width=1,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=95)
    print(f"[saved] {args.output}")
    print("[rows] " + ",".join(f"Q{query}" for query in args.query_indices))
    print("[columns] " + " | ".join(label for label, _names in columns))
    print(f"[gaps] methods={args.gap}px output={args.output_gap}px")
    print("[row labels] omitted")
    print("[seeds] omitted")


if __name__ == "__main__":
    main()
