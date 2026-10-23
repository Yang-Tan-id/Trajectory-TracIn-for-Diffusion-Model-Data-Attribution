"""Create paired Base-vs-LoRA image comparisons from generation manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create side-by-side comparisons for two paired query manifests."
    )
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--lora-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--pairs-per-page", type=int, default=12)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"manifest must be a non-empty JSON list: {path}")
    records = []
    for index, item in enumerate(payload):
        image_path = Path(item["file"])
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        record = dict(item)
        record["image_path"] = image_path
        record["manifest_index"] = index
        records.append(record)
    return records


def pair_records(
    base_records: list[dict[str, Any]], lora_records: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if len(base_records) != len(lora_records):
        raise ValueError(
            f"manifest lengths differ: base={len(base_records)}, lora={len(lora_records)}"
        )
    pairs = []
    for position, base in enumerate(base_records):
        source_index = base.get("source_query_index")
        lora_index = position if source_index is None else int(source_index)
        if not (0 <= lora_index < len(lora_records)):
            raise ValueError(f"base record {position} has invalid source_query_index")
        lora = lora_records[lora_index]
        if str(base.get("prompt")) != str(lora.get("prompt")):
            raise ValueError(f"pair {position} prompt mismatch")
        if int(base["seed"]) != int(lora["seed"]):
            raise ValueError(f"pair {position} seed mismatch")
        pairs.append((base, lora))
    return pairs


def fit_image(path: Path, size: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def shortened(text: str, maximum: int = 72) -> str:
    return text if len(text) <= maximum else text[: maximum - 3] + "..."


def make_pair_image(
    query_index: int,
    base: dict[str, Any],
    lora: dict[str, Any],
    tile_size: int,
) -> Image.Image:
    header_height = 58
    footer_height = 34
    canvas = Image.new(
        "RGB", (tile_size * 2, header_height + tile_size + footer_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 6),
        f"Query {query_index:03d} | seed={int(base['seed'])}",
        fill="black",
    )
    draw.text((8, 27), shortened(str(base["prompt"])), fill="black")
    canvas.paste(fit_image(Path(base["image_path"]), tile_size), (0, header_height))
    canvas.paste(
        fit_image(Path(lora["image_path"]), tile_size), (tile_size, header_height)
    )
    draw.text((tile_size // 2 - 14, header_height + tile_size + 8), "BASE", fill="black")
    draw.text(
        (tile_size + tile_size // 2 - 14, header_height + tile_size + 8),
        "LORA",
        fill="black",
    )
    return canvas


def write_overview_pages(
    pair_images: list[Image.Image], output_dir: Path, pairs_per_page: int
) -> None:
    columns = 3
    gap = 12
    pair_width, pair_height = pair_images[0].size
    for page_index, start in enumerate(range(0, len(pair_images), pairs_per_page)):
        selected = pair_images[start : start + pairs_per_page]
        rows = (len(selected) + columns - 1) // columns
        page = Image.new(
            "RGB",
            (
                columns * pair_width + (columns + 1) * gap,
                rows * pair_height + (rows + 1) * gap,
            ),
            (230, 230, 230),
        )
        for slot, image in enumerate(selected):
            x = gap + (slot % columns) * (pair_width + gap)
            y = gap + (slot // columns) * (pair_height + gap)
            page.paste(image, (x, y))
        page.save(output_dir / f"overview-{page_index:02d}.jpg", quality=92)


def main() -> None:
    args = parse_args()
    if args.tile_size < 32:
        raise ValueError("--tile-size must be at least 32")
    if args.pairs_per_page < 1:
        raise ValueError("--pairs-per-page must be positive")
    base_records = load_manifest(args.base_manifest)
    lora_records = load_manifest(args.lora_manifest)
    pairs = pair_records(base_records, lora_records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_root = args.output_dir / "pairs"
    pair_root.mkdir(parents=True, exist_ok=True)
    pair_images = []
    for query_index, (base, lora) in enumerate(tqdm(pairs, desc="Paired comparisons")):
        image = make_pair_image(query_index, base, lora, args.tile_size)
        image.save(pair_root / f"query-{query_index:03d}.jpg", quality=94)
        pair_images.append(image)
    write_overview_pages(pair_images, args.output_dir, args.pairs_per_page)
    manifest = {
        "base_manifest": str(args.base_manifest.resolve()),
        "lora_manifest": str(args.lora_manifest.resolve()),
        "pair_count": len(pairs),
        "pairing_validation": "exact prompt and actual seed",
        "pairs_per_page": args.pairs_per_page,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[done] pairs={len(pairs)} output={args.output_dir} ",
        f"overview_pages={(len(pairs) + args.pairs_per_page - 1) // args.pairs_per_page}",
        flush=True,
    )


if __name__ == "__main__":
    main()
