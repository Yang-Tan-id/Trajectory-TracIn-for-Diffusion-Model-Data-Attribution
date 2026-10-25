"""Replay queries through completed Top-K removal LoRAs and make an overview."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .config import ArtBenchConfig


DEFAULT_METHODS = (
    "endpoint_pollute_query_train_l2",
    "endpoint_tracin_raw",
    "endpoint_tracin_query_train_l2",
)


def parse_csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def parse_args() -> argparse.Namespace:
    defaults = ArtBenchConfig()
    root = defaults.index_root.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-manifest",
        type=Path,
        default=root / "outputs/artbench2-queries-60/manifest.json",
    )
    parser.add_argument(
        "--removal-root",
        type=Path,
        default=root / "outputs/topk-removal-adamw-q0-9",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs/topk-removal-previews-q0-9",
    )
    parser.add_argument("--query-indices", type=parse_csv_ints, default=list(range(10)))
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"query manifest must be a non-empty list: {path}")
    records: list[dict[str, Any]] = []
    for item in payload:
        record = dict(item)
        image_path = Path(record["file"])
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        record["file"] = str(image_path.resolve())
        records.append(record)
    return records


def generated_image(manifest_path: Path) -> Path:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"expected one generated record: {manifest_path}")
    image_path = Path(payload[0]["file"])
    if not image_path.is_absolute():
        image_path = manifest_path.parent / image_path
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    return image_path


def fit_image(path: Path, size: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def write_overview(
    *,
    output_path: Path,
    records: list[dict[str, Any]],
    query_indices: list[int],
    generated: dict[tuple[str, int], Path],
    missing: set[tuple[str, int]],
    tile_size: int,
) -> None:
    columns = ("FULL-DATA LORA",) + DEFAULT_METHODS
    label_height = 48
    query_width = 92
    gap = 6
    width = query_width + len(columns) * tile_size + (len(columns) + 1) * gap
    height = label_height + len(query_indices) * (tile_size + gap) + gap
    canvas = Image.new("RGB", (width, height), (235, 235, 235))
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(columns):
        x = query_width + gap + column * (tile_size + gap)
        draw.text((x + 4, 14), label, fill="black")

    for row, query_index in enumerate(query_indices):
        y = label_height + gap + row * (tile_size + gap)
        record = records[query_index]
        draw.text((8, y + 8), f"Q{query_index:02d}", fill="black")
        draw.text((8, y + 28), f"seed={record['seed']}", fill="black")
        canvas.paste(fit_image(Path(record["file"]), tile_size), (query_width + gap, y))
        for method_slot, method in enumerate(DEFAULT_METHODS, start=1):
            x = query_width + gap + method_slot * (tile_size + gap)
            key = (method, query_index)
            if key in generated:
                canvas.paste(fit_image(generated[key], tile_size), (x, y))
            else:
                draw.rectangle((x, y, x + tile_size, y + tile_size), fill=(205, 205, 205))
                status = "MISSING" if key in missing else "NOT GENERATED"
                draw.text((x + 12, y + tile_size // 2), status, fill=(130, 0, 0))
    canvas.save(output_path, quality=92)


def main() -> None:
    args = parse_args()
    if args.top_k < 1 or args.steps < 1 or args.tile_size < 64:
        raise ValueError("top-k and steps must be positive; tile-size must be >= 64")
    records = load_records(args.query_manifest)
    for query_index in args.query_indices:
        if not 0 <= query_index < len(records):
            raise IndexError(f"query index outside manifest: {query_index}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    replay_root = args.output_dir / "replay_manifests"
    replay_root.mkdir(parents=True, exist_ok=True)
    generated: dict[tuple[str, int], Path] = {}
    missing: set[tuple[str, int]] = set()

    for method in DEFAULT_METHODS:
        for query_index in args.query_indices:
            run_dir = (
                args.removal_root
                / method
                / f"query-{query_index:03d}"
                / f"topk-{args.top_k:04d}"
            )
            completion_path = run_dir / "TRAINING_COMPLETE"
            lora_dir = run_dir / "model/final"
            key = (method, query_index)
            if not completion_path.is_file() or not lora_dir.is_dir():
                missing.add(key)
                print(f"[missing] method={method} query={query_index}", flush=True)
                continue

            replay_path = replay_root / f"query-{query_index:03d}.json"
            replay_path.write_text(
                json.dumps([records[query_index]], indent=2) + "\n",
                encoding="utf-8",
            )
            sample_dir = args.output_dir / method / f"query-{query_index:03d}"
            sample_manifest = sample_dir / "manifest.json"
            if args.overwrite or not sample_manifest.is_file():
                command = [
                    sys.executable,
                    "-m",
                    "sd_artbench.generate",
                    "--lora-dir",
                    str(lora_dir),
                    "--replay-manifest",
                    str(replay_path),
                    "--output-dir",
                    str(sample_dir),
                    "--steps",
                    str(args.steps),
                    "--guidance-scale",
                    str(args.guidance_scale),
                    "--disable-safety-checker",
                ]
                print(
                    f"[generate] method={method} query={query_index} "
                    f"gpu={os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}",
                    flush=True,
                )
                subprocess.run(command, check=True)
            else:
                print(f"[reuse] method={method} query={query_index}", flush=True)
            generated[key] = generated_image(sample_manifest)

    overview_path = args.output_dir / "overview-q0-9.jpg"
    write_overview(
        output_path=overview_path,
        records=records,
        query_indices=args.query_indices,
        generated=generated,
        missing=missing,
        tile_size=args.tile_size,
    )
    summary = {
        "query_manifest": str(args.query_manifest.resolve()),
        "removal_root": str(args.removal_root.resolve()),
        "query_indices": args.query_indices,
        "methods": list(DEFAULT_METHODS),
        "generated_count": len(generated),
        "missing": [
            {"method": method, "query_index": query_index}
            for method, query_index in sorted(missing)
        ],
        "sampling": {
            "steps": args.steps,
            "scheduler": "DDIM eta=0",
            "guidance_scale": args.guidance_scale,
            "prompt_seed_source": str(args.query_manifest.resolve()),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[done] generated={len(generated)} missing={len(missing)} "
        f"overview={overview_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
