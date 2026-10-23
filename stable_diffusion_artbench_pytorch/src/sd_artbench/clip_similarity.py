"""Rank ArtBench training images by CLIP similarity to generated queries."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

from .config import ArtBenchConfig


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    metadata: dict


def parse_args() -> argparse.Namespace:
    defaults = ArtBenchConfig()
    root = defaults.index_root.parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Compute cosine similarity between generated query endpoints and "
            "the exact 5,000-image ArtBench training split using CLIP."
        )
    )
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, default=defaults.metadata_csv)
    parser.add_argument("--image-root", type=Path, default=defaults.image_root)
    parser.add_argument(
        "--index-path", type=Path, default=defaults.index_root / "idx-train.pkl"
    )
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--output-dir", type=Path, default=root / "outputs" / "clip-similarity"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--contact-sheet-top-k", type=int, default=10)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="Reuse compatible query/train embedding caches in the output directory.",
    )
    parser.add_argument("--skip-contact-sheets", action="store_true")
    return parser.parse_args()


def load_queries(manifest_path: Path) -> list[ImageRecord]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"query manifest must be a non-empty list: {manifest_path}")
    records = []
    for query_index, item in enumerate(payload):
        path = Path(item["file"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        if not path.is_file():
            raise FileNotFoundError(path)
        metadata = dict(item)
        metadata["query_index"] = query_index
        records.append(ImageRecord(path=path.resolve(), metadata=metadata))
    return records


def load_training_records(
    metadata_csv: Path, image_root: Path, index_path: Path
) -> list[ImageRecord]:
    frame = pd.read_csv(metadata_csv)
    with index_path.open("rb") as handle:
        indices = [int(value) for value in pickle.load(handle)]
    selected = frame.loc[indices]
    records = []
    for train_position, (source_index, row) in enumerate(selected.iterrows()):
        path = image_root / str(row["label"]) / str(row["name"])
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            ImageRecord(
                path=path.resolve(),
                metadata={
                    "train_position": train_position,
                    "source_index": int(source_index),
                    "label": str(row["label"]),
                    "name": str(row["name"]),
                },
            )
        )
    return records


def batched(
    values: Sequence[ImageRecord], batch_size: int
) -> Iterable[Sequence[ImageRecord]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def encode_images(
    records: Sequence[ImageRecord],
    *,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    description: str,
) -> np.ndarray:
    features = []
    batches = batched(records, batch_size)
    batch_count = (len(records) + batch_size - 1) // batch_size
    for batch in tqdm(batches, total=batch_count, desc=description):
        images = []
        try:
            for record in batch:
                with Image.open(record.path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
            with torch.inference_mode():
                embedding = model.get_image_features(pixel_values=pixel_values)
                embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
            features.append(embedding.cpu().numpy())
        finally:
            for image in images:
                image.close()
    return np.concatenate(features, axis=0).astype(np.float32, copy=False)


def cache_paths(records: Sequence[ImageRecord]) -> np.ndarray:
    return np.asarray([str(record.path) for record in records])


def load_embedding_cache(
    path: Path, records: Sequence[ImageRecord], model_name: str
) -> np.ndarray | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as payload:
        cached_model = str(payload["model"].item())
        cached_paths = payload["paths"]
        features = payload["features"]
    expected_paths = cache_paths(records)
    if cached_model != model_name or not np.array_equal(cached_paths, expected_paths):
        return None
    if features.ndim != 2 or features.shape[0] != len(records):
        return None
    return np.asarray(features, dtype=np.float32)


def save_embedding_cache(
    path: Path,
    records: Sequence[ImageRecord],
    model_name: str,
    features: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        model=np.asarray(model_name),
        paths=cache_paths(records),
        features=np.asarray(features, dtype=np.float32),
    )


def topk_indices(similarities: np.ndarray, top_k: int) -> np.ndarray:
    if similarities.ndim != 2:
        raise ValueError("similarities must have shape [queries, training images]")
    if not (1 <= top_k <= similarities.shape[1]):
        raise ValueError(f"top-k must be in [1, {similarities.shape[1]}]")
    candidates = np.argpartition(-similarities, top_k - 1, axis=1)[:, :top_k]
    candidate_scores = np.take_along_axis(similarities, candidates, axis=1)
    order = np.argsort(-candidate_scores, axis=1, kind="stable")
    return np.take_along_axis(candidates, order, axis=1)


def write_rankings(
    path: Path,
    queries: Sequence[ImageRecord],
    training: Sequence[ImageRecord],
    similarities: np.ndarray,
    rankings: np.ndarray,
) -> None:
    fieldnames = [
        "query_index",
        "query_file",
        "prompt",
        "seed",
        "requested_seed",
        "rank",
        "train_position",
        "source_index",
        "train_file",
        "train_label",
        "similarity",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for query_index, positions in enumerate(rankings):
            query = queries[query_index]
            for rank, train_position in enumerate(positions, start=1):
                train = training[int(train_position)]
                writer.writerow(
                    {
                        "query_index": query_index,
                        "query_file": str(query.path),
                        "prompt": query.metadata.get("prompt", ""),
                        "seed": query.metadata.get("seed", ""),
                        "requested_seed": query.metadata.get("requested_seed", ""),
                        "rank": rank,
                        "train_position": train.metadata["train_position"],
                        "source_index": train.metadata["source_index"],
                        "train_file": str(train.path),
                        "train_label": train.metadata["label"],
                        "similarity": f"{similarities[query_index, train_position]:.9f}",
                    }
                )


def thumbnail(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def write_contact_sheets(
    output_dir: Path,
    queries: Sequence[ImageRecord],
    training: Sequence[ImageRecord],
    similarities: np.ndarray,
    rankings: np.ndarray,
    count: int,
) -> None:
    sheet_root = output_dir / "contact_sheets"
    sheet_root.mkdir(parents=True, exist_ok=True)
    tile_size = 192
    caption_height = 52
    columns = 4
    for query_index, query in enumerate(tqdm(queries, desc="Contact sheets")):
        selected = rankings[query_index, :count]
        tile_count = 1 + len(selected)
        rows = (tile_count + columns - 1) // columns
        sheet = Image.new(
            "RGB", (columns * tile_size, rows * (tile_size + caption_height)), "white"
        )
        draw = ImageDraw.Draw(sheet)
        records = [(query.path, "QUERY", "")]
        records.extend(
            (
                training[int(position)].path,
                f"#{rank} pos={int(position)} "
                f"src={training[int(position)].metadata['source_index']}",
                f"{training[int(position)].metadata['label']}  "
                f"sim={similarities[query_index, position]:.4f}",
            )
            for rank, position in enumerate(selected, start=1)
        )
        for slot, (path, line_one, line_two) in enumerate(records):
            x = (slot % columns) * tile_size
            y = (slot // columns) * (tile_size + caption_height)
            sheet.paste(thumbnail(path, tile_size), (x, y))
            draw.text((x + 4, y + tile_size + 4), line_one, fill="black")
            draw.text((x + 4, y + tile_size + 23), line_two, fill="black")
        sheet.save(sheet_root / f"query-{query_index:03d}.jpg", quality=92)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    if not (0 <= args.contact_sheet_top_k <= args.top_k):
        raise ValueError("--contact-sheet-top-k must be between zero and --top-k")

    queries = load_queries(args.query_manifest)
    training = load_training_records(args.metadata_csv, args.image_root, args.index_path)
    if args.top_k > len(training):
        raise ValueError("--top-k cannot exceed the number of training images")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    query_cache = args.output_dir / "query_embeddings.npz"
    train_cache = args.output_dir / "train_embeddings.npz"
    query_features = (
        load_embedding_cache(query_cache, queries, args.model)
        if args.reuse_embeddings
        else None
    )
    if query_features is None:
        query_features = encode_images(
            queries,
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            description="CLIP queries",
        )
        save_embedding_cache(query_cache, queries, args.model, query_features)
    else:
        print(f"[cache] query embeddings: {query_cache}", flush=True)

    train_features = (
        load_embedding_cache(train_cache, training, args.model)
        if args.reuse_embeddings
        else None
    )
    if train_features is None:
        train_features = encode_images(
            training,
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            description="CLIP train",
        )
        save_embedding_cache(train_cache, training, args.model, train_features)
    else:
        print(f"[cache] train embeddings: {train_cache}", flush=True)

    similarities = np.matmul(query_features, train_features.T).astype(np.float32)
    rankings = topk_indices(similarities, args.top_k)
    np.save(args.output_dir / "similarities.npy", similarities)
    write_rankings(
        args.output_dir / "topk.csv", queries, training, similarities, rankings
    )
    if not args.skip_contact_sheets and args.contact_sheet_top_k:
        write_contact_sheets(
            args.output_dir,
            queries,
            training,
            similarities,
            rankings,
            args.contact_sheet_top_k,
        )
    manifest = {
        "method": "clip_image_to_image_cosine_similarity",
        "model": args.model,
        "query_manifest": str(args.query_manifest.resolve()),
        "index_path": str(args.index_path.resolve()),
        "query_count": len(queries),
        "train_count": len(training),
        "top_k": args.top_k,
        "similarities_shape": list(similarities.shape),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[done] similarities={similarities.shape} top-k={args.top_k} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
