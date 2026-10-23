import argparse
import csv
import io
import shutil
from pathlib import Path

from datasets import Image as DatasetsImage
from datasets import load_dataset
from PIL import Image

from .config import ArtBenchConfig


FORMAT_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "GIF": ".gif",
    "BMP": ".bmp",
    "TIFF": ".tiff",
}


def parse_args() -> argparse.Namespace:
    defaults = ArtBenchConfig()
    parser = argparse.ArgumentParser(
        description="Download the 256 px ArtBench Hugging Face mirror and export ArtBench-2."
    )
    parser.add_argument("--dataset", default="zguo0525/ArtBench")
    parser.add_argument("--metadata-csv", type=Path, default=defaults.metadata_csv)
    parser.add_argument("--image-root", type=Path, default=defaults.image_root)
    return parser.parse_args()


def image_extension(payload: bytes) -> str:
    with Image.open(io.BytesIO(payload)) as image:
        return FORMAT_EXTENSIONS.get(image.format or "", ".jpg")


def export_image(image_record, destination_without_suffix: Path) -> Path:
    payload = image_record.get("bytes")
    source_path = image_record.get("path")
    if payload is not None:
        destination = destination_without_suffix.with_suffix(image_extension(payload))
        destination.write_bytes(payload)
        return destination
    if source_path:
        source = Path(source_path)
        destination = destination_without_suffix.with_suffix(source.suffix or ".jpg")
        shutil.copy2(source, destination)
        return destination
    raise ValueError("Hugging Face image record contains neither bytes nor a path")


def main() -> None:
    args = parse_args()
    config = ArtBenchConfig(
        metadata_csv=args.metadata_csv.resolve(),
        image_root=args.image_root.resolve(),
    )
    args.metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    for label in config.classes:
        (args.image_root / label).mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset)
    rows = []
    counts = {}
    for split_name in ("train", "test"):
        split = dataset[split_name]
        label_feature = split.features["label"]
        raw_split = split.cast_column("image", DatasetsImage(decode=False))
        for row_index, record in enumerate(raw_split):
            raw_label = record["label"]
            label = label_feature.int2str(raw_label) if isinstance(raw_label, int) else raw_label
            if label not in config.classes:
                continue
            stem = f"{split_name}_{row_index:05d}"
            destination = export_image(
                record["image"], args.image_root / label / stem
            )
            rows.append(
                {"name": destination.name, "label": label, "split": split_name}
            )
            counts[(split_name, label)] = counts.get((split_name, label), 0) + 1

    expected = {("train", label): 5000 for label in config.classes}
    expected.update({("test", label): 1000 for label in config.classes})
    if counts != expected:
        raise RuntimeError(f"Unexpected ArtBench-2 counts: {counts}; expected {expected}")

    with args.metadata_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} images to {args.image_root}")
    print(f"Wrote metadata to {args.metadata_csv}")
    for key in sorted(counts):
        print(f"{key[0]:5s} {key[1]:20s} {counts[key]}")


if __name__ == "__main__":
    main()

