import argparse
from dataclasses import replace
from pathlib import Path

from .config import ArtBenchConfig
from .data import (
    load_metadata,
    make_artbench2_splits,
    validate_image_paths,
    write_prepared_data,
)


def parse_args() -> argparse.Namespace:
    defaults = ArtBenchConfig()
    parser = argparse.ArgumentParser(
        description="Prepare the paper-faithful D-TRAK ArtBench-2 split."
    )
    parser.add_argument("--metadata-csv", type=Path, default=defaults.metadata_csv)
    parser.add_argument("--image-root", type=Path, default=defaults.image_root)
    parser.add_argument("--index-root", type=Path, default=defaults.index_root)
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Create indices without checking whether every image is present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        ArtBenchConfig(),
        metadata_csv=args.metadata_csv.resolve(),
        image_root=args.image_root.resolve(),
        index_root=args.index_root.resolve(),
    )
    frame = load_metadata(config)
    train, validation = make_artbench2_splits(frame, config)
    if not args.skip_image_check:
        validate_image_paths(train)
        validate_image_paths(validation)
    write_prepared_data(frame, train, validation, config)
    print(f"Wrote ArtBench-2 indices to {config.index_root}")
    print(f"train={len(train)} validation={len(validation)}")
    print(train["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

