#!/usr/bin/env python3
"""Print query artifact paths in requested query order for shell launchers."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from run_expected_residual_jacobian_scores import query_artifact_path


def parse_query_ids(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-ids", required=True)
    parser.add_argument("--namespace", required=True)
    args = parser.parse_args()
    for query_id in parse_query_ids(args.query_ids):
        path = query_artifact_path(
            args.experiment,
            args.train_seed,
            args.epochs,
            query_id,
            args.namespace,
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        print(path)


if __name__ == "__main__":
    main()
