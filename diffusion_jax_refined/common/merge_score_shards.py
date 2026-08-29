from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_score_dir(path: Path) -> tuple[np.ndarray, np.ndarray]:
    scores_path = path / "scores.npy"
    indices_path = path / "score_indices.npy"
    if not scores_path.is_file():
        raise FileNotFoundError(str(scores_path))
    if not indices_path.is_file():
        raise FileNotFoundError(str(indices_path))
    scores = np.asarray(np.load(scores_path), dtype=np.float64).reshape(-1)
    indices = np.asarray(np.load(indices_path), dtype=np.int64).reshape(-1)
    if scores.shape[0] != indices.shape[0]:
        raise ValueError(f"{path}: scores length {scores.shape[0]} != score_indices length {indices.shape[0]}")
    return scores, indices


def merge_score_shards(shard_dirs: list[Path], output_dir: Path) -> None:
    scores_parts = []
    index_parts = []
    seen = set()
    for shard_dir in shard_dirs:
        scores, indices = load_score_dir(shard_dir)
        overlap = sorted(set(int(x) for x in indices) & seen)
        if overlap:
            raise ValueError(f"{shard_dir} overlaps previous shards at indices {overlap[:10]}")
        seen.update(int(x) for x in indices)
        scores_parts.append(scores)
        index_parts.append(indices)

    scores = np.concatenate(scores_parts, axis=0)
    indices = np.concatenate(index_parts, axis=0)
    order = np.argsort(indices)
    scores = scores[order]
    indices = indices[order]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "scores.npy", scores.astype(np.float64))
    np.save(output_dir / "score_indices.npy", indices.astype(np.int64))

    top_order = np.argsort(-scores)
    top = [
        {"rank": int(rank), "idx": int(indices[i]), "idx_1based": int(indices[i]) + 1, "score": float(scores[i])}
        for rank, i in enumerate(top_order[: min(2000, len(top_order))], start=1)
    ]
    with open(output_dir / "top_scores.json", "w") as handle:
        json.dump({"top": top, "num_scored": int(len(scores))}, handle, indent=2)
    with open(output_dir / "score_indices.json", "w") as handle:
        json.dump(
            {
                "score_indices": [int(x) for x in indices],
                "score_indices_1based": [int(x) + 1 for x in indices],
            },
            handle,
            indent=2,
        )
    with open(output_dir / "score_artifact_manifest.json", "w") as handle:
        json.dump(
            {
                "mode": "merged_score_shards",
                "score_dir": str(output_dir),
                "num_shards": len(shard_dirs),
                "shards": [str(path) for path in shard_dirs],
                "num_scores": int(len(scores)),
            },
            handle,
            indent=2,
        )
    print(f"[merge] wrote merged scores: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge attribution score shard directories.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    merge_score_shards(args.shards, args.output_dir)


if __name__ == "__main__":
    main()
