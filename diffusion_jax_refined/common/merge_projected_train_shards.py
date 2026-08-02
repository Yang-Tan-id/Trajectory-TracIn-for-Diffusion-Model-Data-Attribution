from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def shard_sort_key(payload: dict[str, np.ndarray]) -> int:
    indices = np.asarray(payload["score_indices"], dtype=np.int64).reshape(-1)
    if len(indices) == 0:
        raise ValueError("empty score_indices shard")
    return int(indices.min())


def require_same(name: str, payloads: list[dict[str, np.ndarray]]) -> np.ndarray:
    first = np.asarray(payloads[0][name])
    for idx, payload in enumerate(payloads[1:], start=2):
        value = np.asarray(payload[name])
        if first.shape != value.shape or not np.array_equal(first, value):
            raise ValueError(f"shard {idx} has different {name}")
    return first


def merge_train_shards(shard_paths: list[Path], output: Path) -> None:
    payloads = [load_npz(path) for path in shard_paths]
    for path, payload in zip(shard_paths, payloads):
        if "train_features" not in payload:
            raise KeyError(f"{path} is missing train_features")
        if "score_indices" not in payload:
            raise KeyError(f"{path} is missing score_indices")

    ordered = sorted(zip(shard_paths, payloads), key=lambda item: shard_sort_key(item[1]))
    ordered_paths = [path for path, _ in ordered]
    ordered_payloads = [payload for _, payload in ordered]

    term_keys = [key for key in ("ckpt_indices", "timesteps", "snapshot_positions", "term_weights") if key in ordered_payloads[0]]
    common = {key: require_same(key, ordered_payloads) for key in term_keys}
    if "proj_dim" in ordered_payloads[0]:
        common["proj_dim"] = require_same("proj_dim", ordered_payloads)

    features = []
    indices = []
    seen = set()
    for path, payload in zip(ordered_paths, ordered_payloads):
        train = np.asarray(payload["train_features"], dtype=np.float32)
        idx = np.asarray(payload["score_indices"], dtype=np.int64).reshape(-1)
        if train.ndim != 3:
            raise ValueError(f"{path} train_features must be rank 3, got {train.shape}")
        if train.shape[1] != len(idx):
            raise ValueError(f"{path} train feature count {train.shape[1]} != score_indices {len(idx)}")
        overlap = sorted(set(int(x) for x in idx) & seen)
        if overlap:
            raise ValueError(f"{path} overlaps previous shards at indices {overlap[:10]}")
        seen.update(int(x) for x in idx)
        features.append(train)
        indices.append(idx)

    merged_features = np.concatenate(features, axis=1).astype(np.float32)
    merged_indices = np.concatenate(indices, axis=0).astype(np.int64)
    order = np.argsort(merged_indices)
    merged_features = merged_features[:, order, :]
    merged_indices = merged_indices[order]

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        train_features=merged_features,
        score_indices=merged_indices,
        **common,
        shard_paths=np.asarray([str(path) for path in ordered_paths]),
    )
    with open(output.with_suffix(output.suffix + ".manifest.json"), "w") as handle:
        json.dump(
            {
                "output": str(output),
                "num_shards": len(ordered_paths),
                "shards": [str(path) for path in ordered_paths],
                "num_terms": int(merged_features.shape[0]),
                "num_points": int(merged_features.shape[1]),
                "proj_dim": int(merged_features.shape[2]),
                "score_indices_min": int(merged_indices.min()),
                "score_indices_max": int(merged_indices.max()),
            },
            handle,
            indent=2,
        )
    print(f"[saved] merged train artifact: {output}")
    print(f"[saved] merged shape: {merged_features.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge projected Traj-TracIn train feature shards.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("shards", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_train_shards(args.shards, args.output)


if __name__ == "__main__":
    main()
