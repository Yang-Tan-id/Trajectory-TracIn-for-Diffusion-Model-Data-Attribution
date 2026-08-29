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


def require_same(name: str, payloads: list[dict[str, np.ndarray]]) -> np.ndarray:
    first = np.asarray(payloads[0][name])
    for idx, payload in enumerate(payloads[1:], start=2):
        value = np.asarray(payload[name])
        if first.shape != value.shape or not np.array_equal(first, value):
            raise ValueError(f"shard {idx} has different {name}")
    return first


def merge_das_global_gram(shard_paths: list[Path], output: Path) -> None:
    payloads = [load_npz(path) for path in shard_paths]
    for path, payload in zip(shard_paths, payloads):
        for key in ("train_features", "score_indices", "gram_undamped"):
            if key not in payload:
                raise KeyError(f"{path} is missing {key}")

    common = {
        key: require_same(key, payloads)
        for key in ("ckpt_indices", "timesteps", "mc_indices", "damping", "damping_sweep_values", "proj_dim")
        if key in payloads[0]
    }
    gram_undamped = np.zeros_like(np.asarray(payloads[0]["gram_undamped"], dtype=np.float32))
    seen = set()
    for path, payload in zip(shard_paths, payloads):
        idx = np.asarray(payload["score_indices"], dtype=np.int64).reshape(-1)
        train = np.asarray(payload["train_features"])
        if train.ndim != 3:
            raise ValueError(f"{path} train_features must be rank 3, got {train.shape}")
        if train.shape[1] != len(idx):
            raise ValueError(f"{path} train feature count {train.shape[1]} != score_indices {len(idx)}")
        overlap = sorted(set(int(x) for x in idx) & seen)
        if overlap:
            raise ValueError(f"{path} overlaps previous shards at indices {overlap[:10]}")
        seen.update(int(x) for x in idx)
        gram_undamped += np.asarray(payload["gram_undamped"], dtype=np.float32)

    damping = float(np.asarray(common.get("damping", np.asarray(0.0))).reshape(()))
    eye = np.eye(gram_undamped.shape[-1], dtype=np.float32)
    gram = gram_undamped + damping * eye[None, :, :]

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        gram=gram.astype(np.float32),
        gram_undamped=gram_undamped.astype(np.float32),
        **common,
        shard_paths=np.asarray([str(path) for path in shard_paths]),
        num_score_indices=np.asarray(len(seen), dtype=np.int64),
    )
    with open(output.with_suffix(output.suffix + ".manifest.json"), "w") as handle:
        json.dump(
            {
                "output": str(output),
                "num_shards": len(shard_paths),
                "shards": [str(path) for path in shard_paths],
                "num_terms": int(gram_undamped.shape[0]),
                "proj_dim": int(gram_undamped.shape[-1]),
                "num_score_indices": int(len(seen)),
            },
            handle,
            indent=2,
        )
    print(f"[saved] merged DAS global gram artifact: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge DAS train shards into a global Gram artifact.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    merge_das_global_gram(args.shards, args.output)


if __name__ == "__main__":
    main()
