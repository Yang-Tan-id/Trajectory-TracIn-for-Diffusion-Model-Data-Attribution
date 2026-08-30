from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _as_1d(data: np.lib.npyio.NpzFile, key: str, dtype) -> np.ndarray:
    if key not in data:
        raise KeyError(f"missing required key {key}")
    return np.asarray(data[key], dtype=dtype).reshape(-1)


def aggregate_train_artifact(input_path: Path, output_path: Path) -> None:
    with np.load(input_path, allow_pickle=False) as data:
        train = np.asarray(data["train_features"], dtype=np.float64)
        if train.ndim != 3:
            raise ValueError(f"{input_path}: train_features must be rank 3, got {train.shape}")
        score_indices = _as_1d(data, "score_indices", np.int64)
        ckpt_indices = _as_1d(data, "ckpt_indices", np.int32)
        term_weights = np.asarray(
            data["term_weights"] if "term_weights" in data else np.ones((train.shape[0],), dtype=np.float64),
            dtype=np.float64,
        ).reshape(-1)
        if ckpt_indices.shape[0] != train.shape[0] or term_weights.shape[0] != train.shape[0]:
            raise ValueError(f"{input_path}: term metadata length does not match train_features")

        unique_ckpts = np.unique(ckpt_indices)
        out_features = []
        out_weights = []
        for ckpt in unique_ckpts:
            mask = ckpt_indices == ckpt
            weights = term_weights[mask]
            weight_sum = float(weights.sum())
            if abs(weight_sum) <= 1e-30:
                weights = np.full((int(mask.sum()),), 1.0 / float(mask.sum()), dtype=np.float64)
                weight_sum = 1.0
            normalized = weights / weight_sum
            out_features.append(np.tensordot(normalized, train[mask], axes=(0, 0)).astype(np.float32))
            out_weights.append(weight_sum)

        payload = {
            "train_features": np.stack(out_features, axis=0).astype(np.float32),
            "score_indices": score_indices.astype(np.int64),
            "ckpt_indices": unique_ckpts.astype(np.int32),
            "timesteps": np.full((len(unique_ckpts),), -1, dtype=np.int32),
            "snapshot_positions": np.full((len(unique_ckpts),), -1, dtype=np.int32),
            "term_weights": np.asarray(out_weights, dtype=np.float32),
            "checkpoint_shared_train_gradient": np.asarray(1, dtype=np.int32),
            "source_train_terms": np.asarray(train.shape[0], dtype=np.int32),
            "source_path": np.asarray(str(input_path)),
        }
        for key in ("ckpt_paths", "proj_dim"):
            if key in data:
                value = np.asarray(data[key])
                if key == "ckpt_paths" and value.shape[0] == train.shape[0]:
                    payload[key] = np.asarray([value[np.where(ckpt_indices == ckpt)[0][0]] for ckpt in unique_ckpts])
                else:
                    payload[key] = value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    np.savez_compressed(tmp, **payload)
    tmp.replace(output_path)
    manifest = {
        "mode": "aggregate_traj_train_by_checkpoint",
        "input": str(input_path),
        "output": str(output_path),
        "source_terms": int(train.shape[0]),
        "output_terms": int(len(unique_ckpts)),
        "num_points": int(score_indices.shape[0]),
    }
    with open(output_path.parent / "checkpoint_shared_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[aggregate] wrote checkpoint-shared train artifact: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate TrajTracIn train terms by checkpoint.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    aggregate_train_artifact(args.input, args.output)


if __name__ == "__main__":
    main()
