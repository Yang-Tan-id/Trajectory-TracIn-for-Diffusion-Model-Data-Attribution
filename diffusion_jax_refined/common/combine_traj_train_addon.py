from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _required_1d(payload: dict[str, np.ndarray], key: str, dtype) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"missing required key {key}")
    return np.asarray(payload[key], dtype=dtype).reshape(-1)


def combine_train_artifacts(
    *,
    base_path: Path,
    addon_path: Path,
    output_path: Path,
    renormalize_by_source_count: bool = True,
) -> None:
    print(f"[combine] base : {base_path}", flush=True)
    print(f"[combine] addon: {addon_path}", flush=True)
    base = _load(base_path)
    addon = _load(addon_path)

    base_train = np.asarray(base["train_features"], dtype=np.float32)
    addon_train = np.asarray(addon["train_features"], dtype=np.float32)
    if base_train.ndim != 3 or addon_train.ndim != 3:
        raise ValueError(f"train_features must be rank 3, got {base_train.shape} and {addon_train.shape}")
    if base_train.shape[1:] != addon_train.shape[1:]:
        raise ValueError(f"train feature point/dim mismatch: {base_train.shape} vs {addon_train.shape}")

    base_indices = _required_1d(base, "score_indices", np.int64)
    addon_indices = _required_1d(addon, "score_indices", np.int64)
    if base_indices.shape != addon_indices.shape or not np.array_equal(base_indices, addon_indices):
        raise ValueError("score_indices differ between base and addon artifacts")

    payload: dict[str, np.ndarray] = {
        "train_features": np.concatenate([base_train, addon_train], axis=0).astype(np.float32),
        "score_indices": base_indices.astype(np.int64),
    }
    for key, dtype in (
        ("ckpt_indices", np.int32),
        ("timesteps", np.int32),
        ("snapshot_positions", np.int32),
        ("term_weights", np.float32),
    ):
        if key in base and key in addon:
            values = np.concatenate(
                [np.asarray(base[key], dtype=dtype).reshape(-1), np.asarray(addon[key], dtype=dtype).reshape(-1)],
                axis=0,
            )
            if key == "term_weights" and renormalize_by_source_count:
                values = values / np.asarray(2.0, dtype=np.float32)
            payload[key] = values.astype(dtype)

    order_keys = ("ckpt_indices", "snapshot_positions", "timesteps")
    if all(key in payload for key in order_keys):
        order = np.lexsort(
            (
                np.asarray(payload["timesteps"], dtype=np.int64),
                np.asarray(payload["snapshot_positions"], dtype=np.int64),
                np.asarray(payload["ckpt_indices"], dtype=np.int64),
            )
        )
        payload["train_features"] = payload["train_features"][order]
        for key in ("ckpt_indices", "timesteps", "snapshot_positions", "term_weights"):
            if key in payload:
                payload[key] = payload[key][order]

    for key in ("ckpt_paths", "proj_dim", "query_objective", "query_target_checkpoint"):
        if key in base:
            payload[key] = np.asarray(base[key])
    payload["combined_train_sources"] = np.asarray([str(base_path), str(addon_path)])
    payload["combined_train_source_term_counts"] = np.asarray([base_train.shape[0], addon_train.shape[0]], dtype=np.int32)
    payload["combined_train_mode"] = np.asarray("base_plus_addon_timestep_terms")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(f"{output_path.name}.tmp")
    print(
        f"[combine] writing terms={payload['train_features'].shape[0]} "
        f"points={payload['train_features'].shape[1]} dim={payload['train_features'].shape[2]} -> {output_path}",
        flush=True,
    )
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **payload)
    tmp.replace(output_path)
    with open(output_path.parent / "combined_train_addon_manifest.json", "w") as handle:
        json.dump(
            {
                "mode": "base_plus_addon_timestep_terms",
                "base": str(base_path),
                "addon": str(addon_path),
                "output": str(output_path),
                "renormalize_by_source_count": bool(renormalize_by_source_count),
                "base_terms": int(base_train.shape[0]),
                "addon_terms": int(addon_train.shape[0]),
                "output_terms": int(payload["train_features"].shape[0]),
                "num_points": int(payload["score_indices"].shape[0]),
            },
            handle,
            indent=2,
        )
    print(f"[combine] done: {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine an existing TrajTracIn train artifact with an add-on timestep artifact.")
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--addon", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-renormalize", action="store_true")
    args = parser.parse_args()
    combine_train_artifacts(
        base_path=args.base,
        addon_path=args.addon,
        output_path=args.output,
        renormalize_by_source_count=not args.no_renormalize,
    )


if __name__ == "__main__":
    main()
