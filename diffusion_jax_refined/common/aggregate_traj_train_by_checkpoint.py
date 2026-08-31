from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _as_1d(data: np.lib.npyio.NpzFile, key: str, dtype) -> np.ndarray:
    if key not in data:
        raise KeyError(f"missing required key {key}")
    return np.asarray(data[key], dtype=dtype).reshape(-1)


def _load_aggregated_payload(input_path: Path, *, input_index: int, num_inputs: int) -> dict[str, np.ndarray]:
    print(f"[aggregate] loading input {input_index}/{num_inputs}: {input_path}", flush=True)
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
        print(
            f"[aggregate] input {input_index}/{num_inputs} loaded | "
            f"terms={train.shape[0]} points={train.shape[1]} dim={train.shape[2]} ckpts={len(unique_ckpts)}",
            flush=True,
        )
        out_features = []
        out_weights = []
        for ckpt_pos, ckpt in enumerate(unique_ckpts, start=1):
            mask = ckpt_indices == ckpt
            weights = term_weights[mask]
            weight_sum = float(weights.sum())
            if abs(weight_sum) <= 1e-30:
                weights = np.full((int(mask.sum()),), 1.0 / float(mask.sum()), dtype=np.float64)
                weight_sum = 1.0
            normalized = weights / weight_sum
            out_features.append(np.tensordot(normalized, train[mask], axes=(0, 0)).astype(np.float32))
            out_weights.append(weight_sum)
            if ckpt_pos == 1 or ckpt_pos % 10 == 0 or ckpt_pos == len(unique_ckpts):
                print(
                    f"[aggregate] input {input_index}/{num_inputs} checkpoint {ckpt_pos}/{len(unique_ckpts)} "
                    f"| source_terms={int(mask.sum())}",
                    flush=True,
                )

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

    return payload


def aggregate_train_artifact(input_paths: list[Path], output_path: Path) -> None:
    if not input_paths:
        raise ValueError("at least one input artifact is required")
    print(f"[aggregate] start | inputs={len(input_paths)} output={output_path}", flush=True)
    payloads = [
        _load_aggregated_payload(path, input_index=i, num_inputs=len(input_paths))
        for i, path in enumerate(input_paths, start=1)
    ]
    print("[aggregate] validating checkpoint metadata", flush=True)
    ref_ckpts = np.asarray(payloads[0]["ckpt_indices"], dtype=np.int32)
    for path, payload in zip(input_paths[1:], payloads[1:]):
        ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
        if ckpts.shape != ref_ckpts.shape or not np.array_equal(ckpts, ref_ckpts):
            raise ValueError(f"{path}: ckpt_indices do not match first input")

    print("[aggregate] concatenating datapoint shards", flush=True)
    payload = dict(payloads[0])
    payload["train_features"] = np.concatenate([p["train_features"] for p in payloads], axis=1).astype(np.float32)
    payload["score_indices"] = np.concatenate([p["score_indices"] for p in payloads], axis=0).astype(np.int64)
    print(
        f"[aggregate] concatenated | terms={payload['train_features'].shape[0]} "
        f"points={payload['train_features'].shape[1]} dim={payload['train_features'].shape[2]}",
        flush=True,
    )
    print("[aggregate] sorting score indices", flush=True)
    order = np.argsort(payload["score_indices"])
    payload["score_indices"] = payload["score_indices"][order]
    payload["train_features"] = payload["train_features"][:, order, :]
    if len(set(int(x) for x in payload["score_indices"])) != int(payload["score_indices"].shape[0]):
        raise ValueError("input artifacts contain overlapping score_indices")
    payload["source_path"] = np.asarray([str(path) for path in input_paths])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    print(f"[aggregate] writing compressed artifact: {tmp}", flush=True)
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **payload)
    print("[aggregate] replacing final artifact", flush=True)
    tmp.replace(output_path)
    manifest = {
        "mode": "aggregate_traj_train_by_checkpoint",
        "inputs": [str(path) for path in input_paths],
        "output": str(output_path),
        "output_terms": int(payload["train_features"].shape[0]),
        "num_points": int(payload["score_indices"].shape[0]),
    }
    with open(output_path.parent / "checkpoint_shared_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[aggregate] wrote checkpoint-shared train artifact: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate TrajTracIn train terms by checkpoint.")
    parser.add_argument("--input", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    aggregate_train_artifact(args.input, args.output)


if __name__ == "__main__":
    main()
