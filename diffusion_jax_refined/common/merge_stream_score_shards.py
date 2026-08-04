from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SCORE_KEYS = (
    "scores_raw",
    "scores_query_l2_normalized",
    "scores_train_l2_normalized",
    "scores_query_train_l2_normalized",
)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge query-cached stream score shards.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()

    payloads = [load_npz(path) for path in args.shards]
    first = payloads[0]
    query_artifacts = first["query_artifacts"]
    proj_dims = np.asarray(first.get("proj_dims", first.get("proj_dim")), dtype=np.int32).reshape(-1)
    for path, payload in zip(args.shards[1:], payloads[1:]):
        if not np.array_equal(query_artifacts, payload["query_artifacts"]):
            raise ValueError(f"query_artifacts mismatch in {path}")
        payload_dims = np.asarray(payload.get("proj_dims", payload.get("proj_dim")), dtype=np.int32).reshape(-1)
        if not np.array_equal(payload_dims, proj_dims):
            raise ValueError(f"proj_dims mismatch in {path}")

    order = np.argsort(np.concatenate([payload["score_indices"] for payload in payloads]))
    merged_indices = np.concatenate([payload["score_indices"] for payload in payloads])[order]

    arrays: dict[str, np.ndarray] = {
        "score_indices": merged_indices.astype(np.int64),
        "query_artifacts": query_artifacts,
        "proj_dims": proj_dims.astype(np.int32),
    }
    for key in SCORE_KEYS:
        parts = [payload[key] for payload in payloads]
        axis = 2 if parts[0].ndim == 3 else 1
        merged = np.concatenate(parts, axis=axis)
        arrays[key] = (merged[:, :, order] if axis == 2 else merged[:, order]).astype(np.float64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(args.output)
    print(f"[merge] wrote {args.output}")


if __name__ == "__main__":
    main()
