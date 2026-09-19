from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TERM_ARRAYS = (
    "train_features",
    "residuals",
    "gram",
    "gram_undamped",
    "ckpt_indices",
    "timesteps",
    "mc_indices",
)
COMMON_ARRAYS = (
    "score_indices",
    "damping",
    "damping_sweep_values",
    "proj_dim",
    "mc_samples_per_term",
    "mc_aggregation",
    "mc_normalize_eps",
)


def load(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def merge(shards: list[Path], output: Path) -> None:
    payloads = [load(path) for path in shards]
    for path, payload in zip(shards, payloads):
        missing = [key for key in (*TERM_ARRAYS, *COMMON_ARRAYS) if key not in payload]
        if missing:
            raise KeyError(f"{path} is missing {missing}")

    common = {}
    for key in COMMON_ARRAYS:
        first = payloads[0][key]
        if any(first.shape != item[key].shape or not np.array_equal(first, item[key]) for item in payloads[1:]):
            raise ValueError(f"term shards disagree on {key}")
        common[key] = first

    combined = {key: np.concatenate([item[key] for item in payloads], axis=0) for key in TERM_ARRAYS}
    order = np.lexsort(
        (
            np.asarray(combined["mc_indices"], dtype=np.int64),
            np.asarray(combined["timesteps"], dtype=np.int64),
            np.asarray(combined["ckpt_indices"], dtype=np.int64),
        )
    )
    for key in TERM_ARRAYS:
        combined[key] = combined[key][order]

    term_ids = np.stack(
        [combined["ckpt_indices"], combined["timesteps"], combined["mc_indices"]],
        axis=1,
    )
    if len(np.unique(term_ids, axis=0)) != len(term_ids):
        raise ValueError("DAS term shards overlap")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **combined, **common, shard_paths=np.asarray([str(x) for x in shards]))
    manifest = {
        "output": str(output),
        "shards": [str(path) for path in shards],
        "terms": int(len(term_ids)),
        "points": int(combined["train_features"].shape[1]),
        "projection_dim": int(combined["train_features"].shape[2]),
        "timesteps": [int(value) for value in combined["timesteps"]],
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[saved] merged DAS term shards: {output}", flush=True)
    print(f"[shape] train_features={combined['train_features'].shape}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge disjoint DAS timestamp/term shards")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    merge(args.shards, args.output)


if __name__ == "__main__":
    main()
