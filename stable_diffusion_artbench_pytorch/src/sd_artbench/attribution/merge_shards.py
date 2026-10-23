"""Merge non-overlapping attribution shards along the training-example axis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_shard(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "path": path,
            "scores": np.asarray(payload["scores"]),
            "indices": np.asarray(payload["train_metadata_indices"], dtype=np.int64),
            "transforms": np.asarray(payload["transform_names"]),
            "variants": (
                np.asarray(payload["variant_names"])
                if "variant_names" in payload.files
                else None
            ),
            "manifest": json.loads(str(np.asarray(payload["manifest_json"]).item())),
        }


def merge_shards(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray | None]:
    if not paths:
        raise ValueError("at least one input shard is required")
    shards = [load_shard(path) for path in paths]
    shards.sort(key=lambda item: int(item["manifest"]["train_positions"][0]))
    reference = shards[0]
    intervals = []
    previous_end = None
    for shard in shards:
        manifest = shard["manifest"]
        start, end = (int(value) for value in manifest["train_positions"])
        if end <= start:
            raise ValueError(f"{shard['path']}: invalid train interval [{start}, {end})")
        if previous_end is not None and start != previous_end:
            raise ValueError(
                f"training shards must be contiguous and non-overlapping: "
                f"previous end={previous_end}, next start={start}"
            )
        previous_end = end
        intervals.append([start, end])
        if shard["scores"].shape[:-1] != reference["scores"].shape[:-1]:
            raise ValueError(f"{shard['path']}: incompatible score shape")
        if shard["scores"].shape[-1] != end - start:
            raise ValueError(f"{shard['path']}: score width does not match manifest")
        if not np.array_equal(shard["transforms"], reference["transforms"]):
            raise ValueError(f"{shard['path']}: transform names differ")
        reference_variants = reference["variants"]
        if (shard["variants"] is None) != (reference_variants is None):
            raise ValueError(f"{shard['path']}: variant schema differs")
        if reference_variants is not None and not np.array_equal(
            shard["variants"], reference_variants
        ):
            raise ValueError(f"{shard['path']}: variant names differ")
        for key in (
            "method",
            "parameter_source",
            "model",
            "rank",
            "learning_rate",
            "weight_decay",
            "epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "conditioning_dropout",
            "attention_dropout",
            "checkpoint_indices",
            "timesteps",
            "proj_dim",
            "query_ids",
            "query_manifest",
            "max_grad_norm",
            "reduction",
            "replay_mode",
            "debug_max_replay_batches",
            "completed_through_checkpoint_slot",
        ):
            if manifest.get(key) != reference["manifest"].get(key):
                raise ValueError(f"{shard['path']}: manifest field {key!r} differs")

    scores = np.concatenate([shard["scores"] for shard in shards], axis=-1)
    indices = np.concatenate([shard["indices"] for shard in shards])
    if any(len(shard["indices"]) != shard["scores"].shape[-1] for shard in shards):
        raise ValueError("a shard's metadata-index count does not match its score width")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("merged shards contain duplicate training metadata indices")
    manifest = dict(reference["manifest"])
    manifest["train_positions"] = [intervals[0][0], intervals[-1][1]]
    manifest["merged_shards"] = [str(shard["path"]) for shard in shards]
    return scores, indices, manifest, reference["transforms"], reference["variants"]


def main() -> None:
    args = parse_args()
    scores, indices, manifest, transforms, variants = merge_shards(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    fields = {
        "scores": scores,
        "train_metadata_indices": indices,
        "transform_names": transforms,
        "manifest_json": np.asarray(json.dumps(manifest, sort_keys=True)),
    }
    if variants is not None:
        fields["variant_names"] = variants
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **fields)
    temporary.replace(args.output)
    print(f"merged {len(args.inputs)} shards -> {args.output} shape={scores.shape}")


if __name__ == "__main__":
    main()
