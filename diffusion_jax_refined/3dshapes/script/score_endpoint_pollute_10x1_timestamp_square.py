#!/usr/bin/env python3
"""Score each timestamp of cached AdamW-full 10x1 endpoint-pollute terms.

For each datapoint/query/timestamp this computes

    (sum_checkpoint term_weight * dot(train_adamw_full, query_delta)) ** 2

and writes raw, query-L2, train-L2, and both-L2 score vectors.  Checkpoint
parts are streamed one at a time; the expensive train gradients are never
recomputed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SHAPES_ROOT.parents[1]
LEGACY_ROOT = SHAPES_ROOT.parent / "legacy_jax"
for root in (SHAPES_ROOT, REPO_ROOT, LEGACY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import jax
import jax.numpy as jnp

from dataset_config import _prompt_tag
from diffusion_jax_refined.common.stage_artifact_runner import _write_score_outputs


VARIANT_DIRS = {
    "raw": "score",
    "query_l2": "score_query_normalized",
    "train_l2": "score_train_l2_normalized",
    "query_train_l2": "score_query_train_l2_normalized",
}
TIMESTAMPS = (0, 111, 222, 333, 444, 555, 666, 777, 888, 999)


def parse_ids(text: str) -> list[int]:
    return [int(token) for token in text.replace(",", " ").split()]


def query_path(
    sample_root: Path,
    checkpoint: Path,
    prompt: str,
    seed: int,
    namespace: str,
) -> Path:
    return (
        sample_root
        / "cifar"
        / f"prompt_{_prompt_tag(prompt)}"
        / f"model_prompted_solo__ckpt_{checkpoint.stem}"
        / f"seed_{seed:06d}_query_gradient_{namespace}"
        / "traj_tracin"
        / "query_gradient_artifact.npz"
    )


def score_root(
    result_root: Path,
    train_seed: int,
    prompt: str,
    seed: int,
    timestep: int,
) -> Path:
    namespace = (
        "traj_tracin_recreate_adamw_full_polluted_endpoint_"
        "delta_l2normalized_timestamp_aware_square_"
        f"t{timestep:03d}_q0_99"
    )
    return (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
        / f"query_{_prompt_tag(prompt)}"
        / f"initial_seed_{seed}"
        / namespace
    )


def query_outputs_complete(
    result_root: Path,
    train_seed: int,
    record: dict[str, object],
) -> bool:
    prompt = str(record["prompt"])
    seed = int(record.get("initial_seed", record.get("seed")))
    for timestep in TIMESTAMPS:
        root = score_root(result_root, train_seed, prompt, seed, timestep)
        for directory in VARIANT_DIRS.values():
            target = root / directory
            if not all(
                (target / filename).is_file()
                for filename in (
                    "scores.npy",
                    "score_indices.npy",
                    "score_artifact_manifest.json",
                )
            ):
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()

    all_ids = parse_ids(args.query_ids)
    query_ids = all_ids[args.shard_index :: args.shard_count]
    if not query_ids:
        print(f"[done] shard {args.shard_index}/{args.shard_count}: no queries")
        return

    records = json.loads(args.query_file.read_text())["queries"]
    result_root = SHAPES_ROOT / "result" / args.experiment
    pending_ids = [
        query_id
        for query_id in query_ids
        if not query_outputs_complete(
            result_root, args.train_seed, records[query_id]
        )
    ]
    skipped = len(query_ids) - len(pending_ids)
    query_ids = pending_ids
    if skipped:
        print(f"[resume] skipped {skipped} complete queries", flush=True)
    if not query_ids:
        print(
            f"[done] shard {args.shard_index}/{args.shard_count}: "
            "all outputs already complete",
            flush=True,
        )
        return
    checkpoint = (
        result_root
        / "model"
        / "prompted_jax"
        / f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt"
    )
    train_artifact = (
        result_root
        / "model"
        / "prompted_solo"
        / f"seed_{args.train_seed}_train_gradient"
        / "traj_tracin_recreate_adamw_dual_mc1_aligned10x1"
        / "train_datapoint_gradient_artifact.npz"
    )
    part_root = Path(str(train_artifact) + ".parts")
    parts = sorted(part_root.glob("ckpt_*.npz"))
    if len(parts) != 50:
        raise RuntimeError(f"expected 50 train checkpoint parts under {part_root}, found {len(parts)}")

    query_namespace = "recreate_q0_99_polluted_endpoint_next_delta_l2normalized_10t"
    sample_root = result_root / "sample_ddim_eta0_1000"
    query_arrays = []
    query_metadata = None
    query_paths = []
    selected_records = []
    for query_id in query_ids:
        record = records[query_id]
        prompt = str(record["prompt"])
        seed = int(record.get("initial_seed", record.get("seed")))
        path = query_path(sample_root, checkpoint, prompt, seed, query_namespace)
        with np.load(path, allow_pickle=False) as payload:
            features = np.asarray(payload["query_features"], dtype=np.float32)
            metadata = np.stack(
                (
                    np.asarray(payload["ckpt_indices"], dtype=np.int32),
                    np.asarray(payload["timesteps"], dtype=np.int32),
                ),
                axis=1,
            )
        if query_metadata is None:
            query_metadata = metadata
        elif not np.array_equal(query_metadata, metadata):
            raise ValueError(f"query term metadata mismatch: {path}")
        query_arrays.append(features)
        query_paths.append(path)
        selected_records.append(record)
        print(f"[load] Q{query_id}: {path}", flush=True)

    assert query_metadata is not None
    queries = np.stack(query_arrays, axis=0)
    lookup = {
        (int(ckpt), int(timestep)): row
        for row, (ckpt, timestep) in enumerate(query_metadata)
    }
    timestamps = tuple(sorted({int(value) for value in query_metadata[:, 1]}))
    if timestamps != TIMESTAMPS:
        raise ValueError(
            f"unexpected timestamp grid {timestamps}; expected {TIMESTAMPS}"
        )
    timestamp_slot = {value: slot for slot, value in enumerate(timestamps)}
    num_queries = len(query_ids)

    with np.load(parts[0], allow_pickle=False) as first:
        score_indices = np.asarray(first["score_indices"], dtype=np.int64)
        first_features = np.asarray(first["train_features"])
        proj_dim = int(first_features.shape[-1])
        num_points = int(first_features.shape[1])
    if queries.shape[-1] != proj_dim:
        raise ValueError(f"projection mismatch: train={proj_dim}, query={queries.shape[-1]}")

    device = jax.devices()[0]
    if device.platform not in ("cuda", "gpu"):
        raise RuntimeError(f"expected a GPU device, got {device}")
    accumulators = {
        variant: [jnp.zeros((num_queries, num_points), dtype=jnp.float32) for _ in timestamps]
        for variant in VARIANT_DIRS
    }

    @jax.jit
    def four_dots(train_term, history_term, query_term):
        effective = train_term + history_term[None, :]
        query_norm = query_term / jnp.maximum(
            jnp.linalg.norm(query_term, axis=1, keepdims=True), args.eps
        )
        train_norm = effective / jnp.maximum(
            jnp.linalg.norm(effective, axis=1, keepdims=True), args.eps
        )
        return (
            query_term @ effective.T,
            query_norm @ effective.T,
            query_term @ train_norm.T,
            query_norm @ train_norm.T,
        )

    variant_names = tuple(VARIANT_DIRS)
    used_parts = 0
    for part_no, part in enumerate(parts, start=1):
        with np.load(part, allow_pickle=False) as payload:
            features = np.asarray(payload["train_features"], dtype=np.float32)
            history = np.asarray(payload["optimizer_history_features"], dtype=np.float32)
            ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            part_timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            weights = np.asarray(payload["term_weights"], dtype=np.float32)
            indices = np.asarray(payload["score_indices"], dtype=np.int64)
        if not np.array_equal(indices, score_indices):
            raise ValueError(f"score-index mismatch: {part}")
        matched = 0
        for row, (ckpt, timestep, weight) in enumerate(zip(ckpts, part_timesteps, weights)):
            key = (int(ckpt), int(timestep))
            query_row = lookup.get(key)
            if query_row is None:
                continue
            matched += 1
            slot = timestamp_slot[int(timestep)]
            outputs = four_dots(
                jax.device_put(features[row], device),
                jax.device_put(history[row], device),
                jax.device_put(queries[:, query_row, :], device),
            )
            for variant, contribution in zip(variant_names, outputs):
                accumulators[variant][slot] = (
                    accumulators[variant][slot] + float(weight) * contribution
                )
        used_parts += int(matched > 0)
        if matched:
            # Keep execution checkpoint-major instead of queueing all 49 parts;
            # this bounds device/host buffers and makes progress truthful.
            jax.block_until_ready(accumulators[variant_names[-1]][slot])
        print(f"[score] checkpoint part {part_no}/50 matched_terms={matched}", flush=True)

    if used_parts != 49:
        raise RuntimeError(f"expected 49 next-checkpoint-aligned parts, used {used_parts}")

    host_scores = {
        variant: [
            np.square(np.asarray(jax.device_get(value), dtype=np.float64))
            for value in by_timestamp
        ]
        for variant, by_timestamp in accumulators.items()
    }
    for query_slot, (query_id, record, source_path) in enumerate(
        zip(query_ids, selected_records, query_paths)
    ):
        prompt = str(record["prompt"])
        seed = int(record.get("initial_seed", record.get("seed")))
        for slot, timestep in enumerate(timestamps):
            root = score_root(result_root, args.train_seed, prompt, seed, timestep)
            for variant, directory in VARIANT_DIRS.items():
                target = root / directory
                _write_score_outputs(
                    target,
                    host_scores[variant][slot][query_slot],
                    score_indices,
                    train_dir=train_artifact.parent,
                    query_dir=source_path.parent,
                    algorithm="traj_tracin",
                    extra_manifest={
                        "formula": "square(sum_checkpoint(term_weight * adamw_full_dot))",
                        "timestamp_aware_square": True,
                        "timestep": int(timestep),
                        "score_variant": variant,
                        "query_id": int(query_id),
                        "train_artifact": str(train_artifact),
                    },
                )
        print(f"[write] Q{query_id} complete", flush=True)
    print(
        f"[done] shard={args.shard_index}/{args.shard_count} "
        f"queries={len(query_ids)} timestamps={timestamps}",
        flush=True,
    )


if __name__ == "__main__":
    main()
