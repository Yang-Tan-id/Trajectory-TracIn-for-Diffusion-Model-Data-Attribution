#!/usr/bin/env python3
"""Merge fused train shards for endpoint-TracIn AdamW-full 100x1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
for path in (SHAPES_ROOT, REFINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.stage_artifact_runner import _write_score_outputs
from dataset_config import _prompt_tag


VARIANTS = (
    ("score", "raw"),
    ("score_query_normalized", "query_l2_normalized"),
    ("score_train_l2_normalized", "train_l2_normalized"),
    ("score_query_train_l2_normalized", "query_train_l2_normalized"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", required=True)
    parser.add_argument("--component-prefix", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--shard-count", type=int, default=2)
    args = parser.parse_args()

    query_ids = [
        int(value)
        for value in args.query_ids.replace(",", " ").split()
        if value
    ]
    records = json.loads(args.query_file.read_text())["queries"]
    shards = []
    for shard_index in range(args.shard_count):
        path = Path(f"{args.component_prefix}.shard_{shard_index:02d}.npz")
        with np.load(path, allow_pickle=False) as payload:
            shards.append({key: np.asarray(payload[key]) for key in payload.files})
    score_indices = shards[0]["score_indices"]
    artifact_order = [str(value) for value in shards[0]["query_artifacts"]]
    ordered_query_ids = []
    for artifact in artifact_order:
        matches = [
            query_id
            for query_id in query_ids
            if f"prompt_{_prompt_tag(records[query_id]['prompt'])}" in artifact
            and (
                f"seed_{int(records[query_id]['initial_seed']):06d}_query_gradient_"
                in artifact
            )
        ]
        if len(matches) != 1:
            raise ValueError(f"cannot map query artifact to one query id: {artifact}")
        ordered_query_ids.append(matches[0])
    for shard in shards[1:]:
        if not np.array_equal(score_indices, shard["score_indices"]):
            raise ValueError("score index mismatch across checkpoint shards")

    linear = sum(
        (shard["linear"] for shard in shards),
        np.zeros_like(shards[0]["linear"]),
    )
    result_root = SHAPES_ROOT / "result" / args.experiment
    for query_position, query_id in enumerate(ordered_query_ids):
        record = records[query_id]
        prompt = str(record["prompt"])
        seed = int(record["initial_seed"])
        root = (
            result_root
            / "attribution_score"
            / "prompted_solo"
            / f"train_seed_{args.train_seed}"
            / f"query_{_prompt_tag(prompt)}"
            / f"initial_seed_{seed}"
            / f"traj_tracin_{args.namespace}"
        )
        for variant_index, (directory, variant) in enumerate(VARIANTS):
            _write_score_outputs(
                root / directory,
                linear[query_position, variant_index],
                score_indices,
                train_dir=args.component_prefix.parent,
                query_dir=result_root / "sample_ddim_eta0_1000",
                algorithm="traj_tracin",
                extra_manifest={
                    "score_variant": variant,
                    "score_contraction": "linear",
                    "train_feature": (
                        "adamw_full_update_of_mean_100t_mc1_loss_gradient"
                    ),
                    "query_objective": "endpoint_simple_loss_mean100t_mc1",
                    "train_timestamp_count": 100,
                    "query_timestamp_count": 100,
                    "train_mc_per_timestamp": 1,
                    "query_mc_per_timestamp": 1,
                    "learning_rate_semantics": (
                        "AdamW update includes checkpoint LR exactly once; L2 "
                        "normalization removes only its positive scalar"
                    ),
                    "persistent_train_gradient_artifact": False,
                },
            )
        print(f"[merge] Q{query_id} complete", flush=True)


if __name__ == "__main__":
    main()
