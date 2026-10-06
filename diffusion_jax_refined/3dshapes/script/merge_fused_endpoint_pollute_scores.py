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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query-file", type=Path, required=True)
    p.add_argument("--query-ids", required=True)
    p.add_argument("--component-prefix", type=Path, required=True)
    p.add_argument("--experiment", default="experiment1")
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--shard-count", type=int, default=2)
    a = p.parse_args()
    qids = [int(x) for x in a.query_ids.replace(",", " ").split()]
    records = json.loads(a.query_file.read_text())["queries"]
    shards = []
    for i in range(a.shard_count):
        path = Path(f"{a.component_prefix}.shard_{i:02d}.npz")
        with np.load(path, allow_pickle=False) as z:
            shards.append({k: np.asarray(z[k]) for k in z.files})
    indices = shards[0]["score_indices"]
    artifact_order = [str(x) for x in shards[0]["query_artifacts"]]
    ordered_qids = []
    for artifact in artifact_order:
        matches = [
            qid for qid in qids
            if f"prompt_{_prompt_tag(records[qid]['prompt'])}" in artifact
            and f"seed_{int(records[qid]['initial_seed']):06d}_query_gradient_" in artifact
        ]
        if len(matches) != 1:
            raise ValueError(f"cannot map query artifact to one qid: {artifact}")
        ordered_qids.append(matches[0])
    for shard in shards[1:]:
        if not np.array_equal(indices, shard["score_indices"]):
            raise ValueError("score index mismatch")
    linear = sum((s["linear"] for s in shards), np.zeros_like(shards[0]["linear"]))
    termwise = sum((s["termwise_squared"] for s in shards), np.zeros_like(shards[0]["termwise_squared"]))
    timestamp = sum((s["timestamp_components"] for s in shards), np.zeros_like(shards[0]["timestamp_components"]))
    reductions = {
        "linear": linear,
        "termwise_squared": termwise,
        "timestamp_sum_squared": np.square(timestamp).sum(axis=2),
    }
    result_root = SHAPES_ROOT / "result" / a.experiment
    for qi, qid in enumerate(ordered_qids):
        record = records[qid]
        prompt, seed = str(record["prompt"]), int(record["initial_seed"])
        for reduction, scores in reductions.items():
            namespace = f"recreate_adamw_full_polluted_endpoint_delta_l2normalized_{reduction}_aligned100x1_q0_99"
            root = (result_root / "attribution_score" / "prompted_solo" /
                    f"train_seed_{a.train_seed}" / f"query_{_prompt_tag(prompt)}" /
                    f"initial_seed_{seed}" / f"traj_tracin_{namespace}")
            for vi, (directory, variant) in enumerate(VARIANTS):
                _write_score_outputs(
                    root / directory, scores[qi, vi], indices,
                    train_dir=a.component_prefix.parent,
                    query_dir=result_root / "sample_ddim_eta0_1000",
                    algorithm="traj_tracin",
                    extra_manifest={
                        "score_variant": variant,
                        "score_contraction": reduction,
                        "train_feature": "adamw_full_direction_lr_removed",
                        "query_objective": "normalized_polluted_endpoint_next_checkpoint_delta_projection",
                        "timestamp_count": 100,
                        "checkpoint_timestamp_noise_aligned": True,
                        "learning_rate_semantics": "linear=lr*dot; termwise=lr*dot^2; timestamp=(sum_ckpt lr*dot)^2",
                        "persistent_train_gradient_artifact": False,
                    },
                )
        print(f"[merge] Q{qid} complete", flush=True)


if __name__ == "__main__":
    main()
