#!/usr/bin/env python3
"""Evaluate only the ten five-method CelebA configs on 50x1 simple loss."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np


CELEBA_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = CELEBA_ROOT.parent
REPO_ROOT = REFINE_ROOT.parent
for path in (CELEBA_ROOT, REFINE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_config_attr40 import _prompt_path_tag
from diffusion_jax_refined.script.export_pint_lds_txt import method_configs
from diffusion_jax_refined.celeba.script.run_endpoint_cached_lds import (
    MODEL_GROUP,
    build_membership,
    load_targets,
    spearman,
)


VARIANT_SCORE_DIR = {
    "raw": "score",
    "query_train_l2": "score_query_train_l2_normalized",
}


def parse_ids(text: str) -> list[int]:
    values: list[int] = []
    for token in text.replace(",", " ").split():
        if "-" in token:
            left, right = token.split("-", 1)
            values.extend(range(int(left), int(right) + 1))
        else:
            values.append(int(token))
    if not values or len(values) != len(set(values)):
        raise ValueError("query IDs must be nonempty and unique")
    return values


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument(
        "--query-file",
        type=Path,
        default=CELEBA_ROOT / "queries_attr40_val25_x4.json",
    )
    parser.add_argument("--query-ids", default="0-99")
    parser.add_argument("--true-f-csv", type=Path, required=True)
    parser.add_argument("--expected-models", type=int, default=192)
    args = parser.parse_args()

    records = json.loads(args.query_file.read_text())["queries"]
    records_by_id = {
        int(record.get("query_id", fallback)): record
        for fallback, record in enumerate(records)
    }
    query_ids = parse_ids(args.query_ids)
    absent = sorted(set(query_ids) - set(records_by_id))
    if absent:
        raise ValueError(f"query IDs absent from manifest: {absent}")

    methods = method_configs("celeba")
    result_root = CELEBA_ROOT / "result" / args.experiment
    eval_root = result_root / "eval" / "prompted_solo"
    score_root = (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{args.train_seed}"
    )
    targets_by_key, subset_dirs = load_targets(args.true_f_csv)
    lds_model_root = result_root / "lds_model"
    membership_cache: dict[bytes, np.ndarray] = {}

    completed = 0
    for query_id in query_ids:
        record = records_by_id[query_id]
        prompt_tag = _prompt_path_tag(str(record["prompt"]))
        seed = int(record.get("initial_seed", record.get("seed")))
        target_rows = targets_by_key.get((query_id, "simple_loss_50x1"))
        if target_rows is None or len(target_rows) != args.expected_models:
            found = 0 if target_rows is None else len(target_rows)
            raise ValueError(
                f"Q{query_id}: simple_loss_50x1 true-f has "
                f"{found}/{args.expected_models} models"
            )
        true_f = np.asarray(
            [float(row["true_f"]) for row in target_rows], dtype=np.float64
        )
        query_scores = score_root / f"query_{prompt_tag}" / f"initial_seed_{seed}"
        query_eval = eval_root / f"query_{prompt_tag}" / f"initial_seed_{seed}"

        for method in methods:
            for transform, namespace, preferred_sign in (
                ("raw_gradient", method.raw_namespace, method.raw_sign),
                ("adamw_full", method.adamw_namespace, method.adamw_sign),
            ):
                score_dir = query_scores / namespace / VARIANT_SCORE_DIR[method.variant]
                score_path = score_dir / "scores.npy"
                index_path = score_dir / "score_indices.npy"
                if not score_path.is_file() or not index_path.is_file():
                    raise FileNotFoundError(score_dir)
                scores = np.asarray(np.load(score_path), dtype=np.float64).reshape(-1)
                indices = np.asarray(np.load(index_path), dtype=np.int64).reshape(-1)
                if scores.shape != indices.shape:
                    raise ValueError(f"score/index shape mismatch: {score_dir}")
                membership_key = indices.tobytes()
                if membership_key not in membership_cache:
                    membership_cache[membership_key] = build_membership(
                        indices, subset_dirs, lds_model_root
                    )
                sign = 1.0 if preferred_sign == "p1" else -1.0
                prediction = sign * (membership_cache[membership_key] @ scores)
                lds = spearman(prediction, true_f)
                out_dir = (
                    query_eval
                    / "lds"
                    / f"{namespace}_{method.variant}"
                    / "simple_loss_50x1"
                    / f"pred_kept_sign_{preferred_sign}"
                    / MODEL_GROUP
                )
                atomic_json(
                    out_dir / "lds_summary.json",
                    {
                        "algorithm": f"{namespace}_{method.variant}",
                        "method_label": method.label,
                        "transform": transform,
                        "target_function": "simple_loss_50x1",
                        "target_definition": (
                            "mean denoising loss on the original 50 uniformly "
                            "spaced, query-specific shared forward-noise probes"
                        ),
                        "target_cache": str(args.true_f_csv.resolve()),
                        "score_source": str(score_dir.resolve()),
                        "num_models": len(target_rows),
                        "lds_spearman": float(lds),
                        "lds_percent": float(100.0 * lds),
                        "prediction_subset": "kept",
                        "prediction_sign": sign,
                    },
                )
                completed += 1
                print(
                    f"[LDS {completed}/{len(query_ids) * 10}] Q{query_id} "
                    f"{transform} {method.label}={100.0 * lds:+.3f}%",
                    flush=True,
                )

    print(f"[done] selected five-method LDS records={completed}", flush=True)


if __name__ == "__main__":
    main()
