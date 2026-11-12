"""Retrain one ArtBench LoRA after DAS Top-K removal."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import ArtBenchConfig


METHOD = "das_lambda_1000"


def parse_args() -> argparse.Namespace:
    config = ArtBenchConfig()
    project = config.index_root.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-index", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument(
        "--score-path",
        type=Path,
        default=(
            project
            / "outputs/attribution/artbench_das_50x1_full5000_dual32768_sm"
            / "scores/lambda_1000/scores.npz"
        ),
    )
    parser.add_argument(
        "--base-index-path",
        type=Path,
        default=config.index_root / "idx-train.pkl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project / "outputs/topk-removal-das-lambda1000-q0-59",
    )
    parser.add_argument("--model", default="lambdalabs/miniSD-diffusers")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", default="fp16")
    parser.add_argument("--num-checkpoints", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_query_scores(
    path: Path, query_index: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        scores = np.asarray(payload["scores"], dtype=np.float64)
        query_ids = np.asarray(payload["query_ids"], dtype=np.int64)
        indices = np.asarray(payload["train_metadata_indices"], dtype=np.int64)
        manifest = (
            json.loads(str(payload["manifest_json"].item()))
            if "manifest_json" in payload.files
            else {}
        )
    if scores.shape != (len(query_ids), len(indices)):
        raise ValueError(f"inconsistent DAS artifact shape: {scores.shape}")
    if len(np.unique(query_ids)) != len(query_ids):
        raise ValueError("DAS artifact contains duplicate query IDs")
    positions = {int(value): slot for slot, value in enumerate(query_ids)}
    if query_index not in positions:
        raise ValueError(f"DAS artifact does not contain Q{query_index}")
    if float(manifest.get("damping", 1000.0)) != 1000.0:
        raise ValueError(f"expected lambda=1000, found {manifest.get('damping')}")
    return scores[positions[query_index]], indices, manifest


def select_topk(scores: np.ndarray, indices: np.ndarray, top_k: int) -> np.ndarray:
    if not 0 < top_k < len(scores):
        raise ValueError(f"top-k must be in [1, {len(scores) - 1}]")
    ranking = np.where(np.isfinite(scores), scores, -np.inf)
    order = np.argsort(-ranking, kind="stable")[:top_k]
    if not np.all(np.isfinite(ranking[order])):
        raise ValueError(f"fewer than {top_k} finite DAS scores")
    return order


def main() -> None:
    args = parse_args()
    if not 0 <= args.query_index < 60:
        raise ValueError("this experiment is Q0-Q59; query-index must be in [0, 59]")
    scores, candidate_indices, score_manifest = load_query_scores(
        args.score_path, args.query_index
    )
    if len(np.unique(candidate_indices)) != len(candidate_indices):
        raise ValueError("DAS candidate indices contain duplicates")
    order = select_topk(scores, candidate_indices, args.top_k)
    removed = candidate_indices[order]

    with args.base_index_path.open("rb") as handle:
        full_indices = [int(value) for value in pickle.load(handle)]
    if len(full_indices) != len(set(full_indices)):
        raise ValueError("base training indices contain duplicates")
    if set(map(int, candidate_indices)) != set(full_indices):
        raise ValueError("DAS candidates do not exactly match the 5,000-point split")

    run_dir = (
        args.output_root
        / METHOD
        / f"query-{args.query_index:03d}"
        / f"topk-{args.top_k:04d}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    retained_path = run_dir / "retained-indices.pkl"
    manifest_path = run_dir / "removal-manifest.json"
    completion_path = run_dir / "TRAINING_COMPLETE"
    removed_set = set(map(int, removed))
    retained = [index for index in full_indices if index not in removed_set]
    if len(retained) != len(full_indices) - args.top_k:
        raise AssertionError("unexpected retained-set size")
    temporary = retained_path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(retained, handle)
    temporary.replace(retained_path)

    manifest: dict[str, Any] = {
        "method": METHOD,
        "algorithm": "das",
        "lambda": 1000.0,
        "query_index": args.query_index,
        "top_k": args.top_k,
        "ranking_direction": "descending",
        "score_path": str(args.score_path.resolve()),
        "score_manifest": score_manifest,
        "base_index_path": str(args.base_index_path.resolve()),
        "full_training_size": len(full_indices),
        "retained_training_size": len(retained),
        "removed_metadata_indices": [int(value) for value in removed],
        "removed_scores": [float(value) for value in scores[order]],
        "training": {
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "mixed_precision": args.mixed_precision,
            "num_checkpoints": args.num_checkpoints,
        },
    }
    atomic_json(manifest_path, manifest)
    print(
        f"[prepared] Q{args.query_index} DAS lambda=1000 "
        f"remove={len(removed)} retain={len(retained)}",
        flush=True,
    )
    if args.prepare_only:
        return

    model_dir = run_dir / "model"
    final_dir = model_dir / "final"
    if completion_path.is_file() and final_dir.is_dir() and any(final_dir.iterdir()):
        print(f"[skip] Q{args.query_index} already trained: {final_dir}", flush=True)
        return
    command = [
        sys.executable,
        "-m",
        "sd_artbench.train_lora",
        "--model",
        args.model,
        "--index-path",
        str(retained_path),
        "--output-dir",
        str(model_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--mixed-precision",
        args.mixed_precision,
        "--num-checkpoints",
        str(args.num_checkpoints),
    ]
    print("[train] " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=os.environ.copy())
    if not final_dir.is_dir() or not any(final_dir.iterdir()):
        raise FileNotFoundError(f"training did not create {final_dir}")
    manifest["status"] = "trained"
    atomic_json(manifest_path, manifest)
    completion_path.write_text("complete\n", encoding="utf-8")
    print(f"[done] Q{args.query_index} {final_dir}", flush=True)


if __name__ == "__main__":
    main()
