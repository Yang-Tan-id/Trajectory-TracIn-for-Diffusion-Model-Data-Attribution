"""Prepare and train one AdamW Top-K removal counterfactual LoRA."""

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


METHODS = (
    "endpoint_pollute_query_train_l2",
    "endpoint_tracin_raw",
    "endpoint_tracin_query_train_l2",
    "retrac_query_train_l2",
)


def load_score_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        return {
            "scores": np.asarray(payload["scores"], dtype=np.float64),
            "indices": np.asarray(
                payload["train_metadata_indices"], dtype=np.int64
            ),
            "transforms": [str(value) for value in payload["transform_names"]],
            "variants": (
                [str(value) for value in payload["variant_names"]]
                if "variant_names" in payload.files
                else None
            ),
        }


def method_scores(
    *,
    method: str,
    query_index: int,
    endpoint_root: Path,
    retrac_path: Path,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Return scores, metadata indices, ranking sign, and score definition."""

    if method.startswith("endpoint_pollute"):
        payload = load_score_file(
            endpoint_root / "endpoint_pollute_timestamp_sum_squared.npz"
        )
        variant = "query_train_l2"
        ranking_sign = 1
        definition = (
            "AdamW-full endpoint-pollute timestamp-sum-squared "
            "query/train-L2 score descending"
        )
    elif method.startswith("endpoint_tracin"):
        payload = load_score_file(endpoint_root / "endpoint_tracin.npz")
        variant = "raw" if method.endswith("_raw") else "query_train_l2"
        ranking_sign = -1
        definition = (
            f"AdamW-full endpoint-TracIn {variant} linear score ascending"
        )
    elif method == "retrac_query_train_l2":
        payload = load_score_file(retrac_path)
        variant = None
        ranking_sign = -1
        definition = "AdamW-full ReTrac query/train-L2 linear score ascending"
    else:
        raise ValueError(f"unknown method {method!r}")

    try:
        transform_slot = payload["transforms"].index("adamw_full")
    except ValueError as error:
        raise ValueError("score artifact does not contain adamw_full") from error

    if variant is None:
        bank = payload["scores"][transform_slot]
    else:
        variants = payload["variants"]
        if variants is None:
            raise ValueError(f"{method} score artifact has no variants")
        try:
            variant_slot = variants.index(variant)
        except ValueError as error:
            raise ValueError(
                f"score artifact does not contain variant {variant!r}"
            ) from error
        bank = payload["scores"][transform_slot, variant_slot]

    if bank.ndim != 2:
        raise ValueError(f"expected [query, train] scores, found {bank.shape}")
    if query_index < 0 or query_index >= bank.shape[0]:
        raise IndexError(
            f"query index {query_index} outside [0, {bank.shape[0]})"
        )
    scores = np.asarray(bank[query_index], dtype=np.float64)
    if scores.shape != payload["indices"].shape:
        raise ValueError("score and training-index shapes differ")
    return scores, payload["indices"], ranking_sign, definition


def select_topk(
    scores: np.ndarray,
    indices: np.ndarray,
    *,
    top_k: int,
    ranking_sign: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if top_k < 1 or top_k >= len(scores):
        raise ValueError(f"top_k must be in [1, {len(scores) - 1}]")
    if ranking_sign not in (-1, 1):
        raise ValueError("ranking_sign must be -1 or +1")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("candidate metadata indices contain duplicates")
    ranking = ranking_sign * scores
    ranking = np.where(np.isfinite(ranking), ranking, -np.inf)
    # Five thousand candidates are small enough for a full stable sort.  This
    # also makes a tie at the Top-K boundary deterministic by split order.
    order = np.argsort(-ranking, kind="stable")[:top_k]
    if np.any(~np.isfinite(ranking[order])):
        raise ValueError(f"fewer than {top_k} finite scores")
    return indices[order], scores[order], ranking[order]


def parse_args() -> argparse.Namespace:
    defaults = ArtBenchConfig()
    root = defaults.index_root.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--query-index", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument(
        "--stage", choices=("prepare", "train", "all"), default="all"
    )
    parser.add_argument(
        "--endpoint-root",
        type=Path,
        default=root / "outputs/attribution/endpoint-q60-t50-full-linear-hooks",
    )
    parser.add_argument(
        "--retrac-path",
        type=Path,
        default=(
            root
            / "outputs/attribution/retrac-q60-t50-full-linear-hooks/retrac.npz"
        ),
    )
    parser.add_argument(
        "--base-index-path",
        type=Path,
        default=defaults.index_root / "idx-train.pkl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "outputs/topk-removal-adamw-q0-9",
    )
    parser.add_argument("--model", default="lambdalabs/miniSD-diffusers")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", default="fp16")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    run_dir = (
        args.output_root
        / args.method
        / f"query-{args.query_index:03d}"
        / f"topk-{args.top_k:04d}"
    )
    model_dir = run_dir / "model"
    retained_path = run_dir / "retained-indices.pkl"
    manifest_path = run_dir / "removal-manifest.json"
    completion_path = run_dir / "TRAINING_COMPLETE"
    run_dir.mkdir(parents=True, exist_ok=True)

    scores, candidate_indices, ranking_sign, definition = method_scores(
        method=args.method,
        query_index=args.query_index,
        endpoint_root=args.endpoint_root,
        retrac_path=args.retrac_path,
    )
    removed, removed_scores, ranking_scores = select_topk(
        scores,
        candidate_indices,
        top_k=args.top_k,
        ranking_sign=ranking_sign,
    )

    with args.base_index_path.open("rb") as handle:
        full_indices = [int(value) for value in pickle.load(handle)]
    if len(full_indices) != len(set(full_indices)):
        raise ValueError("base training index file contains duplicates")
    if set(candidate_indices.tolist()) != set(full_indices):
        raise ValueError(
            "attribution candidates do not exactly match the base training split"
        )
    removed_set = set(int(value) for value in removed)
    retained = [value for value in full_indices if value not in removed_set]
    if len(retained) != len(full_indices) - args.top_k:
        raise AssertionError("retained split has an unexpected size")

    temporary_index = retained_path.with_suffix(".pkl.tmp")
    with temporary_index.open("wb") as handle:
        pickle.dump(retained, handle)
    temporary_index.replace(retained_path)

    manifest = {
        "method": args.method,
        "transform": "adamw_full",
        "query_index": args.query_index,
        "top_k": args.top_k,
        "ranking_sign": ranking_sign,
        "ranking_direction": "descending" if ranking_sign == 1 else "ascending",
        "score_definition": definition,
        "full_training_size": len(full_indices),
        "retained_training_size": len(retained),
        "removal_fraction": args.top_k / len(full_indices),
        "base_index_path": str(args.base_index_path.resolve()),
        "retained_index_path": str(retained_path.resolve()),
        "removed_metadata_indices": [int(value) for value in removed],
        "removed_scores": [float(value) for value in removed_scores],
        "removed_ranking_scores": [float(value) for value in ranking_scores],
        "training": {
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "mixed_precision": args.mixed_precision,
            "num_checkpoints": 0,
        },
    }
    write_json(manifest_path, manifest)
    print(
        f"[prepared] method={args.method} query={args.query_index} "
        f"remove={len(removed)} retain={len(retained)} direction="
        f"{manifest['ranking_direction']}",
        flush=True,
    )
    if args.stage == "prepare":
        return

    final_dir = model_dir / "final"
    if completion_path.is_file() and final_dir.is_dir() and any(final_dir.iterdir()):
        manifest["status"] = "trained"
        write_json(manifest_path, manifest)
        print(f"[skip] final LoRA already exists: {final_dir}", flush=True)
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
        "0",
    ]
    print("[train] " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=os.environ.copy())
    if not final_dir.is_dir() or not any(final_dir.iterdir()):
        raise FileNotFoundError(f"training did not create {final_dir}")
    manifest["status"] = "trained"
    write_json(manifest_path, manifest)
    completion_path.write_text("complete\n", encoding="utf-8")
    print(f"[done] {final_dir}", flush=True)


if __name__ == "__main__":
    main()
