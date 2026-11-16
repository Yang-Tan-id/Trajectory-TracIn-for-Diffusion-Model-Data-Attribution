#!/usr/bin/env python3
"""Run CelebA removal with CLIP image-similarity attribution scores."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import run_endpoint_top500_removal as worker


METHOD = "clip_similarity_full10k_pos"
_base_score_spec = worker.score_spec


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def prepare_clip_scores(*, result_root: Path, train_seed: int, record: dict) -> Path:
    cache_path = Path(
        os.environ.get(
            "CLIP_FEATURE_CACHE",
            str(
                result_root
                / "analysis"
                / "clip_similarity_score_lds_q0_q99"
                / "clip_features.npz"
            ),
        )
    ).expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"missing CLIP feature cache: {cache_path}; run "
            "evaluate_clip_similarity_lds.py for Q0-Q99 first"
        )

    query_id = int(record["query_id"])
    with np.load(cache_path, allow_pickle=False) as cache:
        query_ids = np.asarray(cache["query_ids"], dtype=np.int64).reshape(-1)
        matches = np.flatnonzero(query_ids == query_id)
        if len(matches) != 1:
            raise ValueError(f"Q{query_id} occurs {len(matches)} times in {cache_path}")
        train_clip = np.asarray(cache["train_clip"], dtype=np.float32)
        endpoint_clip = np.asarray(
            cache["endpoint_clip"][int(matches[0])], dtype=np.float32
        )
        clip_model = str(cache["clip_model"].item())
    if train_clip.shape[0] != 10_000 or train_clip.ndim != 2:
        raise ValueError(f"expected train_clip=(10000,D), got {train_clip.shape}")
    if endpoint_clip.shape != (train_clip.shape[1],):
        raise ValueError(
            f"endpoint feature mismatch: {endpoint_clip.shape} vs {train_clip.shape}"
        )

    scores = np.asarray(train_clip @ endpoint_clip, dtype=np.float32)
    if not np.isfinite(scores).all():
        raise ValueError(f"nonfinite CLIP scores for Q{query_id}")
    score_dir = (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
        / f"query_{worker._prompt_path_tag(str(record['prompt']))}"
        / f"initial_seed_{worker.query_seed(record)}"
        / "clip_image_similarity_full10k"
        / "score"
    )
    atomic_npy(score_dir / "scores.npy", scores)
    atomic_npy(score_dir / "score_indices.npy", np.arange(10_000, dtype=np.int64))
    metadata = {
        "algorithm": "clip_image_similarity",
        "clip_model": clip_model,
        "query_id": query_id,
        "score_definition": "cosine(CLIP(training_image), CLIP(query_output))",
        "candidate_count": 10_000,
        "ranking_sign": 1,
        "lds_prediction_sign": -1,
        "feature_cache": str(cache_path),
    }
    info_path = score_dir / "run_info.json"
    temporary = info_path.with_suffix(f".json.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, info_path)
    return score_dir


def score_spec(*, result_root, train_seed, record, method):
    if method != METHOD:
        return _base_score_spec(
            result_root=result_root,
            train_seed=train_seed,
            record=record,
            method=method,
        )
    return (
        prepare_clip_scores(
            result_root=result_root,
            train_seed=train_seed,
            record=record,
        ),
        +1,
        (
            "full-10000 CLIP image cosine similarity to the same-query "
            "full-data output, ranked descending"
        ),
        -1.0,
    )


if __name__ == "__main__":
    worker.METHODS = (*worker.METHODS, METHOD)
    worker.score_spec = score_spec
    worker.main()
