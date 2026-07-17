from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .stage_runner import run_stage_config, stage_root


TRAIN_ARTIFACT = "train_datapoint_gradient_artifact.npz"
QUERY_ARTIFACT = "query_gradient_artifact.npz"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _first_array(payload: dict[str, np.ndarray], names: tuple[str, ...], *, path: Path) -> np.ndarray:
    for name in names:
        if name in payload:
            return np.asarray(payload[name])
    raise KeyError(f"{path} is missing one of {names}")


def _score_indices(train_payload: dict[str, np.ndarray], n: int) -> np.ndarray:
    for key in ("score_indices", "indices", "train_indices", "datapoint_indices"):
        if key in train_payload:
            indices = np.asarray(train_payload[key], dtype=np.int64).reshape(-1)
            if len(indices) != n:
                raise ValueError(f"{key} length {len(indices)} does not match scores length {n}")
            return indices
    return np.arange(n, dtype=np.int64)


def _query_vector(query_payload: dict[str, np.ndarray], *, path: Path) -> np.ndarray:
    q = _first_array(query_payload, ("query_feature", "query_features", "query_gradient", "query_gradients"), path=path)
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        return q
    if q.ndim == 2:
        return q.sum(axis=0)
    raise ValueError(f"{path} query feature must be rank 1 or 2, got shape {q.shape}")


def _combine_dot_scores(train_payload: dict[str, np.ndarray], query_payload: dict[str, np.ndarray], *, train_path: Path, query_path: Path) -> np.ndarray:
    train = _first_array(train_payload, ("train_features", "features", "train_gradients", "gradients"), path=train_path)
    train = np.asarray(train, dtype=np.float64)
    if train.ndim != 2:
        raise ValueError(f"{train_path} train features must be rank 2, got shape {train.shape}")
    query = _query_vector(query_payload, path=query_path)
    if train.shape[1] != query.shape[0]:
        raise ValueError(f"feature dimension mismatch: train {train.shape} vs query {query.shape}")
    return train @ query


def _combine_dtrak_scores(train_payload: dict[str, np.ndarray], query_payload: dict[str, np.ndarray], *, train_path: Path, query_path: Path) -> np.ndarray:
    train = _first_array(train_payload, ("train_features", "features"), path=train_path)
    train = np.asarray(train, dtype=np.float64)
    if train.ndim == 2:
        return _combine_dot_scores(train_payload, query_payload, train_path=train_path, query_path=query_path)
    if train.ndim != 3:
        raise ValueError(f"{train_path} D-TRAK train features must be rank 3, got shape {train.shape}")

    query = _first_array(query_payload, ("query_features", "query_feature"), path=query_path)
    query = np.asarray(query, dtype=np.float64)
    if query.ndim == 1:
        query = query[None, :]
    if query.ndim != 2:
        raise ValueError(f"{query_path} D-TRAK query features must be rank 2, got shape {query.shape}")
    if train.shape[0] != query.shape[0] or train.shape[2] != query.shape[1]:
        raise ValueError(f"D-TRAK feature mismatch: train {train.shape} vs query {query.shape}")

    gram = _first_array(train_payload, ("gram", "gram_matrix", "G"), path=train_path)
    gram = np.asarray(gram, dtype=np.float64)
    if gram.ndim == 2:
        gram = gram[None, :, :]
    if gram.shape[0] != train.shape[0] or gram.shape[1] != train.shape[2] or gram.shape[2] != train.shape[2]:
        raise ValueError(f"D-TRAK Gram mismatch: train {train.shape} vs gram {gram.shape}")

    scores = np.zeros((train.shape[1],), dtype=np.float64)
    for i in range(train.shape[0]):
        u = np.linalg.solve(gram[i], query[i])
        scores += train[i] @ u
    return scores / float(train.shape[0])


def _combine_multiterm_dot_scores(train_payload: dict[str, np.ndarray], query_payload: dict[str, np.ndarray], *, train_path: Path, query_path: Path) -> np.ndarray:
    train = _first_array(train_payload, ("train_features", "features", "train_gradients", "gradients"), path=train_path)
    train = np.asarray(train, dtype=np.float64)
    if train.ndim == 2:
        return _combine_dot_scores(train_payload, query_payload, train_path=train_path, query_path=query_path)
    if train.ndim != 3:
        raise ValueError(f"{train_path} train features must be rank 2 or 3, got shape {train.shape}")
    query = _first_array(query_payload, ("query_features", "query_feature", "query_gradient", "query_gradients"), path=query_path)
    query = np.asarray(query, dtype=np.float64)
    if query.ndim == 1:
        query = query[None, :]
    if query.ndim != 2:
        raise ValueError(f"{query_path} query features must be rank 1 or 2, got shape {query.shape}")
    if train.shape[0] != query.shape[0] or train.shape[2] != query.shape[1]:
        raise ValueError(f"feature dimension mismatch: train {train.shape} vs query {query.shape}")
    weights = np.asarray(query_payload.get("term_weights", np.full((train.shape[0],), 1.0 / float(train.shape[0]))), dtype=np.float64).reshape(-1)
    if weights.shape[0] != train.shape[0]:
        raise ValueError(f"term_weights length {weights.shape[0]} does not match terms {train.shape[0]}")
    scores = np.zeros((train.shape[1],), dtype=np.float64)
    for i in range(train.shape[0]):
        scores += float(weights[i]) * (train[i] @ query[i])
    return scores


def _combine_das_scores(train_payload: dict[str, np.ndarray], query_payload: dict[str, np.ndarray], *, train_path: Path, query_path: Path) -> np.ndarray:
    train = _first_array(train_payload, ("train_features", "features", "phi", "phis"), path=train_path)
    train = np.asarray(train, dtype=np.float64)
    if train.ndim == 2:
        train = train[None, :, :]
    if train.ndim != 3:
        raise ValueError(f"{train_path} DAS train features must be rank 2 or 3, got shape {train.shape}")

    query = _first_array(query_payload, ("query_features", "query_feature", "query_gradient", "query_gradients"), path=query_path)
    query = np.asarray(query, dtype=np.float64)
    if query.ndim == 1:
        query = query[None, :]
    if query.ndim != 2:
        raise ValueError(f"{query_path} DAS query features must be rank 1 or 2, got shape {query.shape}")
    if train.shape[0] != query.shape[0] or train.shape[2] != query.shape[1]:
        raise ValueError(f"feature dimension mismatch: train {train.shape} vs query {query.shape}")

    residual = _first_array(query_payload, ("residuals", "residual", "residual_scalar", "residual_scalars"), path=query_path)
    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim == 1:
        residual = residual[None, :]
    if residual.shape[0] != train.shape[0] or residual.shape[1] != train.shape[1]:
        raise ValueError(f"residual shape {residual.shape} does not match train features {train.shape}")

    gram_inv = None
    gram = None
    for key in ("gram", "gram_matrix", "H", "H_proj"):
        if key in train_payload:
            gram = np.asarray(train_payload[key], dtype=np.float64)
            break
    for key in ("gram_inverse", "inverse_gram", "gram_inv", "h_inverse"):
        if key in train_payload:
            gram_inv = np.asarray(train_payload[key], dtype=np.float64)
            break
    if gram_inv is not None and gram_inv.ndim == 2:
        gram_inv = gram_inv[None, :, :]
    if gram is not None and gram.ndim == 2:
        gram = gram[None, :, :]

    scores = np.zeros((train.shape[1],), dtype=np.float64)
    for i in range(train.shape[0]):
        if gram_inv is not None:
            inv = gram_inv[i]
            u = inv @ query[i]
        elif gram is not None:
            inv = np.linalg.inv(gram[i])
            u = inv @ query[i]
        else:
            inv = None
            u = query[i]
        raw = (train[i] @ u) * residual[i]
        if inv is not None:
            leverage = np.einsum("md,dd,md->m", train[i], inv, train[i])
            denom = np.maximum(1.0 - leverage, 1e-12)
            raw = raw / denom
        scores += np.square(raw)
    return scores / float(train.shape[0])


def _write_score_outputs(out_dir: Path, scores: np.ndarray, indices: np.ndarray, *, train_dir: Path, query_dir: Path, algorithm: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if scores.shape[0] != indices.shape[0]:
        raise ValueError(f"scores length {scores.shape[0]} does not match indices length {indices.shape[0]}")

    np.save(out_dir / "scores.npy", scores)
    np.save(out_dir / "score_indices.npy", indices)

    order = np.argsort(-scores)
    top = [
        {"rank": int(rank), "idx": int(indices[i]), "idx_1based": int(indices[i]) + 1, "score": float(scores[i])}
        for rank, i in enumerate(order[: min(2000, len(order))], start=1)
    ]
    with open(out_dir / "top_scores.json", "w") as handle:
        json.dump({"top": top, "num_scored": int(len(scores))}, handle, indent=2)
    with open(out_dir / "score_indices.json", "w") as handle:
        json.dump(
            {
                "score_indices": [int(x) for x in indices],
                "score_indices_1based": [int(x) + 1 for x in indices],
            },
            handle,
            indent=2,
        )
    with open(out_dir / "score_artifact_manifest.json", "w") as handle:
        json.dump(
            {
                "algorithm": algorithm,
                "train_artifact_dir": str(train_dir),
                "query_artifact_dir": str(query_dir),
                "score_dir": str(out_dir),
                "num_scores": int(len(scores)),
                "mode": "pure_artifact_combiner",
            },
            handle,
            indent=2,
        )


def run_score_combination_stage(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    out_dir = run_stage_config(config_path, "score")
    train_dir = stage_root(config_path, "train_datapoint_gradient")
    query_dir = stage_root(config_path, "query_gradient")

    train_path = train_dir / TRAIN_ARTIFACT
    query_path = query_dir / QUERY_ARTIFACT
    missing = [str(path) for path in (train_path, query_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Score stage is pure now and will not call the monolithic legacy engine. "
            "Missing required artifact(s): " + ", ".join(missing)
        )

    train_payload = _load_npz(train_path)
    query_payload = _load_npz(query_path)
    algorithm = config_path.parent.name
    if algorithm == "das":
        scores = _combine_das_scores(train_payload, query_payload, train_path=train_path, query_path=query_path)
    elif algorithm == "dtrak":
        scores = _combine_dtrak_scores(train_payload, query_payload, train_path=train_path, query_path=query_path)
    elif algorithm in ("end_tracin", "traj_tracin"):
        scores = _combine_multiterm_dot_scores(train_payload, query_payload, train_path=train_path, query_path=query_path)
    else:
        scores = _combine_dot_scores(train_payload, query_payload, train_path=train_path, query_path=query_path)
    indices = _score_indices(train_payload, len(scores))
    _write_score_outputs(out_dir, scores, indices, train_dir=train_dir, query_dir=query_dir, algorithm=algorithm)
    print(f"[score] combined {len(scores)} scores from stage artifacts")
    return out_dir
