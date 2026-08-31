from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import os

import numpy as np

from .stage_runner import run_stage_config, stage_root
from .config_loader import load_config, require_attr

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional on cluster envs.
    tqdm = None


TRAIN_ARTIFACT = "train_datapoint_gradient_artifact.npz"
QUERY_ARTIFACT = "query_gradient_artifact.npz"


def _iter_with_tqdm(iterable, *, total: int | None, desc: str, enabled: bool = True):
    if enabled and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
    return iterable


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
    if train.shape[0] != query.shape[0] and "checkpoint_shared_train_gradient" in train_payload:
        train_ckpts = np.asarray(train_payload.get("ckpt_indices"), dtype=np.int32).reshape(-1)
        query_ckpts = np.asarray(query_payload.get("ckpt_indices"), dtype=np.int32).reshape(-1)
        if train_ckpts.shape[0] != train.shape[0] or query_ckpts.shape[0] != query.shape[0]:
            raise ValueError("checkpoint-shared TrajTracIn artifacts require ckpt_indices on train and query")
        if train.shape[2] != query.shape[1]:
            raise ValueError(f"feature dimension mismatch: train {train.shape} vs query {query.shape}")
        weights = np.asarray(
            query_payload.get("term_weights", np.full((query.shape[0],), 1.0 / float(query.shape[0]))),
            dtype=np.float64,
        ).reshape(-1)
        if weights.shape[0] != query.shape[0]:
            raise ValueError(f"query term_weights length {weights.shape[0]} does not match query terms {query.shape[0]}")
        by_ckpt = {int(ckpt): i for i, ckpt in enumerate(train_ckpts)}
        missing = sorted({int(ckpt) for ckpt in query_ckpts if int(ckpt) not in by_ckpt})
        if missing:
            raise ValueError(f"query has checkpoint(s) missing from train artifact: {missing[:10]}")
        scores = np.zeros((train.shape[1],), dtype=np.float64)
        for i in range(query.shape[0]):
            scores += float(weights[i]) * (train[by_ckpt[int(query_ckpts[i])]] @ query[i])
        return scores
    if train.shape[0] != query.shape[0] or train.shape[2] != query.shape[1]:
        raise ValueError(f"feature dimension mismatch: train {train.shape} vs query {query.shape}")
    weights = np.asarray(query_payload.get("term_weights", np.full((train.shape[0],), 1.0 / float(train.shape[0]))), dtype=np.float64).reshape(-1)
    if weights.shape[0] != train.shape[0]:
        raise ValueError(f"term_weights length {weights.shape[0]} does not match terms {train.shape[0]}")
    if "term_weights" in train_payload:
        train_weights = np.asarray(train_payload["term_weights"], dtype=np.float64).reshape(-1)
        if train_weights.shape[0] != train.shape[0]:
            raise ValueError(f"train term_weights length {train_weights.shape[0]} does not match terms {train.shape[0]}")
        if not np.allclose(train_weights, weights, rtol=1e-5, atol=1e-12):
            raise ValueError("train/query term_weights differ; regenerate both Traj TracIn artifacts with the same LR schedule")
    scores = np.zeros((train.shape[1],), dtype=np.float64)
    for i in range(train.shape[0]):
        scores += float(weights[i]) * (train[i] @ query[i])
    return scores


def _combine_das_scores(
    train_payload: dict[str, np.ndarray],
    query_payload: dict[str, np.ndarray],
    *,
    train_path: Path,
    query_path: Path,
    damping: float | None = None,
) -> np.ndarray:
    score_dtype = _score_float_dtype()
    use_denominator = _env_flag("DAS_SHERMAN_MORRISON_DENOMINATOR", "1")
    use_tqdm = _env_flag("DAS_SCORE_TQDM", "1")
    denom_batch_size = max(1, int(os.environ.get("DAS_SCORE_DENOM_BATCH_SIZE", "256")))
    train = _first_array(train_payload, ("train_features", "features", "phi", "phis"), path=train_path)
    train = np.asarray(train, dtype=score_dtype)
    if train.ndim == 2:
        train = train[None, :, :]
    if train.ndim != 3:
        raise ValueError(f"{train_path} DAS train features must be rank 2 or 3, got shape {train.shape}")

    query = _first_array(query_payload, ("query_features", "query_feature", "query_gradient", "query_gradients"), path=query_path)
    query = np.asarray(query, dtype=score_dtype)
    if query.ndim == 1:
        query = query[None, :]
    if query.ndim != 2:
        raise ValueError(f"{query_path} DAS query features must be rank 1 or 2, got shape {query.shape}")
    if train.shape[0] != query.shape[0] or train.shape[2] != query.shape[1]:
        raise ValueError(f"feature dimension mismatch: train {train.shape} vs query {query.shape}")

    residual = _first_array(query_payload, ("residuals", "residual", "residual_scalar", "residual_scalars"), path=query_path)
    residual = np.asarray(residual, dtype=score_dtype)
    if residual.ndim == 1:
        residual = residual[None, :]
    if residual.shape[0] == train.shape[0] and residual.shape[1] != train.shape[1]:
        train_indices = np.asarray(train_payload.get("score_indices", ()), dtype=np.int64).reshape(-1)
        max_index = int(train_indices.max()) if train_indices.size else -1
        if train_indices.shape[0] == train.shape[1] and residual.shape[1] > max_index:
            residual = residual[:, train_indices]
    if residual.shape[0] != train.shape[0] or residual.shape[1] != train.shape[1]:
        raise ValueError(f"residual shape {residual.shape} does not match train features {train.shape}")

    gram_inv = None
    gram = None
    if damping is not None and "gram_undamped" in train_payload:
        gram = np.asarray(train_payload["gram_undamped"], dtype=score_dtype)
        if gram.ndim == 2:
            gram = gram[None, :, :]
        eye = np.eye(gram.shape[-1], dtype=score_dtype)
        gram = gram + float(damping) * eye[None, :, :]
    else:
        for key in ("gram", "gram_matrix", "H", "H_proj"):
            if key in train_payload:
                gram = np.asarray(train_payload[key], dtype=score_dtype)
                break
    for key in ("gram_inverse", "inverse_gram", "gram_inv", "h_inverse"):
        if key in train_payload:
            gram_inv = np.asarray(train_payload[key], dtype=score_dtype)
            break
    if gram_inv is not None and gram_inv.ndim == 2:
        gram_inv = gram_inv[None, :, :]
    if gram is not None and gram.ndim == 2:
        gram = gram[None, :, :]

    print(
        "[das-score] "
        f"terms={train.shape[0]} points={train.shape[1]} dim={train.shape[2]} "
        f"damping={damping} dtype={np.dtype(score_dtype).name} denominator={int(use_denominator)} "
        f"denom_batch={denom_batch_size}",
        flush=True,
    )
    scores = np.zeros((train.shape[1],), dtype=np.float64)
    term_iter = _iter_with_tqdm(
        range(train.shape[0]),
        total=train.shape[0],
        desc=f"DAS score lambda={_damping_tag(damping or 0)}",
        enabled=use_tqdm,
    )
    for i in term_iter:
        print(f"[das-score] term {i + 1}/{train.shape[0]} | solving query", flush=True)
        if gram_inv is not None:
            inv = gram_inv[i]
            u = inv @ query[i]
        elif gram is not None:
            inv = None
            u = np.linalg.solve(gram[i], query[i])
        else:
            inv = None
            u = query[i]
        raw = (train[i] @ u) * residual[i]
        if use_denominator and (gram_inv is not None or gram is not None):
            print(f"[das-score] term {i + 1}/{train.shape[0]} | denominator", flush=True)
            if inv is not None:
                solved_train = train[i] @ inv.T
                leverage = np.einsum("md,md->m", train[i], solved_train, dtype=np.float64)
            else:
                leverage = np.empty((train.shape[1],), dtype=np.float64)
                starts = range(0, train.shape[1], denom_batch_size)
                denom_iter = _iter_with_tqdm(
                    starts,
                    total=(train.shape[1] + denom_batch_size - 1) // denom_batch_size,
                    desc=f"DAS denom term={i + 1}/{train.shape[0]} lambda={_damping_tag(damping or 0)}",
                    enabled=use_tqdm,
                )
                for start in denom_iter:
                    end = min(start + denom_batch_size, train.shape[1])
                    phi_chunk = train[i, start:end]
                    solved_train = np.linalg.solve(gram[i], phi_chunk.T).T
                    leverage[start:end] = np.einsum("md,md->m", phi_chunk, solved_train, dtype=np.float64)
                    if hasattr(denom_iter, "set_postfix"):
                        denom_iter.set_postfix(samples=f"{end}/{train.shape[1]}")
            denom = 1.0 - leverage
            denom = np.where(
                np.abs(denom) < 1e-6,
                np.where(denom >= 0.0, 1e-6, -1e-6),
                denom,
            )
            raw = raw / denom
        scores += np.square(raw)
        print(f"[das-score] term {i + 1}/{train.shape[0]} done", flush=True)
    return scores / float(train.shape[0])


def _write_score_outputs(
    out_dir: Path,
    scores: np.ndarray,
    indices: np.ndarray,
    *,
    train_dir: Path,
    query_dir: Path,
    algorithm: str,
    extra_manifest: dict[str, Any] | None = None,
) -> None:
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
        manifest = {
                "algorithm": algorithm,
                "train_artifact_dir": str(train_dir),
                "query_artifact_dir": str(query_dir),
                "score_dir": str(out_dir),
                "num_scores": int(len(scores)),
                "mode": "pure_artifact_combiner",
        }
        if extra_manifest:
            manifest.update(extra_manifest)
        json.dump(manifest, handle, indent=2)


def _damping_tag(value: float) -> str:
    text = f"{float(value):g}".replace("+", "").replace("-", "neg_").replace(".", "p")
    return text or "0"


def _parse_float_list(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.replace(",", " ").split() if part.strip())


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False", "no", "No")


def _score_float_dtype() -> np.dtype:
    return np.float64 if _env_flag("DAS_SCORE_FLOAT64", "0") else np.float32


def _das_damping_values(config_path: Path, train_payload: dict[str, np.ndarray]) -> tuple[float, ...]:
    if os.environ.get("DAS_DAMPING_SWEEP", "0") not in ("1", "true", "True", "yes"):
        if "damping" in train_payload:
            return (float(np.asarray(train_payload["damping"]).reshape(())),)
        return (float(os.environ.get("DAS_DAMPING", "2")),)
    if os.environ.get("DAS_DAMPING_SWEEP_VALUES"):
        return _parse_float_list(os.environ["DAS_DAMPING_SWEEP_VALUES"])
    if "damping_sweep_values" in train_payload:
        values = tuple(float(v) for v in np.asarray(train_payload["damping_sweep_values"]).reshape(-1))
        if values:
            return values
    cfg_module = load_config(config_path)
    config_values = dict(require_attr(cfg_module, "ATTRIBUTION_CONFIG"))
    values = tuple(float(v) for v in config_values.get("damping_sweep_values", ()))
    return values or (float(config_values.get("damping", 2.0)),)


def run_score_combination_stage(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    out_dir = run_stage_config(config_path, "score")
    train_dir = stage_root(config_path, "train_datapoint_gradient")
    query_dir = stage_root(config_path, "query_gradient")
    out_dir = Path(os.environ.get("SCORE_OUTPUT_DIR", out_dir))

    train_path = Path(os.environ.get("TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH", train_dir / TRAIN_ARTIFACT))
    query_path = Path(os.environ.get("QUERY_GRADIENT_ARTIFACT_PATH", query_dir / QUERY_ARTIFACT))
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
        global_gram_path = os.environ.get("DAS_GLOBAL_GRAM_ARTIFACT_PATH")
        if global_gram_path:
            global_gram_payload = _load_npz(Path(global_gram_path))
            for key in ("gram", "gram_undamped", "damping", "damping_sweep_values"):
                if key in global_gram_payload:
                    train_payload[key] = global_gram_payload[key]
        damping_values = _das_damping_values(config_path, train_payload)
        written_dirs = []
        sweep = len(damping_values) > 1 or os.environ.get("DAS_DAMPING_SWEEP", "0") in ("1", "true", "True", "yes")
        for damping in damping_values:
            scores = _combine_das_scores(
                train_payload,
                query_payload,
                train_path=train_path,
                query_path=query_path,
                damping=float(damping),
            )
            indices = _score_indices(train_payload, len(scores))
            target_dir = out_dir / f"lambda_{_damping_tag(damping)}" if sweep else out_dir
            _write_score_outputs(
                target_dir,
                scores,
                indices,
                train_dir=train_dir,
                query_dir=query_dir,
                algorithm=algorithm,
                extra_manifest={
                    "damping": float(damping),
                    "damping_sweep_enabled": bool(sweep),
                    "damping_sweep_values": [float(v) for v in damping_values],
                },
            )
            written_dirs.append(str(target_dir))
        print(f"[score] combined DAS scores for {len(damping_values)} damping value(s): {written_dirs}")
        return out_dir
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
