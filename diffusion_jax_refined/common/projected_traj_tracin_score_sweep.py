from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_PROJ_DIMS = (2048, 4096, 8192, 16384, 32768)
TRAIN_KEYS = ("train_features", "features", "train_gradients", "gradients")
QUERY_KEYS = ("query_features", "query_feature", "query_gradient", "query_gradients")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def first_array(payload: dict[str, np.ndarray], names: tuple[str, ...], *, path: Path) -> np.ndarray:
    for name in names:
        if name in payload:
            return np.asarray(payload[name])
    raise KeyError(f"{path} is missing one of {names}")


def parse_proj_dims(text: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not dims:
        raise ValueError("projection dimension list is empty")
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"projection dimensions must be positive, got {dims}")
    return dims


def parse_score_index_ranges(text: str | None) -> tuple[tuple[int, int], ...] | None:
    if text is None or not str(text).strip():
        return None
    out = []
    for part in str(text).replace(",", " ").split():
        if "-" not in part:
            raise ValueError(f"invalid score index range {part!r}; expected START-END")
        start, end = part.split("-", 1)
        start_i = int(start)
        end_i = int(end)
        if end_i < start_i:
            raise ValueError(f"invalid score index range {part!r}; end < start")
        out.append((start_i, end_i))
    return tuple(out)


def score_indices(train_payload: dict[str, np.ndarray], n: int) -> np.ndarray:
    for key in ("score_indices", "indices", "train_indices", "datapoint_indices"):
        if key in train_payload:
            indices = np.asarray(train_payload[key], dtype=np.int64).reshape(-1)
            if len(indices) != n:
                raise ValueError(f"{key} length {len(indices)} does not match score length {n}")
            return indices
    return np.arange(n, dtype=np.int64)


def range_mask(indices: np.ndarray, ranges: tuple[tuple[int, int], ...] | None, *, index_base: int) -> np.ndarray:
    if ranges is None:
        return np.ones((len(indices),), dtype=bool)
    if index_base not in (0, 1):
        raise ValueError(f"score index base must be 0 or 1, got {index_base}")
    comparable = np.asarray(indices, dtype=np.int64) + (1 if index_base == 1 else 0)
    mask = np.zeros((len(indices),), dtype=bool)
    for start, end in ranges:
        mask |= (comparable >= int(start)) & (comparable <= int(end))
    return mask


def as_multiterm_features(
    train_payload: dict[str, np.ndarray],
    query_payload: dict[str, np.ndarray],
    *,
    train_path: Path,
    query_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(first_array(train_payload, TRAIN_KEYS, path=train_path), dtype=np.float32)
    query = np.asarray(first_array(query_payload, QUERY_KEYS, path=query_path), dtype=np.float32)
    if train.ndim == 2:
        train = train[None, :, :]
    if query.ndim == 1:
        query = query[None, :]
    if train.ndim != 3:
        raise ValueError(f"{train_path} train features must be rank 2 or 3, got {train.shape}")
    if query.ndim != 2:
        raise ValueError(f"{query_path} query features must be rank 1 or 2, got {query.shape}")
    if train.shape[0] != query.shape[0]:
        train_ckpts = np.asarray(train_payload.get("ckpt_indices", []), dtype=np.int32).reshape(-1)
        query_ckpts = np.asarray(query_payload.get("ckpt_indices", []), dtype=np.int32).reshape(-1)
        if len(train_ckpts) == train.shape[0] and len(query_ckpts) == query.shape[0]:
            query_weights = np.asarray(
                query_payload.get("term_weights", np.full((query.shape[0],), 1.0 / float(query.shape[0]))),
                dtype=np.float64,
            ).reshape(-1)
            if len(query_weights) != query.shape[0]:
                raise ValueError(
                    f"{query_path} query term_weights length {len(query_weights)} "
                    f"does not match query terms {query.shape[0]}"
                )
            aggregated = np.zeros((train.shape[0], query.shape[1]), dtype=np.float64)
            for term_id, ckpt in enumerate(train_ckpts):
                mask = query_ckpts == ckpt
                if not np.any(mask):
                    raise ValueError(f"no query terms found for train ckpt index {int(ckpt)}")
                aggregated[term_id] = (
                    query[mask].astype(np.float64) * query_weights[mask, None]
                ).sum(axis=0)
            query = aggregated.astype(np.float32)
            train_payload["_aligned_query_terms_are_weighted"] = np.asarray(True)
        else:
            raise ValueError(f"feature mismatch: train {train.shape} vs query {query.shape}")
    if train.shape[2] != query.shape[1]:
        raise ValueError(f"feature mismatch: train {train.shape} vs query {query.shape}")
    return train, query


def term_weights(
    train_payload: dict[str, np.ndarray],
    query_payload: dict[str, np.ndarray],
    num_terms: int,
) -> np.ndarray:
    if bool(np.asarray(train_payload.get("_aligned_query_terms_are_weighted", False)).item()):
        return np.ones((num_terms,), dtype=np.float64)

    weights = np.asarray(
        query_payload.get("term_weights", np.full((num_terms,), 1.0 / float(num_terms))),
        dtype=np.float64,
    ).reshape(-1)
    if len(weights) != num_terms:
        raise ValueError(f"query term_weights length {len(weights)} does not match terms {num_terms}")
    if "term_weights" in train_payload:
        train_weights = np.asarray(train_payload["term_weights"], dtype=np.float64).reshape(-1)
        if len(train_weights) != num_terms:
            raise ValueError(f"train term_weights length {len(train_weights)} does not match terms {num_terms}")
        if not np.allclose(train_weights, weights, rtol=1e-5, atol=1e-12):
            raise ValueError("train/query term_weights differ")
    return weights


def checkpoint_uniform_term_weights(
    payload: dict[str, np.ndarray],
    num_terms: int,
) -> np.ndarray:
    ckpt_indices = np.asarray(payload.get("ckpt_indices", []), dtype=np.int64).reshape(-1)
    if len(ckpt_indices) != num_terms:
        return np.full((num_terms,), 1.0 / float(max(1, num_terms)), dtype=np.float64)

    weights = np.zeros((num_terms,), dtype=np.float64)
    unique_ckpts = np.unique(ckpt_indices)
    if len(unique_ckpts) == 0:
        return np.full((num_terms,), 1.0 / float(max(1, num_terms)), dtype=np.float64)
    per_ckpt = 1.0 / float(len(unique_ckpts))
    for ckpt in unique_ckpts:
        mask = ckpt_indices == ckpt
        weights[mask] = per_ckpt / float(np.count_nonzero(mask))
    return weights


def resolve_term_weights(
    train_payload: dict[str, np.ndarray],
    query_payload: dict[str, np.ndarray],
    num_terms: int,
    mode: str,
) -> np.ndarray:
    if mode == "artifact":
        return term_weights(train_payload, query_payload, num_terms)
    if mode in ("uniform", "uniform_checkpoint"):
        return checkpoint_uniform_term_weights(train_payload, num_terms)
    if mode == "uniform_term":
        return np.full((num_terms,), 1.0 / float(max(1, num_terms)), dtype=np.float64)
    raise ValueError(
        f"unknown term weighting {mode!r}; expected artifact, uniform_checkpoint, or uniform_term"
    )


def l2_normalize_rows(x: np.ndarray, eps: float) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(denom, eps)


def l2_normalize_terms(x: np.ndarray, eps: float) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(denom, eps)


def combine_multiterm_dot(train: np.ndarray, query: np.ndarray, weights: np.ndarray) -> np.ndarray:
    scores = np.zeros((train.shape[1],), dtype=np.float64)
    for term_id in range(train.shape[0]):
        scores += float(weights[term_id]) * (train[term_id].astype(np.float64) @ query[term_id].astype(np.float64))
    return scores


def output_dir_for_variant(root: Path, proj_dim: int, variant: str) -> Path:
    return root / f"proj_{int(proj_dim)}" / variant


def write_score_outputs(
    out_dir: Path,
    scores: np.ndarray,
    indices: np.ndarray,
    *,
    proj_dim: int,
    variant: str,
    metadata: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if len(scores) != len(indices):
        raise ValueError(f"scores length {len(scores)} does not match indices length {len(indices)}")

    np.save(out_dir / "scores.npy", scores)
    np.save(out_dir / "score_indices.npy", indices)

    order = np.argsort(-scores)
    top = [
        {
            "rank": int(rank),
            "idx": int(indices[i]),
            "idx_1based": int(indices[i]) + 1,
            "score": float(scores[i]),
        }
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
        payload = dict(metadata)
        payload.update(
            {
                "algorithm": "traj_tracin_projected",
                "proj_dim": int(proj_dim),
                "score_variant": variant,
                "score_dir": str(out_dir),
                "num_scores": int(len(scores)),
            }
        )
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def selected_variants(include_raw: bool) -> tuple[str, ...]:
    variants = (
        "query_l2_normalized",
        "train_l2_normalized",
        "query_train_l2_normalized",
    )
    if include_raw:
        return ("raw",) + variants
    return variants


def score_projection_dims(
    train_path: Path,
    query_path: Path,
    out_root: Path,
    proj_dims: Iterable[int],
    *,
    normalize_eps: float,
    include_raw: bool,
    score_index_ranges: tuple[tuple[int, int], ...] | None,
    score_index_base: int,
    term_weighting: str,
) -> list[Path]:
    train_payload = load_npz(train_path)
    query_payload = load_npz(query_path)
    train_all, query_all = as_multiterm_features(
        train_payload,
        query_payload,
        train_path=train_path,
        query_path=query_path,
    )
    dims = tuple(int(dim) for dim in proj_dims)
    max_dim = train_all.shape[-1]
    too_large = [dim for dim in dims if dim > max_dim]
    if too_large:
        raise ValueError(f"requested projection dims {too_large} exceed cached feature dim {max_dim}")

    weights = resolve_term_weights(train_payload, query_payload, train_all.shape[0], term_weighting)
    indices_all = score_indices(train_payload, train_all.shape[1])
    keep = range_mask(indices_all, score_index_ranges, index_base=int(score_index_base))
    if not np.any(keep):
        raise ValueError(f"score_index_ranges={score_index_ranges} selected no train features")
    train_all = train_all[:, keep, :]
    indices = indices_all[keep]
    metadata = {
        "train_artifact": str(train_path),
        "query_artifact": str(query_path),
        "cached_feature_dim": int(max_dim),
        "num_terms": int(train_all.shape[0]),
        "normalization": {
            "eps": float(normalize_eps),
            "query_l2_normalized": True,
            "train_l2_normalized": True,
            "both_l2_normalized_variant": "query_train_l2_normalized",
        },
        "score_index_ranges": score_index_ranges,
        "score_index_base": int(score_index_base),
        "term_weighting": term_weighting,
        "term_weight_sum": float(np.sum(weights)),
    }

    written = []
    for proj_dim in dims:
        train = train_all[:, :, :proj_dim]
        query = query_all[:, :proj_dim]

        train_norm = l2_normalize_rows(train, normalize_eps)
        query_norm = l2_normalize_terms(query, normalize_eps)

        scores_by_variant = {
            "query_l2_normalized": combine_multiterm_dot(train, query_norm, weights),
            "train_l2_normalized": combine_multiterm_dot(train_norm, query, weights),
            "query_train_l2_normalized": combine_multiterm_dot(train_norm, query_norm, weights),
        }
        if include_raw:
            scores_by_variant = {"raw": combine_multiterm_dot(train, query, weights), **scores_by_variant}

        for variant in selected_variants(include_raw):
            out_dir = output_dir_for_variant(out_root, proj_dim, variant)
            write_score_outputs(
                out_dir,
                scores_by_variant[variant],
                indices,
                proj_dim=proj_dim,
                variant=variant,
                metadata=metadata,
            )
            written.append(out_dir)
            print(f"[saved] {variant} proj_dim={proj_dim}: {out_dir}")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine projected Traj-TracIn train/query artifacts across projection "
            "dimensions and write normalized score variants."
        )
    )
    parser.add_argument("--train-artifact", required=True, type=Path)
    parser.add_argument("--query-artifact", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument(
        "--proj-dims",
        default=",".join(str(x) for x in DEFAULT_PROJ_DIMS),
        help="Comma-separated projected dimensions. Smaller dims use the prefix of the cached max feature.",
    )
    parser.add_argument("--normalize-eps", type=float, default=1e-8)
    parser.add_argument("--score-index-ranges", default=None, help="Optional START-END ranges to filter final scores.")
    parser.add_argument("--score-index-base", type=int, default=1, choices=(0, 1))
    parser.add_argument(
        "--term-weighting",
        choices=("artifact", "uniform", "uniform_checkpoint", "uniform_term"),
        default="artifact",
        help=(
            "How to weight checkpoint/snapshot terms. artifact preserves cached "
            "learning-rate weights; uniform/uniform_checkpoint gives every checkpoint "
            "equal total weight; uniform_term gives every checkpoint-snapshot term equal weight."
        ),
    )
    parser.add_argument("--no-raw", action="store_true", help="Only write normalized variants.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_projection_dims(
        args.train_artifact,
        args.query_artifact,
        args.out_root,
        parse_proj_dims(args.proj_dims),
        normalize_eps=float(args.normalize_eps),
        include_raw=not bool(args.no_raw),
        score_index_ranges=parse_score_index_ranges(args.score_index_ranges),
        score_index_base=int(args.score_index_base),
        term_weighting=str(args.term_weighting),
    )


if __name__ == "__main__":
    main()
