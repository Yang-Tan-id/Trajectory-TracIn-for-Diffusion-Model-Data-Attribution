#!/usr/bin/env python3
"""Score DAS with independently inverted local train groups, then mix scores.

For every requested group count, the 5000 cached train points are shuffled once
and split into disjoint groups.  Each group constructs its own local DAS Gram,
scores only its own points, and writes the concatenated 5000-point score vector.
No train or query gradients are recomputed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
for path in (SHAPES_ROOT, REFINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_config import DAS_DAMPING_SWEEP_VALUES, _prompt_tag
from common.stage_artifact_runner import (
    _das_score_term_indices,
    _das_term_ids,
    _first_array,
    _require_matching_das_terms,
    _score_indices,
    _write_score_outputs,
)


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.replace(",", " ").split() if value.strip()]


def damping_tag(value: float) -> str:
    return f"{float(value):g}".replace("+", "").replace("-", "neg_").replace(".", "p")


def load_npz(path: Path, *, keys: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with np.load(path, allow_pickle=False) as payload:
        selected = payload.files if keys is None else [key for key in keys if key in payload.files]
        return {key: np.asarray(payload[key]) for key in selected}


def local_group_term_scores(
    features: np.ndarray,
    residuals: np.ndarray,
    queries: np.ndarray,
    lambdas: list[float],
    *,
    device: str,
    denominator_floor: float,
) -> np.ndarray:
    """Return squared DAS scores with a local Gram for one term and group.

    Uses the dual identity
      G (G^T G + lambda I)^-1 q = (G G^T + lambda I)^-1 G q
    and an eigendecomposition of K=G G^T.  The same decomposition also gives
    the exact Sherman--Morrison denominator
      1 - diag(G (G^T G + lambda I)^-1 G^T)
        = lambda * diag((K + lambda I)^-1).
    """
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:  # pragma: no cover - cluster dependency
        raise RuntimeError("score_das_local_gram_mixed.py requires JAX") from exc

    if features.ndim != 2 or queries.ndim != 2:
        raise ValueError("features and queries must both be matrices")
    if features.shape[1] != queries.shape[1]:
        raise ValueError(f"feature/query dimension mismatch: {features.shape} vs {queries.shape}")
    if residuals.shape != (features.shape[0],):
        raise ValueError(f"residual shape {residuals.shape} does not match {features.shape[0]} points")

    requested_platform = device.split(":", 1)[0].lower()
    devices = jax.devices(requested_platform)
    device_index = int(device.split(":", 1)[1]) if ":" in device else 0
    if device_index >= len(devices):
        raise ValueError(f"requested {device}, available {requested_platform} devices={devices}")
    jax_device = devices[device_index]
    g = jax.device_put(jnp.asarray(features, dtype=jnp.float32), jax_device)
    q = jax.device_put(jnp.asarray(queries, dtype=jnp.float32), jax_device)
    residual = jax.device_put(jnp.asarray(residuals, dtype=jnp.float32), jax_device)
    kernel = g @ g.T
    eigenvalues, eigenvectors = jnp.linalg.eigh(kernel)
    # Small negative eigenvalues are floating-point artifacts of a PSD kernel.
    eigenvalues = jnp.maximum(eigenvalues, 0.0)
    projected_rhs = eigenvectors.T @ (g @ q.T)
    eigenvectors_sq = jnp.square(eigenvectors)

    result = np.empty(
        (len(lambdas), queries.shape[0], features.shape[0]),
        dtype=np.float64,
    )
    for lambda_i, damping in enumerate(lambdas):
        inverse_spectrum = jnp.reciprocal(eigenvalues + float(damping))
        contraction = eigenvectors @ (projected_rhs * inverse_spectrum[:, None])
        denominator = float(damping) * (eigenvectors_sq @ inverse_spectrum)
        sign = jnp.where(denominator >= 0, 1.0, -1.0)
        denominator = jnp.where(
            jnp.abs(denominator) < denominator_floor,
            sign * denominator_floor,
            denominator,
        )
        raw = contraction * residual[:, None] / denominator[:, None]
        result[lambda_i] = np.asarray(jax.device_get(jnp.square(raw).T), dtype=np.float64)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mix DAS scores from disjoint groups with independently constructed local Gram inverses."
    )
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--query-file", type=Path, default=SHAPES_ROOT / "queries_in_distribution_plus_zero_seed_100_219.json")
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--group-counts", default="2,4,8,10,20")
    parser.add_argument("--partition-seed", type=int, default=0)
    parser.add_argument("--train-artifact-namespace", default="factorized_mc4_reference100x1")
    parser.add_argument("--query-artifact-namespace", default="")
    parser.add_argument("--score-namespace-prefix", default="factorized_mc4_indist100q_original100x1_localgram_mix")
    parser.add_argument(
        "--lambdas",
        default=",".join(f"{float(value):g}" for value in DAS_DAMPING_SWEEP_VALUES),
    )
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--denominator-floor", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    query_ids = parse_ints(args.query_ids)
    group_counts = parse_ints(args.group_counts)
    lambdas = parse_floats(args.lambdas)
    if not query_ids or not group_counts or not lambdas:
        raise ValueError("query ids, group counts, and lambdas must all be non-empty")
    if any(value <= 0 for value in group_counts):
        raise ValueError(f"group counts must be positive: {group_counts}")
    if any(value <= 0 for value in lambdas):
        raise ValueError(f"DAS lambdas must be positive: {lambdas}")

    records = json.loads(args.query_file.read_text())["queries"]
    result_root = SHAPES_ROOT / "result" / args.experiment
    train_name = f"das_{args.train_artifact_namespace.strip('_/')}"
    train_path = (
        result_root / "model" / "prompted_solo" / f"seed_{args.train_seed}_train_gradient"
        / train_name / "train_datapoint_gradient_artifact.npz"
    )
    print(f"[load] train artifact: {train_path}", flush=True)
    train_payload = load_npz(
        train_path,
        keys=(
            "train_features", "features", "phi", "phis",
            "residuals", "residual", "residual_scalar", "residual_scalars",
            "score_indices", "indices", "train_indices", "datapoint_indices",
            "ckpt_indices", "timesteps", "mc_indices",
        ),
    )
    train = np.asarray(
        _first_array(train_payload, ("train_features", "features", "phi", "phis"), path=train_path),
        dtype=np.float32,
    )
    residuals = np.asarray(
        _first_array(
            train_payload,
            ("residuals", "residual", "residual_scalar", "residual_scalars"),
            path=train_path,
        ),
        dtype=np.float32,
    )
    if train.ndim == 2:
        train = train[None, :, :]
    if residuals.ndim == 1:
        residuals = residuals[None, :]
    if residuals.shape != train.shape[:2]:
        raise ValueError(f"residual shape {residuals.shape} does not match train {train.shape}")
    indices = _score_indices(train_payload, train.shape[1])
    term_ids = _das_term_ids(train_payload, path=train_path, expected_terms=train.shape[0])
    term_indices = _das_score_term_indices(term_ids)

    query_rows: list[np.ndarray] = []
    query_paths: list[Path] = []
    query_meta: list[tuple[int, str, int]] = []
    checkpoint = result_root / "model" / "prompted_jax" / f"seed_{args.train_seed}_epoch_0200.ckpt"
    query_namespace = args.query_artifact_namespace.strip("_/")
    query_suffix = "query_gradient" if not query_namespace else f"query_gradient_{query_namespace}"
    das_query_name = "das" if not query_namespace else f"das_{query_namespace}"
    for query_id in query_ids:
        record = records[query_id]
        prompt = str(record["prompt"])
        seed = int(record["initial_seed"])
        prompt_tag = _prompt_tag(prompt)
        query_path = (
            result_root / "sample_ddim_eta0_1000" / "cifar" / f"prompt_{prompt_tag}"
            / f"model_prompted_solo__ckpt_{checkpoint.stem}"
            / f"seed_{seed:06d}_{query_suffix}" / das_query_name / "query_gradient_artifact.npz"
        )
        payload = load_npz(
            query_path,
            keys=(
                "query_features", "query_feature", "query_gradient", "query_gradients",
                "ckpt_indices", "timesteps", "mc_indices",
            ),
        )
        query = np.asarray(
            _first_array(payload, ("query_features", "query_feature", "query_gradient", "query_gradients"), path=query_path),
            dtype=np.float32,
        )
        if query.ndim == 1:
            query = query[None, :]
        if query.shape != (train.shape[0], train.shape[2]):
            raise ValueError(f"query {query_id} shape {query.shape} does not match train {train.shape}")
        query_term_ids = _das_term_ids(payload, path=query_path, expected_terms=query.shape[0])
        _require_matching_das_terms(term_ids, query_term_ids, path=query_path)
        query_rows.append(query)
        query_paths.append(query_path)
        query_meta.append((query_id, prompt_tag, seed))
    queries = np.stack(query_rows, axis=0)
    print(
        f"[ready] train={train.shape} queries={queries.shape} terms={len(term_indices)} "
        f"lambdas={len(lambdas)} device={args.device}",
        flush=True,
    )

    for group_count in group_counts:
        if train.shape[1] % group_count != 0:
            raise ValueError(f"{train.shape[1]} points are not divisible by {group_count} groups")
        namespace = f"{args.score_namespace_prefix}_g{group_count}_seed{args.partition_seed}"
        output_roots = []
        for query_id, prompt_tag, seed in query_meta:
            output_roots.append(
                result_root / "attribution_score" / "prompted_solo" / f"train_seed_{args.train_seed}"
                / f"query_{prompt_tag}" / f"initial_seed_{seed}" / f"das_{namespace}" / "score"
            )
        expected = [root / f"lambda_{damping_tag(value)}" / "scores.npy" for root in output_roots for value in lambdas]
        if expected and all(path.is_file() for path in expected) and not args.overwrite:
            print(f"[skip] complete mixed local-Gram scores exist: groups={group_count}", flush=True)
            continue

        rng = np.random.default_rng(args.partition_seed)
        permutation = rng.permutation(train.shape[1])
        groups = np.split(permutation, group_count)
        group_size = len(groups[0])
        scores = np.zeros((len(lambdas), len(query_ids), train.shape[1]), dtype=np.float64)
        started = time.time()
        for term_number, term_i in enumerate(term_indices, start=1):
            term_i = int(term_i)
            term_queries = queries[:, term_i, :]
            for group_i, positions in enumerate(groups):
                local_scores = local_group_term_scores(
                    train[term_i, positions],
                    residuals[term_i, positions],
                    term_queries,
                    lambdas,
                    device=args.device,
                    denominator_floor=args.denominator_floor,
                )
                scores[:, :, positions] += local_scores
                print(
                    f"[score] groups={group_count} size={group_size} term={term_number}/{len(term_indices)} "
                    f"group={group_i + 1}/{group_count}",
                    flush=True,
                )
        scores /= float(len(term_indices))

        diagnostics = []
        for group_i, positions in enumerate(groups):
            group_values = scores[:, :, positions]
            diagnostics.append(
                {
                    "group": group_i,
                    "size": len(positions),
                    "dataset_indices_min": int(indices[positions].min()),
                    "dataset_indices_max": int(indices[positions].max()),
                    "score_mean_by_lambda_query": np.mean(group_values, axis=2).tolist(),
                    "score_std_by_lambda_query": np.std(group_values, axis=2).tolist(),
                    "score_median_abs_by_lambda_query": np.median(np.abs(group_values), axis=2).tolist(),
                    "score_l2_by_lambda_query": np.linalg.norm(group_values, axis=2).tolist(),
                }
            )
        diagnostic_root = result_root / "eval" / "das_local_gram_mixed" / namespace
        diagnostic_root.mkdir(parents=True, exist_ok=True)
        (diagnostic_root / "partition.json").write_text(
            json.dumps(
                {
                    "group_count": group_count,
                    "group_size": group_size,
                    "partition_seed": args.partition_seed,
                    "score_indices_by_group": [[int(indices[pos]) for pos in group] for group in groups],
                },
                indent=2,
            )
        )
        (diagnostic_root / "score_scale_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

        for query_i, ((query_id, _prompt_tag_value, _seed), output_root) in enumerate(zip(query_meta, output_roots)):
            for lambda_i, damping in enumerate(lambdas):
                _write_score_outputs(
                    output_root / f"lambda_{damping_tag(damping)}",
                    scores[lambda_i, query_i],
                    indices,
                    train_dir=train_path.parent,
                    query_dir=query_paths[query_i].parent,
                    algorithm="das",
                    extra_manifest={
                        "damping": float(damping),
                        "score_contraction": "squared",
                        "sherman_morrison_denominator": True,
                        "local_gram_mixed": True,
                        "group_count": group_count,
                        "group_size": group_size,
                        "partition_seed": args.partition_seed,
                        "query_id": query_id,
                    },
                )
        print(
            f"[done] groups={group_count} size={group_size} elapsed={time.time() - started:.1f}s "
            f"namespace={namespace}",
            flush=True,
        )


if __name__ == "__main__":
    main()
