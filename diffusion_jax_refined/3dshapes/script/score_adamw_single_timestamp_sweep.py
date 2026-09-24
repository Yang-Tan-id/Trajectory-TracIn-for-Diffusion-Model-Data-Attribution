#!/usr/bin/env python3
"""Score every timestamp in two AdamW aligned grids with one load per artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SHAPES_ROOT.parents[1]
for import_root in (SHAPES_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dataset_config import _prompt_tag
from diffusion_jax_refined.common.stage_artifact_runner import (
    _first_array,
    _load_npz,
    _normalize_rows,
    _score_indices,
    _write_score_outputs,
)


VARIANT_DIRS = {
    "raw": "score",
    "query_l2": "score_query_normalized",
    "train_l2": "score_train_l2_normalized",
    "query_train_l2": "score_query_train_l2_normalized",
}


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value.strip()]


def sample_run_root(sample_root: Path, prompt: str, checkpoint: Path) -> Path:
    return (
        sample_root
        / "cifar"
        / f"prompt_{_prompt_tag(prompt)}"
        / f"model_prompted_solo__ckpt_{checkpoint.stem}"
    )


def query_artifact_path(run_root: Path, seed: int, namespace: str) -> Path:
    return (
        run_root
        / f"seed_{seed:06d}_query_gradient_{namespace}"
        / "traj_tracin"
        / "query_gradient_artifact.npz"
    )


def output_root(
    result_root: Path,
    *,
    train_seed: int,
    prompt: str,
    initial_seed: int,
    kind: str,
    timestep: int,
) -> Path:
    return (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
        / f"query_{_prompt_tag(prompt)}"
        / f"initial_seed_{initial_seed}"
        / f"traj_tracin_adamw_{kind}_single_timestamp_t{timestep:03d}"
    )


def outputs_complete(
    result_root: Path,
    *,
    train_seed: int,
    records: list[dict[str, object]],
    query_ids: list[int],
    timestamps: tuple[int, ...],
) -> bool:
    for query_id in query_ids:
        record = records[query_id]
        for timestep in timestamps:
            for kind in ("residual", "full"):
                root = output_root(
                    result_root,
                    train_seed=train_seed,
                    prompt=str(record["prompt"]),
                    initial_seed=int(record["initial_seed"]),
                    kind=kind,
                    timestep=timestep,
                )
                for directory_name in VARIANT_DIRS.values():
                    if not (root / directory_name / "scores.npy").is_file():
                        return False
    return True


def score_artifact(
    artifact_path: Path,
    *,
    result_root: Path,
    checkpoint: Path,
    records: list[dict[str, object]],
    query_ids: list[int],
    query_namespace: str,
    train_seed: int,
    train_eps: float,
) -> list[int]:
    print(f"[load] train artifact: {artifact_path}", flush=True)
    payload = _load_npz(artifact_path)
    train = np.asarray(
        _first_array(
            payload,
            ("train_features", "features", "train_gradients", "gradients"),
            path=artifact_path,
        ),
        dtype=np.float32,
    )
    if train.ndim != 3:
        raise ValueError(f"expected rank-3 train features, got {train.shape}")
    if "optimizer_history_features" not in payload:
        raise ValueError(f"{artifact_path} lacks optimizer_history_features")
    history = np.asarray(payload["optimizer_history_features"], dtype=np.float32)
    if history.shape != (train.shape[0], train.shape[2]):
        raise ValueError(
            f"optimizer history shape {history.shape} does not match "
            f"{(train.shape[0], train.shape[2])}"
        )
    ckpts = np.asarray(payload.get("ckpt_indices", ()), dtype=np.int64).reshape(-1)
    timesteps = np.asarray(payload.get("timesteps", ()), dtype=np.int64).reshape(-1)
    if ckpts.shape != (train.shape[0],) or timesteps.shape != (train.shape[0],):
        raise ValueError("train artifact is missing aligned checkpoint/timestep metadata")
    sample_root = result_root / "sample_ddim_eta0_1000"
    query_rows: list[np.ndarray] = []
    selected_records: list[dict[str, object]] = []
    query_paths: list[Path] = []
    train_term_indices: np.ndarray | None = None
    for query_id in query_ids:
        record = records[query_id]
        prompt = str(record["prompt"])
        initial_seed = int(record["initial_seed"])
        path = query_artifact_path(
            sample_run_root(sample_root, prompt, checkpoint),
            initial_seed,
            query_namespace,
        )
        query_payload = _load_npz(path)
        query = np.asarray(
            _first_array(
                query_payload,
                ("query_features", "query_feature", "query_gradient", "query_gradients"),
                path=path,
            ),
            dtype=np.float32,
        )
        query_ckpts = np.asarray(query_payload.get("ckpt_indices", ()), dtype=np.int64).reshape(-1)
        query_timesteps = np.asarray(query_payload.get("timesteps", ()), dtype=np.int64).reshape(-1)
        if query.ndim != 2 or query.shape[1] != train.shape[2]:
            raise ValueError(f"query feature shape mismatch in {path}: {query.shape}")
        if query_ckpts.shape != (query.shape[0],) or query_timesteps.shape != (query.shape[0],):
            raise ValueError(f"query artifact is missing term metadata: {path}")
        lookup = {
            (int(ckpt), int(timestep)): row
            for row, (ckpt, timestep) in enumerate(zip(query_ckpts, query_timesteps))
        }
        current_train_indices = np.asarray(
            [
                term_i
                for term_i, (ckpt, timestep) in enumerate(zip(ckpts, timesteps))
                if (int(ckpt), int(timestep)) in lookup
            ],
            dtype=np.int64,
        )
        if current_train_indices.size == 0:
            raise ValueError(f"query {query_id} has no terms aligned with {artifact_path}")
        if train_term_indices is None:
            train_term_indices = current_train_indices
        elif not np.array_equal(train_term_indices, current_train_indices):
            raise ValueError(
                f"query {query_id} has a different train/query term intersection"
            )
        aligned = np.asarray(
            [
                query[lookup[(int(ckpts[term_i]), int(timesteps[term_i]))]]
                for term_i in current_train_indices
            ],
            dtype=np.float32,
        )
        query_rows.append(aligned)
        selected_records.append(record)
        query_paths.append(path)
        print(f"[load] query Q{query_id} seed={initial_seed}: {path}", flush=True)

    if train_term_indices is None:
        raise ValueError("no aligned train/query terms")
    aligned_ckpts = ckpts[train_term_indices]
    aligned_timesteps = timesteps[train_term_indices]
    timestamps = sorted(int(value) for value in np.unique(aligned_timesteps))
    for timestep in timestamps:
        term_ckpts = aligned_ckpts[aligned_timesteps == timestep]
        if len(term_ckpts) != len(np.unique(term_ckpts)):
            raise ValueError(f"t={timestep} contains duplicate checkpoint terms")

    queries = np.stack(query_rows, axis=0)
    normalized_queries = _normalize_rows(queries, 1e-8).astype(np.float32, copy=False)
    num_queries = len(query_ids)
    num_points = train.shape[1]
    scores = {
        kind: {
            variant: {
                timestep: np.zeros((num_queries, num_points), dtype=np.float64)
                for timestep in timestamps
            }
            for variant in VARIANT_DIRS
        }
        for kind in ("residual", "full")
    }

    print(
        f"[score] train={train.shape} aligned_terms={len(train_term_indices)} "
        f"queries={queries.shape} timestamps={timestamps}",
        flush=True,
    )
    for aligned_i, train_i in enumerate(train_term_indices):
        timestep = int(timesteps[train_i])
        train_term = np.asarray(train[train_i], dtype=np.float32)
        query_term = queries[:, aligned_i, :]
        normalized_query_term = normalized_queries[:, aligned_i, :]
        for kind in ("residual", "full"):
            effective_train = (
                train_term
                if kind == "residual"
                else train_term + history[train_i][None, :]
            )
            raw_dot = effective_train @ query_term.T
            query_dot = effective_train @ normalized_query_term.T
            train_norm = np.sqrt(
                np.einsum("ij,ij->i", effective_train, effective_train, optimize=True)
            )
            denominator = np.maximum(train_norm, train_eps)[:, None]
            scores[kind]["raw"][timestep] += raw_dot.T
            scores[kind]["query_l2"][timestep] += query_dot.T
            scores[kind]["train_l2"][timestep] += (raw_dot / denominator).T
            scores[kind]["query_train_l2"][timestep] += (query_dot / denominator).T
        if (aligned_i + 1) % 25 == 0 or aligned_i + 1 == len(train_term_indices):
            print(
                f"[score] term {aligned_i + 1}/{len(train_term_indices)}",
                flush=True,
            )

    indices = _score_indices(payload, num_points)
    for query_slot, (query_id, record, query_path) in enumerate(
        zip(query_ids, selected_records, query_paths)
    ):
        prompt = str(record["prompt"])
        initial_seed = int(record["initial_seed"])
        for timestep in timestamps:
            for kind in ("residual", "full"):
                root = output_root(
                    result_root,
                    train_seed=train_seed,
                    prompt=prompt,
                    initial_seed=initial_seed,
                    kind=kind,
                    timestep=timestep,
                )
                for variant, directory_name in VARIANT_DIRS.items():
                    target = root / directory_name
                    if (target / "scores.npy").is_file():
                        continue
                    _write_score_outputs(
                        target,
                        scores[kind][variant][timestep][query_slot],
                        indices,
                        train_dir=artifact_path.parent,
                        query_dir=query_path.parent,
                        algorithm="traj_tracin",
                        extra_manifest={
                            "single_timestamp_sweep": True,
                            "timestep": timestep,
                            "optimizer_feature": kind,
                            "score_variant": variant,
                            "checkpoint_weighting": "uniform_checkpoint",
                            "query_id": query_id,
                        },
                    )
        print(f"[write] Q{query_id} complete", flush=True)
    return timestamps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--query-file",
        type=Path,
        default=SHAPES_ROOT / "queries_in_distribution_plus_zero_seed_100_219.json",
    )
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--original-query-namespace",
        default="loss_direction_original_f_reference_trajectory_100t_indist_first10",
    )
    parser.add_argument(
        "--addon-query-namespace",
        default="loss_direction_original_f_reference_trajectory_addon10_indist_first10",
    )
    parser.add_argument("--train-normalize-eps", type=float, default=1e-8)
    args = parser.parse_args()

    records = json.loads(args.query_file.read_text())["queries"]
    query_ids = parse_ints(args.query_ids)
    if not query_ids:
        parser.error("--query-ids selected no queries")
    if min(query_ids) < 0 or max(query_ids) >= len(records):
        parser.error("a query id is outside the query manifest")

    result_root = SHAPES_ROOT / "result" / args.experiment
    checkpoint = (
        result_root
        / "model"
        / "prompted_jax"
        / f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt"
    )
    train_root = (
        result_root
        / "model"
        / "prompted_solo"
        / f"seed_{args.train_seed}_train_gradient"
    )
    artifact_jobs = (
        (
            train_root
            / "traj_tracin_adamw_dual_aligned10x10"
            / "train_datapoint_gradient_artifact.npz",
            args.original_query_namespace,
            (0, 111, 222, 333, 444, 555, 666, 777, 888, 999),
        ),
        (
            train_root
            / "traj_tracin_adamw_dual_aligned10x10_addon10"
            / "train_datapoint_gradient_artifact.npz",
            args.addon_query_namespace,
            (49, 149, 249, 349, 449, 549, 649, 749, 849, 949),
        ),
    )
    seen: set[int] = set()
    for artifact, query_namespace, expected_timestamps in artifact_jobs:
        if not artifact.is_file():
            raise FileNotFoundError(str(artifact))
        if outputs_complete(
            result_root,
            train_seed=args.train_seed,
            records=records,
            query_ids=query_ids,
            timestamps=expected_timestamps,
        ):
            print(
                f"[skip] complete single-timestamp grid: {list(expected_timestamps)}",
                flush=True,
            )
            seen.update(expected_timestamps)
            continue
        timestamps = score_artifact(
            artifact,
            result_root=result_root,
            checkpoint=checkpoint,
            records=records,
            query_ids=query_ids,
            query_namespace=query_namespace,
            train_seed=args.train_seed,
            train_eps=args.train_normalize_eps,
        )
        overlap = seen.intersection(timestamps)
        if overlap:
            raise ValueError(f"timestamp grids overlap: {sorted(overlap)}")
        if tuple(timestamps) != expected_timestamps:
            raise ValueError(
                f"unexpected aligned timestamps for {artifact}: {timestamps}; "
                f"expected {list(expected_timestamps)}"
            )
        seen.update(timestamps)
    print(f"[done] single-timestamp scores: {sorted(seen)}", flush=True)


if __name__ == "__main__":
    main()
