#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
NAMESPACE = "aligned100x1_stream"
SCORE_NAMESPACE = f"traj_tracin_{NAMESPACE}"
VARIANTS = (
    ("scores_raw", "score", "raw"),
    ("scores_query_l2_normalized", "score_query_normalized", "query_l2_normalized"),
    ("scores_train_l2_normalized", "score_train_l2_normalized", "train_l2_normalized"),
    (
        "scores_query_train_l2_normalized",
        "score_query_train_l2_normalized",
        "query_train_l2_normalized",
    ),
)

if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from dataset_config import ATTRIBUTION_INDICES_PATH, _prompt_tag


def records() -> list[dict]:
    return json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]


def result_root(experiment: str) -> Path:
    return SHAPES_ROOT / "result" / experiment


def checkpoint_path(experiment: str, train_seed: int, epochs: int) -> Path:
    return (
        result_root(experiment)
        / "model"
        / "prompted_jax"
        / f"seed_{train_seed}_epoch_{epochs:04d}.ckpt"
    )


def sample_run_root(experiment: str, prompt: str, checkpoint: Path) -> Path:
    return (
        result_root(experiment)
        / "sample_ddim_eta0_1000"
        / "cifar"
        / f"prompt_{_prompt_tag(prompt)}"
        / f"model_prompted_solo__ckpt_{checkpoint.stem}"
    )


def query_artifact_path(experiment: str, train_seed: int, epochs: int, query_id: int) -> Path:
    record = records()[query_id]
    seed = int(record["initial_seed"])
    run_root = sample_run_root(
        experiment,
        str(record["prompt"]),
        checkpoint_path(experiment, train_seed, epochs),
    )
    return (
        run_root
        / f"seed_{seed:06d}_query_gradient_{NAMESPACE}"
        / "traj_tracin"
        / "query_gradient_artifact.npz"
    )


def stream_root(experiment: str, train_seed: int) -> Path:
    return (
        result_root(experiment)
        / "stream_score"
        / SCORE_NAMESPACE
        / f"train_seed_{train_seed}"
    )


def run_query(args: argparse.Namespace) -> None:
    query_id = int(args.query_id)
    if query_id < 0 or query_id >= len(records()):
        raise ValueError(f"query id {query_id} is outside 0-{len(records()) - 1}")
    artifact = query_artifact_path(args.experiment, args.train_seed, args.epochs, query_id)
    if artifact.is_file():
        print(f"[skip] query {query_id} artifact exists: {artifact}", flush=True)
        return
    command = [
        args.python_bin,
        str(SHAPES_ROOT / "script" / "run_traj_tracin_queries_and_scores.py"),
        "--execute",
        "--experiment",
        args.experiment,
        "--train-seed",
        str(args.train_seed),
        "--epochs",
        str(args.epochs),
        "--query-ids",
        str(query_id),
        "--gpus",
        str(args.gpu),
        "--artifact-namespace",
        NAMESPACE,
        "--num-snapshots",
        "100",
        "--skip-score",
        "--log-prefix",
        f"query_{query_id}",
        "--python-bin",
        args.python_bin,
    ]
    print(f"[query {query_id}] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=SHAPES_ROOT, check=True)
    if not artifact.is_file():
        raise FileNotFoundError(f"query stage did not produce {artifact}")


def run_stream_shard(args: argparse.Namespace) -> None:
    shard_index = int(args.shard_index)
    shard_count = int(args.shard_count)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard index {shard_index} is outside 0-{shard_count - 1}")
    query_paths = [
        query_artifact_path(args.experiment, args.train_seed, args.epochs, query_id)
        for query_id in range(10)
    ]
    missing = [path for path in query_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing query artifacts: {missing[:3]}")

    output = stream_root(args.experiment, args.train_seed) / "shards" / f"shard_{shard_index:02d}.npz"
    if output.is_file():
        print(f"[skip] stream shard {shard_index}/{shard_count}: {output}", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        EXPERIMENT_TAG=args.experiment,
        TRAIN_SEED=str(args.train_seed),
        JAX_EPOCHS=str(args.epochs),
        QUERY=str(records()[0]["prompt"]),
        INITIAL_SEED="0",
        SAMPLE_SEED="0",
        SAMPLE_MODEL_MODE="prompted_solo",
        ATTRIBUTION_SCORE_MODEL_MODE="prompted_solo",
        TRAJ_QUERY_OBJECTIVE="trajectory_next_checkpoint_noise_mse",
        TRAJ_PARAMETER_SOURCE="raw",
        TRAJ_NUM_SNAPSHOTS="100",
        TRAJ_TRAIN_MC_SAMPLES="1",
        TRAJ_TRACIN_PROJ_DIM="4096",
        TRAJ_SCORE_BATCH_SIZE=str(args.batch_size),
        TRAJ_USE_SAVED_TRAJECTORY="0",
        TRAJ_TRACIN_STAGE_MODE="score_stream",
        TRAJ_TRACIN_STAGE_ARTIFACT_PATH=str(output),
        TRAJ_TRACIN_STREAM_CACHE_DIM="4096",
        TRAJ_TRACIN_STREAM_PROJ_DIMS="4096",
        TRAJ_TRACIN_STREAM_QUERY_ARTIFACTS=os.pathsep.join(str(path) for path in query_paths),
        TRAJ_TRACIN_STREAM_SAVE_TERM_SCORE_VARIANTS="",
        TRAJ_TRACIN_CANDIDATE_SHARD_INDEX=str(shard_index),
        TRAJ_TRACIN_CANDIDATE_SHARD_COUNT=str(shard_count),
        JAX_NUM_DEVICES="1",
        JAX_PLATFORMS="cuda",
        PYTHONUNBUFFERED="1",
        TF_GPU_ALLOCATOR=os.environ.get("TF_GPU_ALLOCATOR", "cuda_malloc_async"),
    )
    command = [
        args.python_bin,
        str(REFINE_ROOT / "common" / "run_original_attribution_config.py"),
        str(SHAPES_ROOT / "data_attribution" / "traj_tracin" / "CONFIG.py"),
    ]
    print(
        f"[stream shard {shard_index + 1}/{shard_count}] batch={args.batch_size} "
        f"queries=10 output={output}",
        flush=True,
    )
    subprocess.run(command, cwd=SHAPES_ROOT, env=env, check=True)
    if not output.is_file():
        raise FileNotFoundError(f"stream stage did not produce {output}")


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, value)
    tmp.replace(path)


def materialize_scores(args: argparse.Namespace, merged_path: Path) -> None:
    with np.load(merged_path, allow_pickle=False) as payload:
        score_indices = np.asarray(payload["score_indices"], dtype=np.int64).reshape(-1)
        proj_dims = np.asarray(payload["proj_dims"], dtype=np.int32).reshape(-1)
        arrays = {key: np.asarray(payload[key], dtype=np.float64) for key, _, _ in VARIANTS}
        term_ckpts = np.asarray(payload["term_ckpt_indices"], dtype=np.int32)
        term_timesteps = np.asarray(payload["term_timesteps"], dtype=np.int32)
        term_weights = np.asarray(payload["term_weights"], dtype=np.float32)

    if proj_dims.tolist() != [4096]:
        raise ValueError(f"expected projection dimension [4096], got {proj_dims.tolist()}")
    if len(score_indices) != 5000 or len(np.unique(score_indices)) != 5000:
        raise ValueError(f"expected 5000 unique score indices, got {score_indices.shape}")
    expected_indices = np.asarray(np.load(ATTRIBUTION_INDICES_PATH), dtype=np.int64).reshape(-1)
    if not np.array_equal(np.sort(score_indices), np.sort(expected_indices)):
        raise ValueError("stream shard indices do not match the fixed attribution_5k subset")
    if len(term_timesteps) != 49 * 100 or len(term_ckpts) != 49 * 100:
        raise ValueError(
            f"expected 4900 aligned query terms, got ckpts={len(term_ckpts)} t={len(term_timesteps)}"
        )

    query_records = records()
    for key, component, variant in VARIANTS:
        values = arrays[key]
        if values.shape != (1, 10, 5000):
            raise ValueError(f"{key} expected shape (1, 10, 5000), got {values.shape}")
        for query_id, record in enumerate(query_records):
            prompt = str(record["prompt"])
            seed = int(record["initial_seed"])
            out_dir = (
                result_root(args.experiment)
                / "attribution_score"
                / "prompted_solo"
                / f"train_seed_{args.train_seed}"
                / f"query_{_prompt_tag(prompt)}"
                / f"initial_seed_{seed}"
                / SCORE_NAMESPACE
                / component
            )
            atomic_save_npy(out_dir / "scores.npy", values[0, query_id])
            atomic_save_npy(out_dir / "score_indices.npy", score_indices)
            manifest = {
                "algorithm": "traj_tracin",
                "score_variant": variant,
                "streaming_train_gradient": True,
                "train_gradient_artifact_saved": False,
                "timestamp_alignment": "exact_ckpt_and_timestep",
                "num_checkpoints": 49,
                "timestamps_per_checkpoint": 100,
                "train_mc_samples_per_timestamp": 1,
                "num_terms": int(len(term_timesteps)),
                "num_queries": 10,
                "num_scores": int(len(score_indices)),
                "projection_dim": 4096,
                "term_weight_sum": float(term_weights.sum()),
                "merged_stream_artifact": str(merged_path),
            }
            (out_dir / "score_artifact_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True)
            )
    print(
        f"[materialize] wrote 4 variants x 10 queries under namespace {SCORE_NAMESPACE}",
        flush=True,
    )


def merge_shards(args: argparse.Namespace) -> None:
    root = stream_root(args.experiment, args.train_seed)
    shard_paths = [root / "shards" / f"shard_{index:02d}.npz" for index in range(args.shard_count)]
    missing = [path for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing stream shards: {missing[:3]}")
    merged = root / "stream_scores_merged.npz"
    command = [
        args.python_bin,
        str(REFINE_ROOT / "common" / "merge_stream_score_shards.py"),
        "--output",
        str(merged),
        *(str(path) for path in shard_paths),
    ]
    subprocess.run(command, cwd=SHAPES_ROOT, check=True)
    materialize_scores(args, merged)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D Shapes aligned 100x1 query-cached streaming Traj TracIn."
    )
    parser.add_argument("phase", choices=("query", "stream", "merge"))
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--query-id", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.phase == "query":
        run_query(args)
    elif args.phase == "stream":
        run_stream_shard(args)
    else:
        merge_shards(args)


if __name__ == "__main__":
    main()
