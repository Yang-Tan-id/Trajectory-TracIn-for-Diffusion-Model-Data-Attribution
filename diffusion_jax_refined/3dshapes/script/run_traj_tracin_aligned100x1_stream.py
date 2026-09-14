#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
NAMESPACE = "aligned100x1_stream"
SCORE_NAMESPACE = f"traj_tracin_{NAMESPACE}"
SAVED_NAMESPACE = "aligned100x1_saved"
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


def saved_train_artifact_path(experiment: str, train_seed: int) -> Path:
    return (
        result_root(experiment)
        / "model"
        / "prompted_solo"
        / f"seed_{train_seed}_train_gradient"
        / f"traj_tracin_{SAVED_NAMESPACE}"
        / "train_datapoint_gradient_artifact.npz"
    )


def saved_score_root(experiment: str, train_seed: int) -> Path:
    return (
        result_root(experiment)
        / "saved_gradient_score"
        / f"traj_tracin_{SAVED_NAMESPACE}"
        / f"train_seed_{train_seed}"
    )


def run_train_shard(args: argparse.Namespace) -> None:
    shard_index = int(args.shard_index)
    shard_count = int(args.shard_count)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard index {shard_index} is outside 0-{shard_count - 1}")
    artifact = saved_train_artifact_path(args.experiment, args.train_seed)
    part_dir = Path(str(artifact) + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        EXPERIMENT_TAG=args.experiment,
        TRAIN_SEED=str(args.train_seed),
        JAX_EPOCHS=str(args.epochs),
        QUERY=str(records()[0]["prompt"]),
        INITIAL_SEED="0",
        SAMPLE_SEED="0",
        DATAPOINT_MODEL_MODE="prompted_solo",
        SAMPLE_MODEL_MODE="prompted_solo",
        ATTRIBUTION_SCORE_MODEL_MODE="prompted_solo",
        TRAJ_QUERY_OBJECTIVE="trajectory_next_checkpoint_noise_mse",
        TRAJ_PARAMETER_SOURCE="raw",
        TRAJ_NUM_SNAPSHOTS="100",
        TRAJ_TRAIN_MC_SAMPLES="1",
        TRAJ_TRACIN_PROJ_DIM="4096",
        TRAJ_SCORE_BATCH_SIZE=str(args.batch_size),
        TRAJ_USE_SAVED_TRAJECTORY="0",
        TRAJ_TRACIN_STAGE_MODE="train",
        TRAJ_TRACIN_STAGE_ARTIFACT_PATH=str(artifact),
        TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH=str(artifact),
        TRAJ_TRACIN_CKPT_SHARD_INDEX=str(shard_index),
        TRAJ_TRACIN_CKPT_SHARD_COUNT=str(shard_count),
        TRAJ_TRACIN_SKIP_STAGE_MERGE="1",
        TRAJ_TRACIN_TRAIN_EXCLUDE_FINAL_CHECKPOINT="1",
        TRAJ_TRACIN_TRAIN_AGGREGATE_TIMESTAMPS="0",
        TRAJ_TRACIN_TRAIN_BATCH_MODE="vmap",
        TRAJ_TRACIN_TRAIN_BATCH_DTYPE="float32",
        JAX_NUM_DEVICES="1",
        JAX_PLATFORMS="cuda",
        PYTHONUNBUFFERED="1",
        TF_GPU_ALLOCATOR=os.environ.get("TF_GPU_ALLOCATOR", "cuda_malloc_async"),
    )
    command = [
        args.python_bin,
        str(SHAPES_ROOT / "data_attribution" / "traj_tracin" / "01_train_datapoint_gradient.py"),
    ]
    print(
        f"[saved train shard {shard_index + 1}/{shard_count}] "
        f"100 timestamps x 1 MC; parts={part_dir}",
        flush=True,
    )
    subprocess.run(command, cwd=SHAPES_ROOT, env=env, check=True)


def validate_saved_train_parts(args: argparse.Namespace) -> None:
    artifact = saved_train_artifact_path(args.experiment, args.train_seed)
    part_dir = Path(str(artifact) + ".parts")
    paths = [part_dir / f"ckpt_{index:04d}.npz" for index in range(49)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)}/49 checkpoint parts; first={missing[:3]}")
    total_bytes = sum(path.stat().st_size for path in paths)
    for index, path in enumerate(paths):
        with zipfile.ZipFile(path) as archive:
            with archive.open("train_features.npy") as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(member)
                else:
                    shape, _, _ = np.lib.format.read_array_header_2_0(member)
        with np.load(path, allow_pickle=False) as payload:
            ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        if tuple(shape) != (100, 5000, 4096):
            raise ValueError(f"{path} expected train_features (100,5000,4096), got {shape}")
        if not np.all(ckpts == index) or len(np.unique(timesteps)) != 100:
            raise ValueError(f"{path} has invalid checkpoint/timestep metadata")
    manifest = {
        "namespace": f"traj_tracin_{SAVED_NAMESPACE}",
        "storage": "checkpoint_parts_without_merged_duplicate",
        "num_checkpoints": 49,
        "timestamps_per_checkpoint": 100,
        "mc_per_timestamp": 1,
        "num_terms": 4900,
        "num_points": 5000,
        "projection_dim": 4096,
        "total_part_bytes": total_bytes,
        "total_part_gib": total_bytes / 2**30,
        "part_dir": str(part_dir),
    }
    manifest_path = part_dir / "saved_gradient_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(
        f"[saved train validate] 49/49 parts complete | size={total_bytes / 2**30:.2f} GiB | "
        f"manifest={manifest_path}",
        flush=True,
    )


def load_query_bank_arrays(args: argparse.Namespace) -> tuple[list[Path], np.ndarray, dict[str, np.ndarray]]:
    paths = [
        query_artifact_path(args.experiment, args.train_seed, args.epochs, query_id)
        for query_id in range(10)
    ]
    features = []
    reference = None
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            features.append(np.asarray(payload["query_features"], dtype=np.float32))
            metadata = {
                "ckpt_indices": np.asarray(payload["ckpt_indices"], dtype=np.int32),
                "timesteps": np.asarray(payload["timesteps"], dtype=np.int32),
                "term_weights": np.asarray(payload["term_weights"], dtype=np.float32),
            }
        if reference is None:
            reference = metadata
        else:
            for key in metadata:
                if not np.allclose(metadata[key], reference[key], rtol=1e-6, atol=1e-12):
                    raise ValueError(f"query artifact metadata mismatch for {key}: {path}")
    assert reference is not None
    return paths, np.stack(features, axis=0), reference


def run_saved_score_shard(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp

    shard_index = int(args.shard_index)
    shard_count = int(args.shard_count)
    output = saved_score_root(args.experiment, args.train_seed) / "shards" / f"shard_{shard_index:02d}.npz"
    if output.is_file():
        print(f"[skip] saved-gradient score shard exists: {output}", flush=True)
        return
    artifact = saved_train_artifact_path(args.experiment, args.train_seed)
    part_dir = Path(str(artifact) + ".parts")
    query_paths, query_features, query_meta = load_query_bank_arrays(args)
    query_lookup = {
        (int(ckpt), int(timestep)): term_i
        for term_i, (ckpt, timestep) in enumerate(
            zip(query_meta["ckpt_indices"], query_meta["timesteps"])
        )
    }
    checkpoint_ids = list(range(shard_index, 49, shard_count))
    score_indices = None
    sums = {key: np.zeros((1, 10, 5000), dtype=np.float64) for key, _, _ in VARIANTS}
    shard_ckpts = []
    shard_timesteps = []
    shard_weights = []

    for local_i, ckpt_i in enumerate(checkpoint_ids, start=1):
        part_path = part_dir / f"ckpt_{ckpt_i:04d}.npz"
        print(
            f"[saved score shard {shard_index}/{shard_count}] loading checkpoint "
            f"{local_i}/{len(checkpoint_ids)}: {part_path}",
            flush=True,
        )
        with np.load(part_path, allow_pickle=False) as payload:
            train = np.asarray(payload["train_features"], dtype=np.float32)
            indices = np.asarray(payload["score_indices"], dtype=np.int64)
            ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            weights = np.asarray(payload["term_weights"], dtype=np.float64)
        if train.shape != (100, 5000, 4096):
            raise ValueError(f"unexpected train feature shape in {part_path}: {train.shape}")
        if score_indices is None:
            score_indices = indices
        elif not np.array_equal(score_indices, indices):
            raise ValueError(f"score_indices mismatch in {part_path}")

        for term_i, (ckpt, timestep, weight) in enumerate(zip(ckpts, timesteps, weights)):
            query_i = query_lookup.get((int(ckpt), int(timestep)))
            if query_i is None:
                raise ValueError(f"query bank missing term ckpt={ckpt} timestep={timestep}")
            query_weight = float(query_meta["term_weights"][query_i])
            if not np.isclose(float(weight), query_weight, rtol=1e-5, atol=1e-12):
                raise ValueError(f"train/query weight mismatch at ckpt={ckpt} timestep={timestep}")
            train_device = jax.device_put(jnp.asarray(train[term_i]))
            query_device = jax.device_put(jnp.asarray(query_features[:, query_i, :]))
            train_norm = jnp.maximum(jnp.linalg.norm(train_device, axis=1, keepdims=True), 1e-8)
            query_norm = jnp.maximum(jnp.linalg.norm(query_device, axis=1, keepdims=True), 1e-8)
            raw = train_device @ query_device.T
            query_l2 = train_device @ (query_device / query_norm).T
            train_l2 = (train_device / train_norm) @ query_device.T
            both_l2 = (train_device / train_norm) @ (query_device / query_norm).T
            both_l2.block_until_ready()
            for key, value in (
                ("scores_raw", raw),
                ("scores_query_l2_normalized", query_l2),
                ("scores_train_l2_normalized", train_l2),
                ("scores_query_train_l2_normalized", both_l2),
            ):
                sums[key][0] += float(weight) * np.asarray(value, dtype=np.float64).T
            shard_ckpts.append(int(ckpt))
            shard_timesteps.append(int(timestep))
            shard_weights.append(float(weight))
        del train
        print(
            f"[saved score shard {shard_index}/{shard_count}] checkpoint {ckpt_i + 1}/49 done",
            flush=True,
        )

    if score_indices is None:
        raise RuntimeError(f"score shard {shard_index}/{shard_count} owned no checkpoints")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        **sums,
        score_indices=score_indices,
        query_artifacts=np.asarray([str(path) for path in query_paths]),
        proj_dims=np.asarray([4096], dtype=np.int32),
        cache_dim=np.asarray(4096, dtype=np.int32),
        term_ckpt_indices=np.asarray(shard_ckpts, dtype=np.int32),
        term_timesteps=np.asarray(shard_timesteps, dtype=np.int32),
        term_weights=np.asarray(shard_weights, dtype=np.float32),
    )
    tmp.replace(output)
    print(f"[saved score shard] wrote {output}", flush=True)


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


def materialize_scores(
    args: argparse.Namespace,
    merged_path: Path,
    *,
    score_namespace: str = SCORE_NAMESPACE,
    train_gradients_saved: bool = False,
) -> None:
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
                / score_namespace
                / component
            )
            atomic_save_npy(out_dir / "scores.npy", values[0, query_id])
            atomic_save_npy(out_dir / "score_indices.npy", score_indices)
            manifest = {
                "algorithm": "traj_tracin",
                "score_variant": variant,
                "streaming_train_gradient": not train_gradients_saved,
                "train_gradient_artifact_saved": train_gradients_saved,
                "timestamp_alignment": "exact_ckpt_and_timestep",
                "num_checkpoints": 49,
                "timestamps_per_checkpoint": 100,
                "train_mc_samples_per_timestamp": 1,
                "num_terms": int(len(term_timesteps)),
                "num_queries": 10,
                "num_scores": int(len(score_indices)),
                "projection_dim": 4096,
                "term_weight_sum": float(term_weights.sum()),
                "merged_score_artifact": str(merged_path),
            }
            (out_dir / "score_artifact_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True)
            )
    print(
        f"[materialize] wrote 4 variants x 10 queries under namespace {score_namespace}",
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


def merge_saved_score_shards(args: argparse.Namespace) -> None:
    root = saved_score_root(args.experiment, args.train_seed)
    paths = [root / "shards" / f"shard_{index:02d}.npz" for index in range(args.shard_count)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing saved-gradient score shards: {missing[:3]}")
    arrays = {key: None for key, _, _ in VARIANTS}
    score_indices = None
    query_artifacts = None
    ckpts = []
    timesteps = []
    weights = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            indices = np.asarray(payload["score_indices"], dtype=np.int64)
            queries = np.asarray(payload["query_artifacts"])
            if score_indices is None:
                score_indices = indices
                query_artifacts = queries
            elif not np.array_equal(score_indices, indices) or not np.array_equal(query_artifacts, queries):
                raise ValueError(f"saved-gradient score shard metadata mismatch: {path}")
            for key in arrays:
                value = np.asarray(payload[key], dtype=np.float64)
                arrays[key] = value if arrays[key] is None else arrays[key] + value
            ckpts.append(np.asarray(payload["term_ckpt_indices"], dtype=np.int32))
            timesteps.append(np.asarray(payload["term_timesteps"], dtype=np.int32))
            weights.append(np.asarray(payload["term_weights"], dtype=np.float32))
    assert score_indices is not None and query_artifacts is not None
    ckpts_all = np.concatenate(ckpts)
    timesteps_all = np.concatenate(timesteps)
    weights_all = np.concatenate(weights)
    order = np.lexsort((timesteps_all, ckpts_all))
    merged = root / "saved_gradient_scores_merged.npz"
    tmp = merged.with_name(merged.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        **arrays,
        score_indices=score_indices,
        query_artifacts=query_artifacts,
        proj_dims=np.asarray([4096], dtype=np.int32),
        cache_dim=np.asarray(4096, dtype=np.int32),
        term_ckpt_indices=ckpts_all[order],
        term_timesteps=timesteps_all[order],
        term_weights=weights_all[order],
    )
    tmp.replace(merged)
    materialize_scores(
        args,
        merged,
        score_namespace=f"traj_tracin_{SAVED_NAMESPACE}",
        train_gradients_saved=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D Shapes aligned 100x1 query, streaming, or saved-gradient Traj TracIn."
    )
    parser.add_argument(
        "phase",
        choices=("query", "train", "validate-train", "score-saved", "merge-saved", "stream", "merge"),
    )
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
    elif args.phase == "train":
        run_train_shard(args)
    elif args.phase == "validate-train":
        validate_saved_train_parts(args)
    elif args.phase == "score-saved":
        run_saved_score_shard(args)
    elif args.phase == "merge-saved":
        merge_saved_score_shards(args)
    elif args.phase == "stream":
        run_stream_shard(args)
    else:
        merge_shards(args)


if __name__ == "__main__":
    main()
