#!/usr/bin/env python3
"""Score squared predicted-noise directional changes from saved Traj TracIn gradients."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "predicted_noise_jvp_l2_squared"

import sys

if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from dataset_config import ATTRIBUTION_INDICES_PATH, _prompt_tag


def records() -> list[dict]:
    return json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]


def result_root(experiment: str) -> Path:
    return SHAPES_ROOT / "result" / experiment


def checkpoint_path(experiment: str, train_seed: int, epochs: int) -> Path:
    return result_root(experiment) / "model" / "prompted_jax" / f"seed_{train_seed}_epoch_{epochs:04d}.ckpt"


def probe_namespace(num_probes: int, probe_index: int) -> str:
    if num_probes == 1:
        return NAMESPACE
    return f"{NAMESPACE}_probe{num_probes}_r{probe_index}"


def score_namespace(num_probes: int) -> str:
    suffix = "" if num_probes == 1 else f"_probe{num_probes}"
    return f"traj_tracin_{NAMESPACE}{suffix}"


def query_artifact_path(
    experiment: str,
    train_seed: int,
    epochs: int,
    query_id: int,
    *,
    num_probes: int,
    probe_index: int,
) -> Path:
    record = records()[query_id]
    checkpoint = checkpoint_path(experiment, train_seed, epochs)
    run_root = (
        result_root(experiment)
        / "sample_ddim_eta0_1000"
        / "cifar"
        / f"prompt_{_prompt_tag(str(record['prompt']))}"
        / f"model_prompted_solo__ckpt_{checkpoint.stem}"
    )
    seed = int(record["initial_seed"])
    return (
        run_root
        / f"seed_{seed:06d}_query_gradient_{probe_namespace(num_probes, probe_index)}"
        / "traj_tracin"
        / "query_gradient_artifact.npz"
    )


def train_part_dir(experiment: str, train_seed: int) -> Path:
    artifact = (
        result_root(experiment)
        / "model"
        / "prompted_solo"
        / f"seed_{train_seed}_train_gradient"
        / "traj_tracin"
        / "train_datapoint_gradient_artifact.npz"
    )
    return Path(str(artifact) + ".parts")


def shard_root(experiment: str, train_seed: int, run_id: str, num_probes: int) -> Path:
    return (
        result_root(experiment)
        / "stream_score"
        / score_namespace(num_probes)
        / f"train_seed_{train_seed}"
        / f"run_{run_id}"
    )


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def load_query_bank(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    probe_banks = []
    reference = None
    for probe_index in range(args.num_probes):
        features = []
        for query_id in range(10):
            path = query_artifact_path(
                args.experiment,
                args.train_seed,
                args.epochs,
                query_id,
                num_probes=args.num_probes,
                probe_index=probe_index,
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as payload:
                feature = np.asarray(payload["query_features"], dtype=np.float32)
                objective = str(np.asarray(payload["query_objective"]).item())
                proj_dim = int(np.asarray(payload["proj_dim"]).item())
                stored_probe_index = int(np.asarray(payload.get("output_probe_index", 0)).item())
                metadata = {
                    "ckpt_indices": np.asarray(payload["ckpt_indices"], dtype=np.int32),
                    "timesteps": np.asarray(payload["timesteps"], dtype=np.int32),
                    "term_weights": np.asarray(payload["term_weights"], dtype=np.float64),
                }
            if objective != "trajectory_predicted_noise_probe":
                raise ValueError(f"{path} contains query objective {objective!r}")
            if stored_probe_index != probe_index:
                raise ValueError(
                    f"{path} contains output probe {stored_probe_index}, expected {probe_index}"
                )
            if feature.shape != (500, 4096) or proj_dim != 4096:
                raise ValueError(f"{path} expected query features (500,4096), got {feature.shape}")
            features.append(feature)
            if reference is None:
                reference = metadata
            else:
                for key, value in metadata.items():
                    if not np.allclose(value, reference[key], rtol=1e-6, atol=1e-12):
                        raise ValueError(
                            f"probe {probe_index} query {query_id} metadata mismatch for {key}"
                        )
        probe_banks.append(np.stack(features, axis=0))
    assert reference is not None
    return np.stack(probe_banks, axis=0), reference


def score_shard(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp

    output = (
        shard_root(args.experiment, args.train_seed, args.run_id, args.num_probes)
        / "shards"
        / f"shard_{args.shard_index:02d}.npz"
    )
    if output.is_file():
        print(f"[skip] score shard exists: {output}", flush=True)
        return
    query, query_meta = load_query_bank(args)
    lookup = {
        (int(ckpt), int(timestep)): term_id
        for term_id, (ckpt, timestep) in enumerate(zip(query_meta["ckpt_indices"], query_meta["timesteps"]))
    }
    sums = {
        "score": np.zeros((10, 5000), dtype=np.float64),
        "score_query_normalized": np.zeros((10, 5000), dtype=np.float64),
        "score_train_l2_normalized": np.zeros((10, 5000), dtype=np.float64),
        "score_query_train_l2_normalized": np.zeros((10, 5000), dtype=np.float64),
    }
    score_indices = None
    used_terms = 0

    for ckpt_i in range(args.shard_index, 50, args.shard_count):
        part_path = train_part_dir(args.experiment, args.train_seed) / f"ckpt_{ckpt_i:04d}.npz"
        if not part_path.is_file():
            raise FileNotFoundError(part_path)
        with np.load(part_path, allow_pickle=False) as payload:
            train = np.asarray(payload["train_features"], dtype=np.float32)
            indices = np.asarray(payload["score_indices"], dtype=np.int64)
            ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            weights = np.asarray(payload["term_weights"], dtype=np.float64)
        if train.shape != (10, 5000, 4096):
            raise ValueError(f"{part_path} expected (10,5000,4096), got {train.shape}")
        if score_indices is None:
            score_indices = indices
        elif not np.array_equal(score_indices, indices):
            raise ValueError(f"score indices differ in {part_path}")

        for local_term, (ckpt, timestep, weight) in enumerate(zip(ckpts, timesteps, weights)):
            query_term = lookup.get((int(ckpt), int(timestep)))
            if query_term is None:
                raise ValueError(f"no query feature for checkpoint={ckpt} timestep={timestep}")
            train_device = jax.device_put(jnp.asarray(train[local_term]))
            train_norm = jnp.linalg.norm(train_device, axis=1) + 1e-8
            term_scores = {
                component: np.zeros((10, 5000), dtype=np.float64)
                for component in sums
            }
            for probe_index in range(args.num_probes):
                query_device = jax.device_put(
                    jnp.asarray(query[probe_index, :, query_term, :])
                )
                directional = train_device @ query_device.T
                query_norm = jnp.linalg.norm(query_device, axis=1) + 1e-8
                probe_scores = {
                    "score": jnp.square(directional),
                    "score_query_normalized": jnp.square(
                        directional / query_norm[None, :]
                    ),
                    "score_train_l2_normalized": jnp.square(
                        directional / train_norm[:, None]
                    ),
                    "score_query_train_l2_normalized": jnp.square(
                        directional / train_norm[:, None] / query_norm[None, :]
                    ),
                }
                for component, values in probe_scores.items():
                    term_scores[component] += np.asarray(
                        jax.device_get(values), dtype=np.float64
                    ).T / float(args.num_probes)
            # Only the gradient-produced directional derivative is squared.
            # The TrajTracIn learning-rate weight remains linear, and the
            # stored per-snapshot weight already averages the 10 snapshots.
            term_weight = float(weight)
            for component, values in term_scores.items():
                sums[component] += term_weight * values
            used_terms += 1
        print(f"[score shard {args.shard_index}/{args.shard_count}] checkpoint={ckpt_i} terms={used_terms}", flush=True)

    if score_indices is None:
        raise RuntimeError("checkpoint shard selected no train parts")
    atomic_savez(
        output,
        **{f"sums_{component}": values for component, values in sums.items()},
        score_indices=score_indices,
        used_terms=np.asarray(used_terms, dtype=np.int32),
        weighting_semantics=np.asarray(
            "learning_rate_weighted_sum_of_squared_gradient_contractions"
        ),
    )
    print(f"[saved] {output}", flush=True)


def atomic_save(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, value)
    tmp.replace(path)


def merge(args: argparse.Namespace) -> None:
    totals = {
        "score": np.zeros((10, 5000), dtype=np.float64),
        "score_query_normalized": np.zeros((10, 5000), dtype=np.float64),
        "score_train_l2_normalized": np.zeros((10, 5000), dtype=np.float64),
        "score_query_train_l2_normalized": np.zeros((10, 5000), dtype=np.float64),
    }
    terms = 0
    score_indices = None
    for shard in range(args.shard_count):
        path = (
            shard_root(args.experiment, args.train_seed, args.run_id, args.num_probes)
            / "shards"
            / f"shard_{shard:02d}.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            for component in totals:
                totals[component] += np.asarray(
                    payload[f"sums_{component}"], dtype=np.float64
                )
            semantics = str(np.asarray(payload.get("weighting_semantics", "")).item())
            if semantics != "learning_rate_weighted_sum_of_squared_gradient_contractions":
                raise ValueError(
                    f"{path} has incompatible weighting semantics {semantics!r}; "
                    "recompute this score shard with linear learning-rate weights"
                )
            terms += int(payload["used_terms"])
            indices = np.asarray(payload["score_indices"], dtype=np.int64)
        if score_indices is None:
            score_indices = indices
        elif not np.array_equal(score_indices, indices):
            raise ValueError(f"score indices differ in {path}")
    if terms != 500:
        raise ValueError(f"expected 500 terms, got terms={terms}")
    expected = np.asarray(np.load(ATTRIBUTION_INDICES_PATH), dtype=np.int64)
    if score_indices is None or not np.array_equal(np.sort(score_indices), np.sort(expected)):
        raise ValueError("score indices do not match attribution_5k_indices.npy")

    scores = totals
    for component, values in scores.items():
        for query_id, record in enumerate(records()):
            out_dir = (
                result_root(args.experiment)
                / "attribution_score"
                / "prompted_solo"
                / f"train_seed_{args.train_seed}"
                / f"query_{_prompt_tag(str(record['prompt']))}"
                / f"initial_seed_{int(record['initial_seed'])}"
                / score_namespace(args.num_probes)
                / component
            )
            atomic_save(out_dir / "scores.npy", values[query_id])
            atomic_save(out_dir / "score_indices.npy", score_indices)
            manifest = {
                "algorithm": score_namespace(args.num_probes),
                "score_variant": component,
                "definition": "learning_rate_weighted_sum_terms(squared_normalized_dot_product)",
                "normalization_variants": [
                    "raw",
                    "query_l2",
                    "train_l2",
                    "query_train_l2",
                ],
                "num_output_probes_per_term": args.num_probes,
                "num_checkpoints": 50,
                "timestamps_per_checkpoint": 10,
                "train_mc_samples_per_timestamp": 10,
                "num_terms": 500,
                "projection_dim": 4096,
                "query_gradient_retained": False,
                "transient_run_id": args.run_id,
            }
            (out_dir / "score_artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    if args.cleanup_query_artifacts:
        for query_id in range(10):
            for probe_index in range(args.num_probes):
                path = query_artifact_path(
                    args.experiment,
                    args.train_seed,
                    args.epochs,
                    query_id,
                    num_probes=args.num_probes,
                    probe_index=probe_index,
                )
                if path.is_file():
                    path.unlink()
                    print(f"[cleanup] removed transient query gradient: {path}", flush=True)
    print(f"[done] materialized four normalized score variants for 10 queries; terms={terms}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("score-shard", "merge"))
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=16)
    parser.add_argument("--run-id", default=os.environ.get("PRED_NOISE_JVP_RUN_ID", "manual"))
    parser.add_argument("--num-probes", type=int, default=1)
    parser.add_argument("--cleanup-query-artifacts", action="store_true")
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    if args.num_probes <= 0:
        raise ValueError("--num-probes must be positive")
    if not args.run_id or "/" in args.run_id:
        raise ValueError("--run-id must be a non-empty path component")
    if args.command == "score-shard":
        score_shard(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
