#!/usr/bin/env python3
"""Score the normalized-expected-Jacobian times expected-residual train artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from dataset_config import ATTRIBUTION_INDICES_PATH, _prompt_tag


TRAIN_NAMESPACE = "traj_tracin_expected_residual_jacobian_probe_reused"
ORIGINAL_QUERY_NAMESPACE = "expected_residual_jacobian_original_f"
PREDICTED_QUERY_NAMESPACE = "expected_residual_jacobian_predicted_noise"
SCORE_NAMESPACES = {
    ("fnorm", "original"): "traj_tracin_expected_residual_jacobian_fnorm_original_f",
    ("v_l2", "original"): "traj_tracin_expected_residual_jacobian_v_l2_original_f",
    ("fnorm", "predicted"): "traj_tracin_expected_residual_jacobian_fnorm_predicted_noise",
    ("v_l2", "predicted"): "traj_tracin_expected_residual_jacobian_v_l2_predicted_noise",
}


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


def query_artifact_path(
    experiment: str,
    train_seed: int,
    epochs: int,
    query_id: int,
    namespace: str,
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
    query_dir_name = f"seed_{seed:06d}_query_gradient"
    if namespace:
        query_dir_name += f"_{namespace}"
    return (
        run_root
        / query_dir_name
        / "traj_tracin"
        / "query_gradient_artifact.npz"
    )


def train_part_dir(experiment: str, train_seed: int) -> Path:
    artifact = (
        result_root(experiment)
        / "model"
        / "prompted_solo"
        / f"seed_{train_seed}_train_gradient"
        / TRAIN_NAMESPACE
        / "train_datapoint_gradient_artifact.npz"
    )
    return Path(str(artifact) + ".parts")


def shard_path(args: argparse.Namespace) -> Path:
    return (
        result_root(args.experiment)
        / "stream_score"
        / TRAIN_NAMESPACE
        / f"train_seed_{args.train_seed}"
        / f"run_{args.run_id}"
        / "shards"
        / f"shard_{args.shard_index:02d}.npz"
    )


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def atomic_save(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, values)
    tmp.replace(path)


def load_query_bank(args: argparse.Namespace, namespace: str, objective: str):
    features = []
    reference = None
    for query_id in range(10):
        path = query_artifact_path(
            args.experiment, args.train_seed, args.epochs, query_id, namespace
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            found_objective = str(np.asarray(data["query_objective"]).item())
            if found_objective != objective:
                raise ValueError(
                    f"{path} objective={found_objective!r}; expected {objective!r}"
                )
            feature = np.asarray(data["query_features"], dtype=np.float32)
            meta = {
                "ckpt_indices": np.asarray(data["ckpt_indices"], dtype=np.int32),
                "timesteps": np.asarray(data["timesteps"], dtype=np.int32),
                "term_weights": np.asarray(data["term_weights"], dtype=np.float64),
            }
        if feature.ndim != 2 or feature.shape[1] != 4096:
            raise ValueError(f"{path} has invalid query feature shape {feature.shape}")
        features.append(feature)
        if reference is None:
            reference = meta
        else:
            for key in meta:
                if not np.allclose(meta[key], reference[key], rtol=1e-6, atol=1e-12):
                    raise ValueError(f"query metadata mismatch for {key}: {path}")
    return np.stack(features, axis=0), reference


def score_shard(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp

    output = shard_path(args)
    if output.is_file():
        print(f"[skip] score shard exists: {output}", flush=True)
        return

    original_query, original_meta = load_query_bank(
        args, ORIGINAL_QUERY_NAMESPACE, "trajectory_next_checkpoint_noise_mse"
    )
    predicted_query, predicted_meta = load_query_bank(
        args, PREDICTED_QUERY_NAMESPACE, "trajectory_predicted_noise_probe"
    )
    query_banks = {
        "original": (original_query, original_meta),
        "predicted": (predicted_query, predicted_meta),
    }
    lookups = {
        name: {
            (int(ckpt), int(timestep)): term_id
            for term_id, (ckpt, timestep) in enumerate(
                zip(meta["ckpt_indices"], meta["timesteps"])
            )
        }
        for name, (_, meta) in query_banks.items()
    }

    sums = {
        (normalization, target, query_variant): np.zeros((10, 5000), dtype=np.float64)
        for normalization in ("fnorm", "v_l2")
        for target in ("original", "predicted")
        for query_variant in ("raw", "query_l2")
    }
    original_weight = 0.0
    predicted_weight = 0.0
    original_terms = 0
    predicted_terms = 0
    score_indices = None
    jacobian_norm_probes = None

    for ckpt_i in range(args.shard_index, 50, args.shard_count):
        path = train_part_dir(args.experiment, args.train_seed) / f"ckpt_{ckpt_i:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            train = np.asarray(data["train_features"], dtype=np.float32)
            train_v_l2 = np.asarray(
                data["train_features_v_l2_normalized"], dtype=np.float32
            )
            indices = np.asarray(data["score_indices"], dtype=np.int64)
            ckpts = np.asarray(data["ckpt_indices"], dtype=np.int32)
            timesteps = np.asarray(data["timesteps"], dtype=np.int32)
            weights = np.asarray(data["term_weights"], dtype=np.float64)
            semantics = str(np.asarray(data["train_feature_semantics"]).item())
            part_probe_count = int(np.asarray(data["jacobian_norm_probes"]).item())
        if semantics != "hutchinson_projected_expected_jacobian_transpose_expected_residual_over_frobenius_norm":
            raise ValueError(f"unexpected train feature semantics in {path}: {semantics}")
        if train.shape != (10, 5000, 4096):
            raise ValueError(f"{path} expected (10,5000,4096), got {train.shape}")
        if train_v_l2.shape != train.shape:
            raise ValueError(
                f"{path} v-L2 feature shape {train_v_l2.shape} != F-norm shape {train.shape}"
            )
        if jacobian_norm_probes is None:
            jacobian_norm_probes = part_probe_count
        elif jacobian_norm_probes != part_probe_count:
            raise ValueError(f"Jacobian probe-count mismatch: {path}")
        if score_indices is None:
            score_indices = indices
        elif not np.array_equal(score_indices, indices):
            raise ValueError(f"score indices mismatch: {path}")

        for local_term, (ckpt, timestep, weight) in enumerate(zip(ckpts, timesteps, weights)):
            train_device = jax.device_put(jnp.asarray(train[local_term]))
            train_v_l2_device = jax.device_put(jnp.asarray(train_v_l2[local_term]))

            original_term = lookups["original"].get((int(ckpt), int(timestep)))
            if original_term is not None:
                query_device = jax.device_put(jnp.asarray(original_query[:, original_term, :]))
                for normalization, train_values in (
                    ("fnorm", train_device),
                    ("v_l2", train_v_l2_device),
                ):
                    dots = train_values @ query_device.T
                    query_norms = jnp.linalg.norm(query_device, axis=1) + 1e-8
                    sums[(normalization, "original", "raw")] += float(weight) * np.asarray(
                        jax.device_get(dots), dtype=np.float64
                    ).T
                    sums[(normalization, "original", "query_l2")] += float(weight) * np.asarray(
                        jax.device_get(dots / query_norms[None, :]), dtype=np.float64
                    ).T
                original_weight += abs(float(weight))
                original_terms += 1

            predicted_term = lookups["predicted"].get((int(ckpt), int(timestep)))
            if predicted_term is None:
                raise ValueError(f"missing predicted-noise query term ckpt={ckpt} t={timestep}")
            query_device = jax.device_put(jnp.asarray(predicted_query[:, predicted_term, :]))
            for normalization, train_values in (
                ("fnorm", train_device),
                ("v_l2", train_v_l2_device),
            ):
                dots = train_values @ query_device.T
                query_norms = jnp.linalg.norm(query_device, axis=1) + 1e-8
                sums[(normalization, "predicted", "raw")] += float(weight) ** 2 * np.asarray(
                    jax.device_get(jnp.square(dots)), dtype=np.float64
                ).T
                sums[(normalization, "predicted", "query_l2")] += float(weight) ** 2 * np.asarray(
                    jax.device_get(jnp.square(dots / query_norms[None, :])),
                    dtype=np.float64,
                ).T
            predicted_weight += float(weight) ** 2
            predicted_terms += 1

        print(
            f"[score shard {args.shard_index}/{args.shard_count}] checkpoint={ckpt_i} "
            f"original_terms={original_terms} predicted_terms={predicted_terms}",
            flush=True,
        )

    if score_indices is None:
        raise RuntimeError("score shard selected no checkpoint parts")
    atomic_savez(
        output,
        **{
            f"{normalization}_{target}_{query_variant}_sum": values
            for (normalization, target, query_variant), values in sums.items()
        },
        original_weight=np.asarray(original_weight, dtype=np.float64),
        predicted_weight=np.asarray(predicted_weight, dtype=np.float64),
        original_terms=np.asarray(original_terms, dtype=np.int32),
        predicted_terms=np.asarray(predicted_terms, dtype=np.int32),
        score_indices=score_indices,
        jacobian_norm_probes=np.asarray(jacobian_norm_probes, dtype=np.int32),
    )
    print(f"[saved] {output}", flush=True)


def materialize_scores(
    args: argparse.Namespace,
    namespace: str,
    values: np.ndarray,
    score_indices: np.ndarray,
    definition: str,
    query_variant: str,
    jacobian_norm_probes: int,
):
    normalization = "v_l2" if "_v_l2_" in namespace else "fnorm"
    for query_id, record in enumerate(records()):
        component = "score" if query_variant == "raw" else "score_query_normalized"
        out_dir = (
            result_root(args.experiment)
            / "attribution_score"
            / "prompted_solo"
            / f"train_seed_{args.train_seed}"
            / f"query_{_prompt_tag(str(record['prompt']))}"
            / f"initial_seed_{int(record['initial_seed'])}"
            / namespace
            / component
        )
        atomic_save(out_dir / "scores.npy", values[query_id])
        atomic_save(out_dir / "score_indices.npy", score_indices)
        manifest = {
            "algorithm": namespace,
            "score_variant": f"fixed_train_jacobian_normalization_{query_variant}",
            "definition": definition,
            "train_feature": (
                "mean_l (projected_residual_l * unit(P E[J]^T v_l))"
                if normalization == "v_l2"
                else "P(E[J]^T E[r]) / estimated_frobenius_norm(E[PJ])"
            ),
            "train_normalization": normalization,
            "query_normalization": query_variant,
            "train_mc_samples": 10,
            "jacobian_norm_probes": jacobian_norm_probes,
            "num_checkpoints": 50,
            "timestamps_per_checkpoint": 10,
            "projection_dim": 4096,
        }
        (out_dir / "score_artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )


def merge(args: argparse.Namespace) -> None:
    totals = {
        (normalization, target, query_variant): np.zeros((10, 5000), dtype=np.float64)
        for normalization in ("fnorm", "v_l2")
        for target in ("original", "predicted")
        for query_variant in ("raw", "query_l2")
    }
    original_weight = predicted_weight = 0.0
    original_terms = predicted_terms = 0
    score_indices = None
    jacobian_norm_probes = None
    for shard in range(args.shard_count):
        args.shard_index = shard
        path = shard_path(args)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            for normalization, target, query_variant in totals:
                totals[(normalization, target, query_variant)] += np.asarray(
                    data[f"{normalization}_{target}_{query_variant}_sum"], dtype=np.float64
                )
            original_weight += float(data["original_weight"])
            predicted_weight += float(data["predicted_weight"])
            original_terms += int(data["original_terms"])
            predicted_terms += int(data["predicted_terms"])
            indices = np.asarray(data["score_indices"], dtype=np.int64)
            shard_probe_count = int(np.asarray(data["jacobian_norm_probes"]).item())
        if score_indices is None:
            score_indices = indices
        elif not np.array_equal(score_indices, indices):
            raise ValueError(f"score indices mismatch: {path}")
        if jacobian_norm_probes is None:
            jacobian_norm_probes = shard_probe_count
        elif jacobian_norm_probes != shard_probe_count:
            raise ValueError(f"Jacobian probe-count mismatch: {path}")
    if original_terms != 490 or predicted_terms != 500:
        raise ValueError(
            f"expected original/predicted terms 490/500, got {original_terms}/{predicted_terms}"
        )
    expected = np.asarray(np.load(ATTRIBUTION_INDICES_PATH), dtype=np.int64)
    if score_indices is None or not np.array_equal(np.sort(score_indices), np.sort(expected)):
        raise ValueError("score indices do not match attribution subset")
    for normalization in ("fnorm", "v_l2"):
        for query_variant in ("raw", "query_l2"):
            materialize_scores(
                args,
                SCORE_NAMESPACES[(normalization, "original")],
                totals[(normalization, "original", query_variant)] / original_weight,
                score_indices,
                f"learning_rate_weighted_mean(dot({normalization}_train_feature, {query_variant}_grad(original_f)))",
                query_variant,
                int(jacobian_norm_probes),
            )
            materialize_scores(
                args,
                SCORE_NAMESPACES[(normalization, "predicted")],
                totals[(normalization, "predicted", query_variant)] / predicted_weight,
                score_indices,
                f"learning_rate_squared_weighted_mean(square(dot({normalization}_train_feature, {query_variant}_predicted_noise_probe_grad)))",
                query_variant,
                int(jacobian_norm_probes),
            )
    print(
        "[done] materialized 2 train normalizations x 2 query targets x 2 query normalizations",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("score-shard", "merge"))
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    if args.command == "score-shard":
        score_shard(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
