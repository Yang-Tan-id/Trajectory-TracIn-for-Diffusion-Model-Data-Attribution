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


DEFAULT_TRAIN_NAMESPACE = "traj_tracin_expected_residual_jacobian_probe_aligned"
DEFAULT_ORIGINAL_QUERY_NAMESPACE = "expected_residual_jacobian_original_f"
DEFAULT_PREDICTED_QUERY_NAMESPACE = "expected_residual_jacobian_predicted_noise"
DEFAULT_SCORE_NAMESPACE_PREFIX = "traj_tracin_expected_residual_jacobian_probe_aligned_v_l2"
DEFAULT_TRAIN_FEATURE_SEMANTICS = "mean_probe_projected_residual_times_unit_probe_gradient"


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


def train_part_dir(args: argparse.Namespace) -> Path:
    artifact = (
        result_root(args.experiment)
        / "model"
        / "prompted_solo"
        / f"seed_{args.train_seed}_train_gradient"
        / args.train_namespace
        / "train_datapoint_gradient_artifact.npz"
    )
    return Path(str(artifact) + ".parts")


def shard_path(args: argparse.Namespace) -> Path:
    return (
        result_root(args.experiment)
        / "stream_score"
        / args.train_namespace
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


def load_predicted_query_probes(args: argparse.Namespace):
    banks = []
    reference = None
    for probe_index in range(args.predicted_num_probes):
        namespace = args.predicted_query_namespace.format(probe_index=probe_index)
        bank, metadata = load_query_bank(
            args, namespace, "trajectory_predicted_noise_probe"
        )
        if reference is None:
            reference = metadata
        else:
            for key in metadata:
                if not np.allclose(metadata[key], reference[key], rtol=1e-6, atol=1e-12):
                    raise ValueError(
                        f"predicted-noise probe metadata mismatch for {key}: {namespace}"
                    )
        banks.append(bank)
    assert reference is not None
    return np.stack(banks, axis=0), reference


def score_shard(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp

    output = shard_path(args)
    if output.is_file():
        print(f"[skip] score shard exists: {output}", flush=True)
        return

    original_query, original_meta = load_query_bank(
        args, args.original_query_namespace, "trajectory_next_checkpoint_noise_mse"
    )
    predicted_query, predicted_meta = load_predicted_query_probes(args)
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
        (target, query_variant): np.zeros((10, 5000), dtype=np.float64)
        for target in ("original", "predicted")
        for query_variant in ("raw", "query_l2")
    }
    original_terms = 0
    predicted_terms = 0
    score_indices = None
    jacobian_norm_probes = None

    for ckpt_i in range(args.shard_index, args.num_checkpoints, args.shard_count):
        path = train_part_dir(args) / f"ckpt_{ckpt_i:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            train = np.asarray(data["train_features"], dtype=np.float32)
            indices = np.asarray(data["score_indices"], dtype=np.int64)
            ckpts = np.asarray(data["ckpt_indices"], dtype=np.int32)
            timesteps = np.asarray(data["timesteps"], dtype=np.int32)
            weights = np.asarray(data["term_weights"], dtype=np.float64)
            semantics = str(np.asarray(data["train_feature_semantics"]).item())
            part_probe_count = int(np.asarray(data.get("jacobian_norm_probes", 0)).item())
        if semantics != args.train_feature_semantics:
            raise ValueError(f"unexpected train feature semantics in {path}: {semantics}")
        expected_shape = (args.num_snapshots, 5000, 4096)
        if train.shape != expected_shape:
            raise ValueError(f"{path} expected {expected_shape}, got {train.shape}")
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

            original_term = lookups["original"].get((int(ckpt), int(timestep)))
            if original_term is not None:
                query_device = jax.device_put(jnp.asarray(original_query[:, original_term, :]))
                dots = train_device @ query_device.T
                query_norms = jnp.linalg.norm(query_device, axis=1) + 1e-8
                sums[("original", "raw")] += float(weight) * np.asarray(
                    jax.device_get(dots), dtype=np.float64
                ).T
                sums[("original", "query_l2")] += float(weight) * np.asarray(
                    jax.device_get(dots / query_norms[None, :]), dtype=np.float64
                ).T
                original_terms += 1

            predicted_term = lookups["predicted"].get((int(ckpt), int(timestep)))
            if predicted_term is None:
                raise ValueError(f"missing predicted-noise query term ckpt={ckpt} t={timestep}")
            predicted_raw = np.zeros((10, 5000), dtype=np.float64)
            predicted_query_l2 = np.zeros((10, 5000), dtype=np.float64)
            for probe_index in range(args.predicted_num_probes):
                query_device = jax.device_put(
                    jnp.asarray(predicted_query[probe_index, :, predicted_term, :])
                )
                dots = train_device @ query_device.T
                query_norms = jnp.linalg.norm(query_device, axis=1) + 1e-8
                if args.predicted_contraction == "squared":
                    raw_values = jnp.square(dots)
                    query_l2_values = jnp.square(dots / query_norms[None, :])
                else:
                    raw_values = dots
                    query_l2_values = dots / query_norms[None, :]
                predicted_raw += np.asarray(
                    jax.device_get(raw_values), dtype=np.float64
                ).T / float(args.predicted_num_probes)
                predicted_query_l2 += np.asarray(
                    jax.device_get(query_l2_values),
                    dtype=np.float64,
                ).T / float(args.predicted_num_probes)
            # Keep the TrajTracIn learning-rate weight linear; `weight`
            # already includes the per-snapshot averaging factor.
            sums[("predicted", "raw")] += float(weight) * predicted_raw
            sums[("predicted", "query_l2")] += (
                float(weight) * predicted_query_l2
            )
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
            f"{target}_{query_variant}_sum": values
            for (target, query_variant), values in sums.items()
        },
        original_terms=np.asarray(original_terms, dtype=np.int32),
        predicted_terms=np.asarray(predicted_terms, dtype=np.int32),
        score_indices=score_indices,
        jacobian_norm_probes=np.asarray(jacobian_norm_probes, dtype=np.int32),
        weighting_semantics=np.asarray(
            f"learning_rate_weighted_sum_with_{args.predicted_contraction}_predicted_contractions"
        ),
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
            "train_feature": args.train_feature_description,
            "train_normalization": "fixed_in_saved_train_feature",
            "query_normalization": query_variant,
            "train_mc_samples": 10,
            "jacobian_norm_probes": jacobian_norm_probes,
            "predicted_noise_query_probes": args.predicted_num_probes,
            "num_checkpoints": args.num_checkpoints,
            "timestamps_per_checkpoint": args.num_snapshots,
            "projection_dim": 4096,
        }
        (out_dir / "score_artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )


def merge(args: argparse.Namespace) -> None:
    totals = {
        (target, query_variant): np.zeros((10, 5000), dtype=np.float64)
        for target in ("original", "predicted")
        for query_variant in ("raw", "query_l2")
    }
    original_terms = predicted_terms = 0
    score_indices = None
    jacobian_norm_probes = None
    for shard in range(args.shard_count):
        args.shard_index = shard
        path = shard_path(args)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            for target, query_variant in totals:
                totals[(target, query_variant)] += np.asarray(
                    data[f"{target}_{query_variant}_sum"], dtype=np.float64
                )
            semantics = str(np.asarray(data.get("weighting_semantics", "")).item())
            expected_semantics = (
                f"learning_rate_weighted_sum_with_{args.predicted_contraction}_predicted_contractions"
            )
            if semantics != expected_semantics:
                raise ValueError(
                    f"{path} has incompatible weighting semantics {semantics!r}; "
                    f"expected {expected_semantics!r}"
                )
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
    expected_original_terms = (args.num_checkpoints - 1) * args.num_snapshots
    expected_predicted_terms = args.num_checkpoints * args.num_snapshots
    if original_terms != expected_original_terms or predicted_terms != expected_predicted_terms:
        raise ValueError(
            f"expected original/predicted terms {expected_original_terms}/{expected_predicted_terms}, "
            f"got {original_terms}/{predicted_terms}"
        )
    expected = np.asarray(np.load(ATTRIBUTION_INDICES_PATH), dtype=np.int64)
    if score_indices is None or not np.array_equal(np.sort(score_indices), np.sort(expected)):
        raise ValueError("score indices do not match attribution subset")
    for query_variant in ("raw", "query_l2"):
        materialize_scores(
            args,
            f"{args.score_namespace_prefix}_original_f",
            totals[("original", query_variant)],
            score_indices,
            f"learning_rate_weighted_sum(dot(v_l2_train_feature, {query_variant}_grad(original_f)))",
            query_variant,
            int(jacobian_norm_probes),
        )
        materialize_scores(
            args,
            f"{args.score_namespace_prefix}_predicted_noise",
            totals[("predicted", query_variant)],
            score_indices,
            (
                f"learning_rate_weighted_sum("
                f"{args.predicted_contraction}(dot(v_l2_train_feature, "
                f"{query_variant}_predicted_noise_probe_grad)))"
                if args.predicted_contraction == "squared"
                else f"learning_rate_weighted_sum(dot(v_l2_train_feature, "
                f"{query_variant}_predicted_noise_probe_grad))"
            ),
            query_variant,
            int(jacobian_norm_probes),
        )
    print(
        "[done] materialized 1 train normalization x 2 query targets x 2 query normalizations",
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
    parser.add_argument("--num-checkpoints", type=int, default=50)
    parser.add_argument("--num-snapshots", type=int, default=10)
    parser.add_argument("--train-namespace", default=DEFAULT_TRAIN_NAMESPACE)
    parser.add_argument("--original-query-namespace", default=DEFAULT_ORIGINAL_QUERY_NAMESPACE)
    parser.add_argument("--predicted-query-namespace", default=DEFAULT_PREDICTED_QUERY_NAMESPACE)
    parser.add_argument("--predicted-num-probes", type=int, default=1)
    parser.add_argument(
        "--predicted-contraction",
        choices=("squared", "signed"),
        default="squared",
        help="Apply either square(dot) or the signed dot to predicted-noise probes.",
    )
    parser.add_argument("--score-namespace-prefix", default=DEFAULT_SCORE_NAMESPACE_PREFIX)
    parser.add_argument(
        "--train-feature-semantics", default=DEFAULT_TRAIN_FEATURE_SEMANTICS
    )
    parser.add_argument(
        "--train-feature-description",
        default="mean_l (projected_residual_l * unit(P E[J]^T v_l))",
    )
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    if args.predicted_num_probes <= 0:
        raise ValueError("--predicted-num-probes must be positive")
    if args.predicted_num_probes > 1 and "{probe_index}" not in args.predicted_query_namespace:
        raise ValueError(
            "--predicted-query-namespace must contain {probe_index} when multiple probes are used"
        )
    if args.command == "score-shard":
        score_shard(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
