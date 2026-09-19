#!/usr/bin/env python3
"""Score predicted-noise probe contractions from saved Traj TracIn gradients."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
SQUARED_NAMESPACE = "predicted_noise_jvp_l2_squared"
SIGNED_NAMESPACE = "predicted_noise_jvp_signed"
FINAL_LINEAR_MEAN_NAMESPACE = "predicted_noise_jvp_final_linear_mean"
FINAL_SQUARE_THEN_MEAN_NAMESPACE = "predicted_noise_jvp_final_square_then_mean"
FINAL_MEAN_THEN_SQUARE_NAMESPACE = "predicted_noise_jvp_final_mean_then_square"
FINAL_POST_SQUARE_STAGING_NAMESPACE = "predicted_noise_jvp_final_post_square"
TIMESTAMP_CHECKPOINT_SQUARE_NAMESPACE = (
    "predicted_noise_jvp_timestamp_checkpoint_sum_square"
)

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
        return SQUARED_NAMESPACE
    return f"{SQUARED_NAMESPACE}_probe{num_probes}_r{probe_index}"


def with_namespace_suffix(namespace: str, namespace_suffix: str) -> str:
    return f"{namespace}_{namespace_suffix}" if namespace_suffix else namespace


def score_namespace(
    num_probes: int,
    contraction: str = "squared",
    namespace_suffix: str = "",
) -> str:
    if contraction == "signed":
        namespace = SIGNED_NAMESPACE
    elif contraction == "squared":
        namespace = SQUARED_NAMESPACE
    elif contraction == "final_post_square":
        namespace = FINAL_POST_SQUARE_STAGING_NAMESPACE
    elif contraction == "timestamp_checkpoint_square":
        namespace = TIMESTAMP_CHECKPOINT_SQUARE_NAMESPACE
    elif contraction == "termwise_squared_per_probe":
        namespace = "predicted_noise_jvp_termwise_squared_per_probe"
    elif contraction == "signed_squared":
        namespace = "predicted_noise_jvp_signed_squared"
    elif contraction == "absolute":
        namespace = "predicted_noise_jvp_absolute"
    elif contraction == "rms":
        namespace = "predicted_noise_jvp_rms"
    elif contraction == "probe_l2":
        namespace = "predicted_noise_jvp_probe_l2"
    elif contraction == "timestamp_probe_l2":
        namespace = "predicted_noise_jvp_timestamp_probe_l2"
    elif contraction == "median_absolute":
        namespace = "predicted_noise_jvp_median_absolute"
    elif contraction == "coordinatewise_squared":
        namespace = "predicted_noise_jvp_coordinatewise_squared"
    elif contraction == "checkpoint_timestamp_sum_square":
        namespace = "predicted_noise_jvp_checkpoint_timestamp_sum_square"
    else:
        raise ValueError(f"unknown contraction {contraction!r}")
    suffix = "" if num_probes == 1 else f"_probe{num_probes}"
    return with_namespace_suffix(f"traj_tracin_{namespace}{suffix}", namespace_suffix)


def final_score_namespace(
    num_probes: int,
    reduction: str,
    namespace_suffix: str = "",
) -> str:
    namespaces = {
        "linear_mean": FINAL_LINEAR_MEAN_NAMESPACE,
        "square_then_mean": FINAL_SQUARE_THEN_MEAN_NAMESPACE,
        "mean_then_square": FINAL_MEAN_THEN_SQUARE_NAMESPACE,
    }
    try:
        namespace = namespaces[reduction]
    except KeyError as exc:
        raise ValueError(f"unknown final-score reduction {reduction!r}") from exc
    suffix = "" if num_probes == 1 else f"_probe{num_probes}"
    return with_namespace_suffix(f"traj_tracin_{namespace}{suffix}", namespace_suffix)


def weighting_semantics(contraction: str) -> str:
    if contraction == "squared":
        return "learning_rate_weighted_sum_of_squared_gradient_contractions"
    if contraction == "final_post_square":
        return "square_after_learning_rate_weighted_term_sum_per_probe"
    if contraction == "timestamp_checkpoint_square":
        return "mean_timestamp_probe_square_of_checkpoint_lr_weighted_sum"
    if contraction == "termwise_squared_per_probe":
        return "per_probe_learning_rate_weighted_sum_of_squared_gradient_contractions"
    if contraction == "signed_squared":
        return "learning_rate_weighted_sum_of_signed_squared_gradient_contractions"
    if contraction == "absolute":
        return "learning_rate_weighted_sum_of_absolute_gradient_contractions"
    if contraction == "rms":
        return "learning_rate_weighted_sum_of_probe_rms_gradient_contractions"
    if contraction == "probe_l2":
        return "learning_rate_weighted_sum_of_probe_l2_gradient_contractions"
    if contraction == "timestamp_probe_l2":
        return "sum_timestamp_sqrt_sum_checkpoint_lr_times_sum_probe_squared_contractions"
    if contraction == "median_absolute":
        return "learning_rate_weighted_sum_of_probe_median_absolute_gradient_contractions"
    if contraction == "coordinatewise_squared":
        return "learning_rate_weighted_sum_of_coordinatewise_squared_gradient_products"
    if contraction == "checkpoint_timestamp_sum_square":
        return "sum_checkpoint_lr_times_mean_probe_square_of_mean_timestamps"
    return "learning_rate_weighted_sum_of_signed_gradient_contractions"


def reduce_final_probe_scores(probe_scores: np.ndarray) -> dict[str, np.ndarray]:
    """Reduce fully accumulated per-probe sample scores in three ways."""
    return {
        "linear_mean": np.mean(probe_scores, axis=0),
        "square_then_mean": np.mean(np.square(probe_scores), axis=0),
        "mean_then_square": np.square(np.mean(probe_scores, axis=0)),
    }


def reduce_timestamp_checkpoint_sums(checkpoint_sums: np.ndarray) -> np.ndarray:
    """Square checkpoint sums, then average timestamp and probe axes."""
    if checkpoint_sums.ndim != 4:
        raise ValueError(
            "checkpoint_sums must have shape "
            "(timestamps, probes, queries, datapoints)"
        )
    return np.mean(np.square(checkpoint_sums), axis=(0, 1))


def reduce_checkpoint_timestamp_means(probe_timestamp_means: np.ndarray) -> np.ndarray:
    """Square per-probe timestamp means, then average probes."""
    if probe_timestamp_means.ndim != 3:
        raise ValueError(
            "probe_timestamp_means must have shape "
            "(probes, queries, datapoints)"
        )
    return np.mean(np.square(probe_timestamp_means), axis=0)


def reduce_probe_rms(mean_probe_squares: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert a per-term mean of probe squares into its RMS magnitude."""
    return np.sqrt(np.maximum(mean_probe_squares, 0.0) + eps)


def reduce_probe_median_absolute(probe_values: np.ndarray) -> np.ndarray:
    """Take the median absolute contraction across the probe axis."""
    if probe_values.ndim < 1:
        raise ValueError("probe_values must include a probe axis")
    return np.median(np.abs(probe_values), axis=0)


def query_artifact_path(
    experiment: str,
    train_seed: int,
    epochs: int,
    query_id: int,
    *,
    num_probes: int,
    probe_index: int,
    query_namespace_pattern: str = "",
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
    namespace = (
        query_namespace_pattern.format(probe_index=probe_index)
        if query_namespace_pattern
        else probe_namespace(num_probes, probe_index)
    )
    return (
        run_root
        / f"seed_{seed:06d}_query_gradient_{namespace}"
        / "traj_tracin"
        / "query_gradient_artifact.npz"
    )


def train_part_dir(experiment: str, train_seed: int, namespace: str = "traj_tracin") -> Path:
    artifact = (
        result_root(experiment)
        / "model"
        / "prompted_solo"
        / f"seed_{train_seed}_train_gradient"
        / namespace
        / "train_datapoint_gradient_artifact.npz"
    )
    return Path(str(artifact) + ".parts")


def shard_root(
    experiment: str,
    train_seed: int,
    run_id: str,
    num_probes: int,
    contraction: str = "squared",
    namespace_suffix: str = "",
) -> Path:
    return (
        result_root(experiment)
        / "stream_score"
        / score_namespace(num_probes, contraction, namespace_suffix)
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
    patterns = (
        [value.strip() for value in args.query_namespace_patterns.split(",")]
        if args.query_namespace_patterns
        else []
    )
    seeds = (
        [int(value.strip()) for value in args.expected_query_probe_seeds.split(",")]
        if args.expected_query_probe_seeds
        else []
    )
    if patterns and len(patterns) != args.num_probes:
        raise ValueError("--query-namespace-patterns must contain one entry per probe")
    if seeds and len(seeds) != args.num_probes:
        raise ValueError("--expected-query-probe-seeds must contain one entry per probe")
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
                query_namespace_pattern=(
                    patterns[probe_index] if patterns else args.query_namespace_pattern
                ),
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as payload:
                feature = np.asarray(payload["query_features"], dtype=np.float32)
                objective = str(np.asarray(payload["query_objective"]).item())
                proj_dim = int(np.asarray(payload["proj_dim"]).item())
                stored_probe_index = int(np.asarray(payload.get("output_probe_index", 0)).item())
                stored_probe_mode = str(
                    np.asarray(payload.get("output_probe_mode", "independent_gaussian")).item()
                )
                stored_probe_bank_size = int(
                    np.asarray(payload.get("output_probe_bank_size", 1)).item()
                )
                stored_probe_seed = int(
                    np.asarray(payload.get("output_probe_seed", args.train_seed)).item()
                )
                metadata = {
                    "ckpt_indices": np.asarray(payload["ckpt_indices"], dtype=np.int32),
                    "timesteps": np.asarray(payload["timesteps"], dtype=np.int32),
                    "term_weights": np.asarray(payload["term_weights"], dtype=np.float64),
                }
            if objective != "trajectory_predicted_noise_probe":
                raise ValueError(f"{path} contains query objective {objective!r}")
            expected_stored_probe_index = 0 if patterns else probe_index
            if stored_probe_index != expected_stored_probe_index:
                raise ValueError(
                    f"{path} contains output probe {stored_probe_index}, expected "
                    f"{expected_stored_probe_index}"
                )
            expected_probe_seed = (
                seeds[probe_index] if seeds else args.expected_query_probe_seed
            )
            if (
                expected_probe_seed is not None
                and stored_probe_seed != expected_probe_seed
            ):
                raise ValueError(
                    f"{path} contains output probe seed {stored_probe_seed}, expected "
                    f"{expected_probe_seed}"
                )
            expected_mode = args.expected_query_probe_mode
            mode_matches = stored_probe_mode == expected_mode
            if expected_mode == "shared_orthogonal_extended":
                mode_matches = (
                    probe_index < 4
                    and stored_probe_mode == "shared_orthogonal"
                    and stored_probe_bank_size == 4
                ) or (
                    probe_index >= 4
                    and stored_probe_mode == "shared_orthogonal_extended"
                    and stored_probe_bank_size == args.num_probes
                )
            if expected_mode and not mode_matches:
                raise ValueError(
                    f"{path} contains probe mode {stored_probe_mode!r}, expected "
                    f"{expected_mode!r}"
                )
            if (
                args.expected_query_probe_mode == "shared_orthogonal"
                and stored_probe_bank_size != args.num_probes
            ):
                raise ValueError(
                    f"{path} contains orthogonal bank size {stored_probe_bank_size}, "
                    f"expected {args.num_probes}"
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
        shard_root(
            args.experiment,
            args.train_seed,
            args.run_id,
            args.num_probes,
            args.contraction,
            args.namespace_suffix,
        )
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
    timestep_values = list(dict.fromkeys(int(value) for value in query_meta["timesteps"]))
    if len(timestep_values) != 10:
        raise ValueError(f"expected 10 unique query timesteps, got {timestep_values}")
    timestep_slots = {value: slot for slot, value in enumerate(timestep_values)}
    if args.contraction in ("final_post_square", "termwise_squared_per_probe"):
        score_shape = (args.num_probes, 10, 5000)
    elif args.contraction == "timestamp_checkpoint_square":
        score_shape = (len(timestep_values), args.num_probes, 10, 5000)
    elif args.contraction == "timestamp_probe_l2":
        score_shape = (len(timestep_values), 10, 5000)
    else:
        score_shape = (10, 5000)
    sums = {
        "score": np.zeros(score_shape, dtype=np.float64),
        "score_query_normalized": np.zeros(score_shape, dtype=np.float64),
        "score_train_l2_normalized": np.zeros(score_shape, dtype=np.float64),
        "score_query_train_l2_normalized": np.zeros(score_shape, dtype=np.float64),
    }
    checkpoint_scores = None
    if args.retain_checkpoint_scores:
        if args.contraction not in ("checkpoint_timestamp_sum_square", "squared"):
            raise ValueError(
                "--retain-checkpoint-scores currently requires contraction "
                "checkpoint_timestamp_sum_square or squared"
            )
        checkpoint_scores = {
            component: np.zeros((50, 10, 5000), dtype=np.float64)
            for component in sums
        }
    score_indices = None
    used_terms = 0

    for ckpt_i in range(args.shard_index, 50, args.shard_count):
        part_path = train_part_dir(
            args.experiment, args.train_seed, args.train_namespace
        ) / f"ckpt_{ckpt_i:04d}.npz"
        if not part_path.is_file():
            raise FileNotFoundError(part_path)
        with np.load(part_path, allow_pickle=False) as payload:
            train = np.asarray(payload["train_features"], dtype=np.float32)
            indices = np.asarray(payload["score_indices"], dtype=np.int64)
            ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            weights = np.asarray(payload["term_weights"], dtype=np.float64)
            stored_semantics = (
                str(np.asarray(payload["train_feature_semantics"]).item())
                if "train_feature_semantics" in payload.files
                else ""
            )
        timestamp_shared_train = train.shape == (1, 5000, 4096)
        if train.shape != (10, 5000, 4096) and not timestamp_shared_train:
            raise ValueError(
                f"{part_path} expected (10,5000,4096) or timestamp-shared "
                f"(1,5000,4096), got {train.shape}"
            )
        if args.train_feature_semantics and stored_semantics != args.train_feature_semantics:
            raise ValueError(
                f"{part_path} contains train feature semantics {stored_semantics!r}, "
                f"expected {args.train_feature_semantics!r}"
            )
        if score_indices is None:
            score_indices = indices
        elif not np.array_equal(score_indices, indices):
            raise ValueError(f"score indices differ in {part_path}")

        checkpoint_probe_sums = None
        if args.contraction == "checkpoint_timestamp_sum_square":
            checkpoint_probe_sums = {
                component: np.zeros((args.num_probes, 10, 5000), dtype=np.float64)
                for component in sums
            }

        for local_term, (ckpt, timestep, weight) in enumerate(zip(ckpts, timesteps, weights)):
            query_term = lookup.get((int(ckpt), int(timestep)))
            if query_term is None:
                raise ValueError(f"no query feature for checkpoint={ckpt} timestep={timestep}")
            train_slot = 0 if timestamp_shared_train else local_term
            train_device = jax.device_put(jnp.asarray(train[train_slot]))
            train_norm = jnp.linalg.norm(train_device, axis=1) + 1e-8
            train_squared = (
                jnp.square(train_device)
                if args.contraction == "coordinatewise_squared"
                else None
            )
            term_scores = None
            term_probe_values = None
            if args.contraction == "median_absolute":
                term_probe_values = {
                    component: np.zeros((args.num_probes, 10, 5000), dtype=np.float64)
                    for component in sums
                }
            elif args.contraction not in (
                "final_post_square",
                "timestamp_checkpoint_square",
                "termwise_squared_per_probe",
                "checkpoint_timestamp_sum_square",
                "timestamp_probe_l2",
            ):
                term_scores = {
                    component: np.zeros((10, 5000), dtype=np.float64)
                    for component in sums
                }
            for probe_index in range(args.num_probes):
                query_device = jax.device_put(
                    jnp.asarray(query[probe_index, :, query_term, :])
                )
                query_norm = jnp.linalg.norm(query_device, axis=1) + 1e-8
                if args.contraction == "coordinatewise_squared":
                    assert train_squared is not None
                    coordinate_energy = train_squared @ jnp.square(query_device).T
                    train_norm_sq = jnp.square(train_norm[:, None])
                    query_norm_sq = jnp.square(query_norm[None, :])
                    probe_scores = {
                        "score": coordinate_energy,
                        "score_query_normalized": coordinate_energy / query_norm_sq,
                        "score_train_l2_normalized": coordinate_energy / train_norm_sq,
                        "score_query_train_l2_normalized": (
                            coordinate_energy / train_norm_sq / query_norm_sq
                        ),
                    }
                else:
                    directional = train_device @ query_device.T
                    if args.contraction in (
                        "squared",
                        "termwise_squared_per_probe",
                        "rms",
                        "probe_l2",
                    ):
                        transform = jnp.square
                    elif args.contraction == "signed_squared":
                        transform = lambda value: value * jnp.abs(value)
                    elif args.contraction == "absolute":
                        transform = jnp.abs
                    else:
                        transform = lambda value: value
                    probe_scores = {
                        "score": transform(directional),
                        "score_query_normalized": transform(
                            directional / query_norm[None, :]
                        ),
                        "score_train_l2_normalized": transform(
                            directional / train_norm[:, None]
                        ),
                        "score_query_train_l2_normalized": transform(
                            directional / train_norm[:, None] / query_norm[None, :]
                        ),
                    }
                for component, values in probe_scores.items():
                    host_values = np.asarray(jax.device_get(values), dtype=np.float64).T
                    if args.contraction == "median_absolute":
                        assert term_probe_values is not None
                        term_probe_values[component][probe_index] = host_values
                    elif args.contraction == "final_post_square":
                        # Keep each probe separate through the complete trajectory sum.
                        # Squaring and cross-probe reduction happen only after shards merge.
                        sums[component][probe_index] += float(weight) * host_values
                    elif args.contraction == "termwise_squared_per_probe":
                        sums[component][probe_index] += float(weight) * host_values
                    elif args.contraction == "timestamp_checkpoint_square":
                        # Stored term weights are eta_c / num_timestamps. Restore eta_c
                        # here because timestamp averaging occurs after checkpoint sums.
                        checkpoint_lr = float(weight) * float(len(timestep_values))
                        sums[component][timestep_slots[int(timestep)], probe_index] += (
                            checkpoint_lr * host_values
                        )
                    elif args.contraction == "checkpoint_timestamp_sum_square":
                        assert checkpoint_probe_sums is not None
                        checkpoint_probe_sums[component][probe_index] += (
                            host_values / float(len(timesteps))
                        )
                    elif args.contraction == "timestamp_probe_l2":
                        sums[component][timestep_slots[int(timestep)]] += (
                            float(weight) * host_values
                        )
                    else:
                        assert term_scores is not None
                        divisor = (
                            1.0
                            if args.contraction == "probe_l2"
                            else float(args.num_probes)
                        )
                        term_scores[component] += host_values / divisor
            # The requested transform applies only to the gradient-produced
            # directional derivative. The learning-rate weight remains linear, and the
            # stored per-snapshot weight already averages the 10 snapshots.
            if term_scores is not None:
                term_weight = float(weight)
                for component, values in term_scores.items():
                    if args.contraction in ("rms", "probe_l2"):
                        values = reduce_probe_rms(values)
                    sums[component] += term_weight * values
                    if (
                        checkpoint_scores is not None
                        and args.contraction == "squared"
                    ):
                        # Retain E_timestamp[E_probe[z^2]] without the positive
                        # checkpoint-wide LR scalar for independent LDS auditing.
                        checkpoint_scores[component][ckpt_i] += (
                            values / float(len(timesteps))
                        )
            if term_probe_values is not None:
                term_weight = float(weight)
                for component, values in term_probe_values.items():
                    sums[component] += term_weight * reduce_probe_median_absolute(values)
            used_terms += 1

        if args.contraction == "checkpoint_timestamp_sum_square":
            assert checkpoint_probe_sums is not None
            checkpoint_lr_values = weights * float(len(timesteps))
            if not np.allclose(
                checkpoint_lr_values,
                checkpoint_lr_values[0],
                rtol=1e-6,
                atol=1e-12,
            ):
                raise ValueError(
                    f"checkpoint {ckpt_i} has inconsistent learning-rate weights"
                )
            checkpoint_lr = float(checkpoint_lr_values[0])
            for component, values in checkpoint_probe_sums.items():
                checkpoint_unweighted_score = reduce_checkpoint_timestamp_means(values)
                checkpoint_score = checkpoint_lr * checkpoint_unweighted_score
                sums[component] += checkpoint_score
                if checkpoint_scores is not None:
                    # A positive checkpoint-wide LR scalar cannot change Spearman LDS.
                    # Retain the unweighted score so the final cosine-LR=0 checkpoint
                    # can still be audited independently.
                    checkpoint_scores[component][ckpt_i] = checkpoint_unweighted_score
        print(f"[score shard {args.shard_index}/{args.shard_count}] checkpoint={ckpt_i} terms={used_terms}", flush=True)

    if score_indices is None:
        raise RuntimeError("checkpoint shard selected no train parts")
    arrays = {f"sums_{component}": values for component, values in sums.items()}
    if checkpoint_scores is not None:
        arrays.update(
            {
                f"checkpoint_scores_{component}": values
                for component, values in checkpoint_scores.items()
            }
        )
    atomic_savez(
        output,
        **arrays,
        score_indices=score_indices,
        used_terms=np.asarray(used_terms, dtype=np.int32),
        weighting_semantics=np.asarray(weighting_semantics(args.contraction)),
    )
    print(f"[saved] {output}", flush=True)


def atomic_save(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, value)
    tmp.replace(path)


def merge(args: argparse.Namespace) -> None:
    if args.contraction == "final_post_square":
        score_shape = (args.num_probes, 10, 5000)
    elif args.contraction == "timestamp_checkpoint_square":
        score_shape = (10, args.num_probes, 10, 5000)
    elif args.contraction == "timestamp_probe_l2":
        score_shape = (10, 10, 5000)
    else:
        score_shape = (10, 5000)
    totals = {
        "score": np.zeros(score_shape, dtype=np.float64),
        "score_query_normalized": np.zeros(score_shape, dtype=np.float64),
        "score_train_l2_normalized": np.zeros(score_shape, dtype=np.float64),
        "score_query_train_l2_normalized": np.zeros(score_shape, dtype=np.float64),
    }
    checkpoint_totals = None
    if args.retain_checkpoint_scores:
        if args.contraction not in ("checkpoint_timestamp_sum_square", "squared"):
            raise ValueError(
                "--retain-checkpoint-scores currently requires contraction "
                "checkpoint_timestamp_sum_square or squared"
            )
        checkpoint_totals = {
            component: np.zeros((50, 10, 5000), dtype=np.float64)
            for component in totals
        }
    terms = 0
    score_indices = None
    for shard in range(args.shard_count):
        path = (
            shard_root(
                args.experiment,
                args.train_seed,
                args.run_id,
                args.num_probes,
                args.contraction,
                args.namespace_suffix,
            )
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
                if checkpoint_totals is not None:
                    checkpoint_totals[component] += np.asarray(
                        payload[f"checkpoint_scores_{component}"],
                        dtype=np.float64,
                    )
            semantics = str(np.asarray(payload.get("weighting_semantics", "")).item())
            expected_semantics = weighting_semantics(args.contraction)
            if semantics != expected_semantics:
                raise ValueError(
                    f"{path} has incompatible weighting semantics {semantics!r}; "
                    f"expected {expected_semantics!r}"
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

    if checkpoint_totals is not None:
        per_checkpoint_path = (
            shard_root(
                args.experiment,
                args.train_seed,
                args.run_id,
                args.num_probes,
                args.contraction,
                args.namespace_suffix,
            )
            / "per_checkpoint_scores.npz"
        )
        atomic_savez(
            per_checkpoint_path,
            **{
                f"checkpoint_scores_{component}": values
                for component, values in checkpoint_totals.items()
            },
            score_indices=score_indices,
            checkpoint_indices=np.arange(50, dtype=np.int32),
            checkpoint_learning_rate_applied=np.asarray(False),
            weighting_semantics=np.asarray(weighting_semantics(args.contraction)),
        )
        print(f"[saved] {per_checkpoint_path}", flush=True)

    if args.contraction == "final_post_square":
        score_sets = {
            final_score_namespace(
                args.num_probes,
                reduction,
                args.namespace_suffix,
            ): {
                component: reduced[reduction]
                for component, probe_values in totals.items()
                for reduced in (reduce_final_probe_scores(probe_values),)
            }
            for reduction in ("linear_mean", "square_then_mean", "mean_then_square")
        }
    elif args.contraction == "timestamp_checkpoint_square":
        score_sets = {
            score_namespace(
                args.num_probes,
                args.contraction,
                args.namespace_suffix,
            ): {
                component: reduce_timestamp_checkpoint_sums(values)
                for component, values in totals.items()
            }
        }
    elif args.contraction == "timestamp_probe_l2":
        score_sets = {
            score_namespace(
                args.num_probes,
                args.contraction,
                args.namespace_suffix,
            ): {
                component: np.sqrt(np.maximum(values, 0.0)).sum(axis=0)
                for component, values in totals.items()
            }
        }
    else:
        score_sets = {
            score_namespace(
                args.num_probes,
                args.contraction,
                args.namespace_suffix,
            ): totals
        }

    for output_namespace, scores in score_sets.items():
        for component, values in scores.items():
            for query_id, record in enumerate(records()):
                out_dir = (
                    result_root(args.experiment)
                    / "attribution_score"
                    / "prompted_solo"
                    / f"train_seed_{args.train_seed}"
                    / f"query_{_prompt_tag(str(record['prompt']))}"
                    / f"initial_seed_{int(record['initial_seed'])}"
                    / output_namespace
                    / component
                )
                atomic_save(out_dir / "scores.npy", values[query_id])
                atomic_save(out_dir / "score_indices.npy", score_indices)
                manifest = {
                    "algorithm": output_namespace,
                    "score_variant": component,
                    "definition": (
                        "mean_probes(square(learning_rate_weighted_sum_terms(signed_normalized_dot_product)))"
                        if "final_square_then_mean" in output_namespace
                        else "square(mean_probes(learning_rate_weighted_sum_terms(signed_normalized_dot_product)))"
                        if "final_mean_then_square" in output_namespace
                        else "mean_timestamp_probe(square(sum_checkpoint(learning_rate_times_signed_normalized_dot_product)))"
                        if "timestamp_checkpoint_sum_square" in output_namespace
                        else "mean_probes(learning_rate_weighted_sum_terms(signed_normalized_dot_product))"
                        if "final_linear_mean" in output_namespace
                        else "learning_rate_weighted_sum_terms(squared_normalized_dot_product)"
                        if args.contraction == "squared"
                        else "learning_rate_weighted_sum_terms(signed_square(normalized_dot_product))"
                        if args.contraction == "signed_squared"
                        else "learning_rate_weighted_sum_terms(abs(normalized_dot_product))"
                        if args.contraction == "absolute"
                        else "learning_rate_weighted_sum_terms(sqrt(mean_probe(square(normalized_dot_product))))"
                        if args.contraction == "rms"
                        else "learning_rate_weighted_sum_terms(sqrt(sum_probe(square(normalized_dot_product))))"
                        if args.contraction == "probe_l2"
                        else "sum_timestamp(sqrt(sum_checkpoint(learning_rate_times_sum_probe(square(normalized_dot_product))))))"
                        if args.contraction == "timestamp_probe_l2"
                        else "learning_rate_weighted_sum_terms(median_probe(abs(normalized_dot_product)))"
                        if args.contraction == "median_absolute"
                        else "learning_rate_weighted_sum_terms(sum_coordinate_squared_products_after_normalization)"
                        if args.contraction == "coordinatewise_squared"
                        else "sum_checkpoint(learning_rate_times_mean_probe(square(mean_timestamp(normalized_dot_product))))"
                        if args.contraction == "checkpoint_timestamp_sum_square"
                        else "learning_rate_weighted_sum_terms(signed_normalized_dot_product)"
                    ),
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
                (out_dir / "score_artifact_manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True)
                )
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
                    query_namespace_pattern=args.query_namespace_pattern,
                )
                if path.is_file():
                    path.unlink()
                    print(f"[cleanup] removed transient query gradient: {path}", flush=True)
    print(
        f"[done] materialized {len(score_sets)} score reduction(s) x four normalization "
        f"variants for 10 queries; terms={terms}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("score-shard", "merge"))
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--train-namespace", default="traj_tracin")
    parser.add_argument(
        "--train-feature-semantics",
        default="",
        help="Optional exact semantic tag required in every train part.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=16)
    parser.add_argument("--run-id", default=os.environ.get("PRED_NOISE_JVP_RUN_ID", "manual"))
    parser.add_argument("--num-probes", type=int, default=1)
    parser.add_argument(
        "--contraction",
        choices=(
            "squared",
            "signed",
            "final_post_square",
            "timestamp_checkpoint_square",
            "termwise_squared_per_probe",
            "signed_squared",
            "absolute",
            "rms",
            "probe_l2",
            "timestamp_probe_l2",
            "median_absolute",
            "coordinatewise_squared",
            "checkpoint_timestamp_sum_square",
        ),
        default="squared",
        help=(
            "squared: square each term before trajectory accumulation; signed: keep the "
            "signed term; final_post_square: retain per-probe trajectory sums and emit "
            "both mean(square(S_r)) and square(mean(S_r)); "
            "timestamp_checkpoint_square: for each timestamp/probe, sum all "
            "learning-rate-weighted checkpoints, square, then average timestamps/probes; "
            "termwise_squared_per_probe: square each product and retain every probe "
            "through the trajectory sum for subgroup analysis; signed_squared: z*abs(z); "
            "absolute: abs(z); "
            "rms: sqrt(mean_probe(z^2)) for every checkpoint/timestamp term; "
            "probe_l2: sqrt(sum_probe(z^2)) for every checkpoint/timestamp term; "
            "median_absolute: median_probe(abs(z)) for every checkpoint/timestamp term; "
            "coordinatewise_squared: sum_k (train_k*query_k)^2."
            " checkpoint_timestamp_sum_square: within each checkpoint and probe, "
            "mean timestamps, square, mean probes, then apply the checkpoint LR once."
        ),
    )
    parser.add_argument(
        "--query-namespace-pattern",
        default="",
        help="Optional artifact namespace with {probe_index}; allows reuse of an existing probe bank.",
    )
    parser.add_argument(
        "--query-namespace-patterns",
        default="",
        help="Comma-separated artifact namespace patterns, one independently generated bank per probe.",
    )
    parser.add_argument(
        "--expected-query-probe-seeds",
        default="",
        help="Comma-separated probe seeds corresponding to --query-namespace-patterns.",
    )
    parser.add_argument(
        "--expected-query-probe-mode",
        choices=(
            "",
            "independent_gaussian",
            "timestamp_shared_gaussian",
            "shared_orthogonal",
            "shared_orthogonal_extended",
        ),
        default="",
    )
    parser.add_argument(
        "--namespace-suffix",
        default="",
        help="Optional suffix keeping a specialized probe experiment independent.",
    )
    parser.add_argument(
        "--expected-query-probe-seed",
        type=int,
        default=None,
        help="Reject query artifacts that were generated from a different probe-bank seed.",
    )
    parser.add_argument("--cleanup-query-artifacts", action="store_true")
    parser.add_argument(
        "--retain-checkpoint-scores",
        action="store_true",
        help="Also retain each checkpoint's independent score contribution.",
    )
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    if args.num_probes <= 0:
        raise ValueError("--num-probes must be positive")
    if not args.run_id or "/" in args.run_id:
        raise ValueError("--run-id must be a non-empty path component")
    if args.namespace_suffix and any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for char in args.namespace_suffix
    ):
        raise ValueError("--namespace-suffix may contain only letters, numbers, _ and -")
    if args.command == "score-shard":
        score_shard(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
