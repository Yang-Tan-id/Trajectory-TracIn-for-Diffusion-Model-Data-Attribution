#!/usr/bin/env python3
"""Optimize one sign per checkpoint/timestamp component for each Q0 probe.

The 2^500 search is approximated with multi-start best-coordinate ascent.
Both the deliberately leaked full-data oracle and two-fold held-out estimates
are reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT / "script") not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT / "script"))

from analyze_predicted_noise_old_fresh_term_signs import BANKS, atomic_savez, write_csv
from analyze_predicted_noise_probe12_sign_flips import rowwise_spearman
from analyze_predicted_noise_probe8_choose4 import cache_group, load_target_data
from run_predicted_noise_jvp_l2_squared import query_artifact_path, train_part_dir


CF_TARGETS = ("endpoint_contarfactual", "traj_contarfactual")


def load_query_bank(args):
    probes = []
    reference = None
    for bank in ("old12", "fresh12"):
        for probe_index in range(12):
            path = query_artifact_path(
                args.experiment,
                args.train_seed,
                args.epochs,
                args.query_id,
                num_probes=12,
                probe_index=probe_index,
                query_namespace_pattern=BANKS[bank]["query_pattern"],
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as payload:
                feature = np.asarray(payload["query_features"], dtype=np.float32)
                metadata = {
                    "ckpt_indices": np.asarray(payload["ckpt_indices"], dtype=np.int32),
                    "timesteps": np.asarray(payload["timesteps"], dtype=np.int32),
                }
            feature /= np.maximum(np.linalg.norm(feature, axis=1, keepdims=True), 1e-8)
            probes.append(feature)
            if reference is None:
                reference = metadata
            else:
                for key, expected in reference.items():
                    if not np.array_equal(metadata[key], expected):
                        raise ValueError(f"query metadata mismatch: {path}:{key}")
    assert reference is not None
    query = np.stack(probes, axis=0)
    if query.shape != (24, 500, 4096):
        raise ValueError(f"expected query bank (24,500,4096), got {query.shape}")
    return query, reference


def target_data(args, score_indices):
    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    record = records[args.query_id]
    eval_root = (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "eval"
        / "prompted_solo"
        / f"query_{str(record['prompt']).replace(',', '_')}"
        / f"initial_seed_{int(record['initial_seed'])}"
    )
    return load_target_data(cache_group(eval_root), score_indices)


def shard(args):
    import jax
    import jax.numpy as jnp

    output = args.out_dir / f"term_prediction_shard_{args.shard_index}.npz"
    if output.is_file():
        print(f"[skip] {output}", flush=True)
        return

    query, metadata = load_query_bank(args)
    lookup = {
        (int(checkpoint), int(timestep)): term
        for term, (checkpoint, timestep) in enumerate(
            zip(metadata["ckpt_indices"], metadata["timesteps"])
        )
    }
    predictions = None
    used = np.zeros(500, dtype=np.bool_)
    score_indices = None
    incidence_device = None

    for checkpoint_slot in range(args.shard_index, 50, args.shard_count):
        path = train_part_dir(args.experiment, args.train_seed) / f"ckpt_{checkpoint_slot:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            train = np.asarray(payload["train_features"], dtype=np.float32)
            indices = np.asarray(payload["score_indices"], dtype=np.int64)
            checkpoints = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            weights = np.asarray(payload["term_weights"], dtype=np.float64)
        if score_indices is None:
            score_indices = indices
            incidence, _ = target_data(args, score_indices)
            predictions = np.zeros((24, 500, incidence.shape[0]), dtype=np.float32)
            incidence_device = jax.device_put(jnp.asarray(incidence.T, dtype=jnp.float32))
        elif not np.array_equal(score_indices, indices):
            raise ValueError(f"score indices differ in {path}")
        assert predictions is not None and incidence_device is not None

        for local_term, (checkpoint, timestep, weight) in enumerate(
            zip(checkpoints, timesteps, weights)
        ):
            term = lookup[(int(checkpoint), int(timestep))]
            train_device = jax.device_put(jnp.asarray(train[local_term]))
            train_unit = train_device / jnp.maximum(
                jnp.linalg.norm(train_device, axis=1, keepdims=True), 1e-8
            )
            query_device = jax.device_put(jnp.asarray(query[:, term]))
            datapoint_scores = train_unit @ query_device.T
            subset_predictions = datapoint_scores.T @ incidence_device
            predictions[:, term] = (
                float(weight)
                * np.asarray(jax.device_get(subset_predictions), dtype=np.float32)
            )
            used[term] = True
        print(f"[term predictions] checkpoint={checkpoint_slot + 1}/50", flush=True)

    assert predictions is not None and score_indices is not None
    atomic_savez(
        output,
        predictions=predictions,
        used=used,
        score_indices=score_indices,
        ckpt_indices=metadata["ckpt_indices"],
        timesteps=metadata["timesteps"],
    )
    print(f"[saved] {output}", flush=True)


def cf_lds(predictions, endpoint, trajectory):
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    endpoint_lds = 100.0 * rowwise_spearman(values, endpoint)
    trajectory_lds = 100.0 * rowwise_spearman(values, trajectory)
    return endpoint_lds, trajectory_lds, 0.5 * (endpoint_lds + trajectory_lds)


def covariance_start(features, endpoint, trajectory):
    def centered_rank(values):
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        return ranks - ranks.mean()

    centered = features - features.mean(axis=1, keepdims=True)
    direction = centered @ (centered_rank(endpoint) + centered_rank(trajectory))
    return np.where(direction >= 0.0, 1, -1).astype(np.int8)


def coordinate_ascent(features, endpoint, trajectory, initial, max_steps, tolerance):
    signs = np.asarray(initial, dtype=np.int8).copy()
    prediction = signs.astype(np.float64) @ features
    _, _, current_array = cf_lds(prediction, endpoint, trajectory)
    current = float(current_array[0])
    steps = 0
    while steps < max_steps:
        candidates = prediction[None, :] - 2.0 * signs[:, None] * features
        _, _, objectives = cf_lds(candidates, endpoint, trajectory)
        best_term = int(np.argmax(objectives))
        best = float(objectives[best_term])
        if best <= current + tolerance:
            break
        signs[best_term] *= -1
        prediction = candidates[best_term]
        current = best
        steps += 1
    endpoint_value, trajectory_value, joint = cf_lds(
        prediction, endpoint, trajectory
    )
    return signs, float(endpoint_value[0]), float(trajectory_value[0]), float(joint[0]), steps


def optimize(features, endpoint, trajectory, starts, max_steps, tolerance):
    best = None
    for start_name, signs in starts:
        result = coordinate_ascent(
            features, endpoint, trajectory, signs, max_steps, tolerance
        )
        candidate = (*result, start_name)
        if best is None or candidate[3] > best[3]:
            best = candidate
    assert best is not None
    return best


def expanded_timestamp_signs(args, metadata):
    path = (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "eval"
        / "predicted_noise_24_individual_probe_timestamp_signs"
        / f"source_run_{args.source_run_id}"
        / "individual_probe_best_binary_vectors.npz"
    )
    with np.load(path, allow_pickle=False) as payload:
        query_ids = np.asarray(payload["query_ids"], dtype=np.int32)
        timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        best_signs = np.asarray(payload["best_signs"], dtype=np.int8)
    qslots = np.flatnonzero(query_ids == args.query_id)
    if len(qslots) != 1:
        raise ValueError(f"query {args.query_id} missing from {path}")
    slots = {int(value): slot for slot, value in enumerate(timesteps)}
    term_slots = np.asarray([slots[int(value)] for value in metadata["timesteps"]])
    return best_signs[:, int(qslots[0]), :][:, term_slots]


def merge_optimize(args):
    payloads = []
    for shard_index in range(args.shard_count):
        path = args.out_dir / f"term_prediction_shard_{shard_index}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        payloads.append(np.load(path, allow_pickle=False))
    metadata = {
        "ckpt_indices": np.asarray(payloads[0]["ckpt_indices"], dtype=np.int32),
        "timesteps": np.asarray(payloads[0]["timesteps"], dtype=np.int32),
    }
    score_indices = np.asarray(payloads[0]["score_indices"], dtype=np.int64)
    used_sum = sum(np.asarray(payload["used"], dtype=np.int8) for payload in payloads)
    if not np.all(used_sum == 1):
        raise ValueError(f"each term must occur once; counts={np.unique(used_sum)}")
    predictions = sum(
        np.asarray(payload["predictions"], dtype=np.float64) for payload in payloads
    )
    incidence, true_values = target_data(args, score_indices)
    del incidence
    endpoint = true_values["endpoint_contarfactual"]
    trajectory = true_values["traj_contarfactual"]
    timestamp_starts = expanded_timestamp_signs(args, metadata)

    rng = np.random.default_rng(args.random_seed)
    permutation = rng.permutation(len(endpoint))
    folds = (permutation[::2], permutation[1::2])
    full_signs = np.empty((24, 500), dtype=np.int8)
    fold_signs = np.empty((24, 2, 500), dtype=np.int8)
    rows = []

    print("Q0 PER-PROBE 500-COMPONENT SIGN ORACLE — BOTH-L2 CF JOINT")
    print(
        f"{'P':>2s} {'ALL+':>8s} {'TS10':>8s} {'FULL':>8s} {'STEPS':>6s} "
        f"{'CV END':>8s} {'CV TRAJ':>8s} {'CV JOINT':>9s}"
    )
    print("-" * 78)
    for probe in range(24):
        features = predictions[probe]
        all_plus = np.ones(500, dtype=np.int8)
        _, _, all_plus_joint = cf_lds(all_plus @ features, endpoint, trajectory)
        _, _, timestamp_joint = cf_lds(
            timestamp_starts[probe] @ features, endpoint, trajectory
        )

        starts = [
            ("all_plus", all_plus),
            ("timestamp10", timestamp_starts[probe]),
            ("covariance", covariance_start(features, endpoint, trajectory)),
        ]
        for restart in range(args.random_restarts):
            starts.append(
                (
                    f"random_{restart}",
                    rng.choice(np.asarray([-1, 1], dtype=np.int8), size=500),
                )
            )
        full = optimize(
            features,
            endpoint,
            trajectory,
            starts,
            args.max_steps,
            args.tolerance,
        )
        full_signs[probe] = full[0]

        heldout_metrics = []
        for fold_index in range(2):
            train_indices = folds[fold_index]
            test_indices = folds[1 - fold_index]
            train_features = features[:, train_indices]
            train_endpoint = endpoint[train_indices]
            train_trajectory = trajectory[train_indices]
            fold_starts = [
                ("all_plus", all_plus),
                (
                    "covariance",
                    covariance_start(
                        train_features, train_endpoint, train_trajectory
                    ),
                ),
            ]
            trained = optimize(
                train_features,
                train_endpoint,
                train_trajectory,
                fold_starts,
                args.max_steps,
                args.tolerance,
            )
            fold_signs[probe, fold_index] = trained[0]
            heldout_metrics.append(
                cf_lds(
                    trained[0] @ features[:, test_indices],
                    endpoint[test_indices],
                    trajectory[test_indices],
                )
            )
        cv_endpoint = float(np.mean([value[0][0] for value in heldout_metrics]))
        cv_trajectory = float(np.mean([value[1][0] for value in heldout_metrics]))
        cv_joint = 0.5 * (cv_endpoint + cv_trajectory)
        row = {
            "global_probe": probe + 1,
            "bank": "old12" if probe < 12 else "fresh12",
            "probe_in_bank": probe + 1 if probe < 12 else probe - 11,
            "all_plus_cf_joint_lds_percent": float(all_plus_joint[0]),
            "timestamp10_oracle_cf_joint_lds_percent": float(timestamp_joint[0]),
            "full500_oracle_endpoint_lds_percent": full[1],
            "full500_oracle_traj_lds_percent": full[2],
            "full500_oracle_cf_joint_lds_percent": full[3],
            "full500_steps": full[4],
            "full500_start": full[5],
            "full500_positive_sign_fraction": float(np.mean(full[0] > 0)),
            "crossfit_endpoint_lds_percent": cv_endpoint,
            "crossfit_traj_lds_percent": cv_trajectory,
            "crossfit_cf_joint_lds_percent": cv_joint,
        }
        rows.append(row)
        print(
            f"{probe + 1:2d} {all_plus_joint[0]:7.3f}% "
            f"{timestamp_joint[0]:7.3f}% {full[3]:7.3f}% {full[4]:6d} "
            f"{cv_endpoint:7.3f}% {cv_trajectory:7.3f}% {cv_joint:8.3f}%",
            flush=True,
        )

    rows.sort(key=lambda row: float(row["full500_oracle_cf_joint_lds_percent"]), reverse=True)
    write_csv(args.out_dir / "probe500_component_sign_oracle.csv", rows)
    atomic_savez(
        args.out_dir / "probe500_component_best_signs.npz",
        global_probes=np.arange(1, 25, dtype=np.int32),
        full_signs=full_signs,
        crossfit_signs=fold_signs,
        fold0_indices=folds[0],
        fold1_indices=folds[1],
        ckpt_indices=metadata["ckpt_indices"],
        timesteps=metadata["timesteps"],
    )
    print("\nTOP FULL-DATA ORACLE")
    for rank, row in enumerate(rows[:10], start=1):
        print(
            f"{rank:2d}. P{int(row['global_probe']):02d} "
            f"full={float(row['full500_oracle_cf_joint_lds_percent']):.3f}% "
            f"crossfit={float(row['crossfit_cf_joint_lds_percent']):.3f}% "
            f"start={row['full500_start']}"
        )
    print(f"[saved] {args.out_dir}")


def add_common(parser):
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-id", type=int, default=0)
    parser.add_argument("--source-run-id", default="3506389")
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, required=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard_parser = subparsers.add_parser("shard")
    add_common(shard_parser)
    shard_parser.add_argument("--shard-index", type=int, required=True)
    merge_parser = subparsers.add_parser("merge-optimize")
    add_common(merge_parser)
    merge_parser.add_argument("--random-seed", type=int, default=20260916)
    merge_parser.add_argument("--random-restarts", type=int, default=2)
    merge_parser.add_argument("--max-steps", type=int, default=250)
    merge_parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    if args.query_id != 0:
        raise ValueError("this focused experiment currently expects Q0")
    if args.command == "shard":
        shard(args)
    else:
        merge_optimize(args)


if __name__ == "__main__":
    main()
