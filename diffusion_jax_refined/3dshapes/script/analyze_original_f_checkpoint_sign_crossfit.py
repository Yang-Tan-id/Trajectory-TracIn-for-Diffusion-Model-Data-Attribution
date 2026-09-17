#!/usr/bin/env python3
"""Crossfit structured checkpoint signs for direct-loss original-f scores."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from analyze_original_f_timestamp_sign_crossfit import (
    TARGETS,
    VARIANTS,
    lds,
    parse_ints,
    target_data,
)
from analyze_predicted_noise_probe12_sign_flips import sign_matrix
from run_expected_residual_jacobian_scores import load_query_bank, train_part_dir


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_checkpoint_components(args, query_ids: list[int]):
    import jax
    import jax.numpy as jnp

    query_bank, metadata = load_query_bank(
        args, args.original_query_namespace, "trajectory_next_checkpoint_noise_mse"
    )
    query_bank = query_bank[query_ids]
    term_ckpts = np.asarray(metadata["ckpt_indices"], dtype=np.int32)
    term_timesteps = np.asarray(metadata["timesteps"], dtype=np.int32)
    checkpoints = np.asarray(sorted(set(int(x) for x in term_ckpts)), dtype=np.int32)
    checkpoint_slot = {int(value): slot for slot, value in enumerate(checkpoints)}
    lookup = {
        (int(ckpt), int(timestep)): term
        for term, (ckpt, timestep) in enumerate(zip(term_ckpts, term_timesteps))
    }
    components = {
        variant: np.zeros((len(query_ids), len(checkpoints), 5000), dtype=np.float64)
        for variant in VARIANTS
    }
    score_indices = None
    used_terms = 0

    for checkpoint in range(args.num_checkpoints):
        path = train_part_dir(args) / f"ckpt_{checkpoint:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            train_terms = np.asarray(payload["train_features"], dtype=np.float32)
            part_ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            part_timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            weights = np.asarray(payload["term_weights"], dtype=np.float64)
            indices = np.asarray(payload["score_indices"], dtype=np.int64)
            semantics = str(
                np.asarray(
                    payload.get(
                        "train_feature_semantics",
                        "raw_projected_expected_loss_gradient",
                    )
                ).item()
            )
        if semantics != args.train_feature_semantics:
            raise ValueError(f"{path}: unexpected semantics {semantics!r}")
        if score_indices is None:
            score_indices = indices
        elif not np.array_equal(score_indices, indices):
            raise ValueError(f"score indices mismatch: {path}")

        for local, (ckpt, timestep, weight) in enumerate(
            zip(part_ckpts, part_timesteps, weights)
        ):
            term = lookup.get((int(ckpt), int(timestep)))
            if term is None:
                continue
            train = jax.device_put(jnp.asarray(train_terms[local]))
            query = jax.device_put(jnp.asarray(query_bank[:, term, :]))
            dots = train @ query.T
            train_norm = jnp.linalg.norm(train, axis=1) + 1e-8
            query_norm = jnp.linalg.norm(query, axis=1) + 1e-8
            values = {
                "raw": dots,
                "query_l2": dots / query_norm[None, :],
                "train_l2": dots / train_norm[:, None],
                "query_train_l2": dots / (train_norm[:, None] * query_norm[None, :]),
            }
            slot = checkpoint_slot[int(ckpt)]
            for variant, value in values.items():
                components[variant][:, slot] += float(weight) * np.asarray(
                    jax.device_get(value), dtype=np.float64
                ).T
            used_terms += 1
        print(
            f"[checkpoint components] checkpoint={checkpoint + 1}/50 terms={used_terms}",
            flush=True,
        )

    if used_terms != 490 or score_indices is None:
        raise ValueError(f"expected 490 original-f terms, found {used_terms}")
    return components, checkpoints, score_indices


def save_components(path, components, query_ids, checkpoints, score_indices):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        query_ids=np.asarray(query_ids, dtype=np.int32),
        checkpoints=checkpoints,
        score_indices=score_indices,
        **components,
    )


def load_components(path: Path, query_ids: list[int]):
    with np.load(path, allow_pickle=False) as payload:
        found = np.asarray(payload["query_ids"], dtype=np.int32)
        if not np.array_equal(found, np.asarray(query_ids, dtype=np.int32)):
            raise ValueError(f"{path}: query IDs {found.tolist()} do not match {query_ids}")
        checkpoints = np.asarray(payload["checkpoints"], dtype=np.int32)
        score_indices = np.asarray(payload["score_indices"], dtype=np.int64)
        components = {
            variant: np.asarray(payload[variant], dtype=np.float64)
            for variant in VARIANTS
        }
    return components, checkpoints, score_indices


def grouped_signs(num_checkpoints: int, num_groups: int) -> np.ndarray:
    assignments = sign_matrix(num_groups).astype(np.int8)
    groups = np.array_split(np.arange(num_checkpoints), num_groups)
    expanded = np.ones((len(assignments), num_checkpoints), dtype=np.int8)
    for group_id, indices in enumerate(groups):
        expanded[:, indices] = assignments[:, group_id, None]
    return expanded


def change_point_signs(num_checkpoints: int) -> np.ndarray:
    candidates = []
    for cut in range(num_checkpoints + 1):
        signs = np.ones(num_checkpoints, dtype=np.int8)
        signs[cut:] = -1
        candidates.extend((signs, -signs))
    return np.unique(np.stack(candidates), axis=0)


def joint_value(prediction, endpoint, trajectory) -> float:
    return float(lds(prediction, endpoint, trajectory)[2][0])


def coordinate_ascent_signs(
    component_predictions: np.ndarray,
    endpoint: np.ndarray,
    trajectory: np.ndarray,
    rng: np.random.Generator,
    max_sweeps: int = 12,
) -> tuple[np.ndarray, float]:
    per_component = np.asarray(
        [joint_value(row, endpoint, trajectory) for row in component_predictions]
    )
    correlation_start = np.where(per_component >= 0.0, 1, -1).astype(np.int8)
    starts = [
        np.ones(len(component_predictions), dtype=np.int8),
        -np.ones(len(component_predictions), dtype=np.int8),
        correlation_start,
        -correlation_start,
        rng.choice(np.asarray([-1, 1], dtype=np.int8), len(component_predictions)),
        rng.choice(np.asarray([-1, 1], dtype=np.int8), len(component_predictions)),
    ]
    best_signs = starts[0]
    best_value = -np.inf
    for initial in starts:
        signs = initial.copy()
        prediction = signs @ component_predictions
        value = joint_value(prediction, endpoint, trajectory)
        for _ in range(max_sweeps):
            improved = False
            for component in rng.permutation(len(signs)):
                candidate_prediction = (
                    prediction
                    - 2.0 * float(signs[component]) * component_predictions[component]
                )
                candidate_value = joint_value(
                    candidate_prediction, endpoint, trajectory
                )
                if candidate_value > value + 1e-12:
                    signs[component] *= -1
                    prediction = candidate_prediction
                    value = candidate_value
                    improved = True
            if not improved:
                break
        if value > best_value:
            best_signs = signs.copy()
            best_value = value
    return best_signs, best_value


def best_from_candidates(signs, components, endpoint, trajectory):
    _, _, values = lds(signs @ components, endpoint, trajectory)
    index = int(np.argmax(values))
    return signs[index], float(values[index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-ids", default="1,3,4,8")
    parser.add_argument("--num-checkpoints", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260916)
    parser.add_argument("--train-namespace", default="traj_tracin")
    parser.add_argument(
        "--original-query-namespace", default="loss_direction_residual_rms_original_f"
    )
    parser.add_argument(
        "--train-feature-semantics", default="raw_projected_expected_loss_gradient"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    query_ids = parse_ints(args.query_ids)
    args.skip_predicted = True
    args.shard_index = 0
    args.shard_count = 1
    args.run_id = "checkpoint_crossfit"

    component_cache = args.out_dir / "checkpoint_components.npz"
    if component_cache.is_file():
        print(f"[reuse] checkpoint components: {component_cache}", flush=True)
        components, checkpoints, score_indices = load_components(
            component_cache, query_ids
        )
    else:
        components, checkpoints, score_indices = build_checkpoint_components(
            args, query_ids
        )
        save_components(
            component_cache, components, query_ids, checkpoints, score_indices
        )
        print(f"[components saved] {component_cache}", flush=True)

    candidate_sets = {
        "single_change_point": change_point_signs(len(checkpoints)),
        "five_bins": grouped_signs(len(checkpoints), 5),
        "ten_bins": grouped_signs(len(checkpoints), 10),
    }
    summary_rows = []
    split_rows = []

    for qslot, query_id in enumerate(query_ids):
        incidence, true = target_data(args, query_id, score_indices)
        endpoint = true[TARGETS[0]]
        trajectory = true[TARGETS[1]]
        rng = np.random.default_rng(args.random_seed)
        folds = []
        for _ in range(args.repeats):
            permutation = rng.permutation(len(endpoint))
            folds.append((permutation[::2], permutation[1::2]))

        for variant in VARIANTS:
            checkpoint_predictions = -components[variant][qslot] @ incidence.T
            plus = joint_value(checkpoint_predictions.sum(axis=0), endpoint, trajectory)
            minus = -plus
            for method in (*candidate_sets, "individual_coordinate"):
                method_rng = np.random.default_rng(
                    args.random_seed + 1000 * query_id + 37 * VARIANTS.index(variant)
                )
                if method == "individual_coordinate":
                    full_signs, full_value = coordinate_ascent_signs(
                        checkpoint_predictions, endpoint, trajectory, method_rng
                    )
                else:
                    full_signs, full_value = best_from_candidates(
                        candidate_sets[method], checkpoint_predictions, endpoint, trajectory
                    )

                repeat_values = []
                repeat_global = []
                for repeat, pair in enumerate(folds):
                    fold_values = []
                    fold_global = []
                    for train_fold in range(2):
                        train_ids = pair[train_fold]
                        test_ids = pair[1 - train_fold]
                        train_components = checkpoint_predictions[:, train_ids]
                        if method == "individual_coordinate":
                            selected_signs, train_value = coordinate_ascent_signs(
                                train_components,
                                endpoint[train_ids],
                                trajectory[train_ids],
                                method_rng,
                            )
                        else:
                            selected_signs, train_value = best_from_candidates(
                                candidate_sets[method],
                                train_components,
                                endpoint[train_ids],
                                trajectory[train_ids],
                            )
                        heldout_prediction = (
                            selected_signs @ checkpoint_predictions[:, test_ids]
                        )
                        heldout_end, heldout_traj, heldout_joint = lds(
                            heldout_prediction,
                            endpoint[test_ids],
                            trajectory[test_ids],
                        )
                        train_plus = joint_value(
                            train_components.sum(axis=0),
                            endpoint[train_ids],
                            trajectory[train_ids],
                        )
                        global_sign = 1 if train_plus >= 0.0 else -1
                        heldout_global = joint_value(
                            global_sign
                            * checkpoint_predictions[:, test_ids].sum(axis=0),
                            endpoint[test_ids],
                            trajectory[test_ids],
                        )
                        value = float(heldout_joint[0])
                        fold_values.append(value)
                        fold_global.append(heldout_global)
                        split_rows.append(
                            {
                                "query": query_id,
                                "variant": variant,
                                "method": method,
                                "repeat": repeat,
                                "train_fold": train_fold,
                                "train_cf_joint_percent": train_value,
                                "heldout_endpoint_percent": float(heldout_end[0]),
                                "heldout_trajectory_percent": float(heldout_traj[0]),
                                "heldout_cf_joint_percent": value,
                                "heldout_global_sign_cf_joint_percent": heldout_global,
                                "heldout_improvement_percent": value - heldout_global,
                                "selected_positive_checkpoints": int(
                                    np.sum(selected_signs > 0)
                                ),
                            }
                        )
                    repeat_values.append(float(np.mean(fold_values)))
                    repeat_global.append(float(np.mean(fold_global)))

                values = np.asarray(repeat_values)
                globals_ = np.asarray(repeat_global)
                improvements = values - globals_
                row = {
                    "query": query_id,
                    "variant": variant,
                    "method": method,
                    "all_plus_cf_joint_percent": plus,
                    "all_minus_cf_joint_percent": minus,
                    "full_oracle_cf_joint_percent": full_value,
                    "full_positive_checkpoints": int(np.sum(full_signs > 0)),
                    "crossfit_cf_joint_mean_percent": float(values.mean()),
                    "crossfit_cf_joint_std_percent": float(values.std(ddof=1)),
                    "crossfit_positive_repeat_fraction": float(np.mean(values > 0)),
                    "crossfit_global_baseline_mean_percent": float(globals_.mean()),
                    "crossfit_improvement_mean_percent": float(improvements.mean()),
                    "crossfit_beat_global_fraction": float(np.mean(improvements > 0)),
                }
                summary_rows.append(row)
                print(
                    f"Q{query_id} {variant:16s} {method:21s} "
                    f"full={full_value:+7.3f}% CV={values.mean():+7.3f}% "
                    f"global={globals_.mean():+7.3f}% "
                    f"delta={improvements.mean():+7.3f}% "
                    f"beat={np.mean(improvements > 0):.2f}",
                    flush=True,
                )

    write_csv(args.out_dir / "summary.csv", summary_rows)
    write_csv(args.out_dir / "per_split.csv", split_rows)
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
