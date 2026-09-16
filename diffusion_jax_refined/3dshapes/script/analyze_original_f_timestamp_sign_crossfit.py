#!/usr/bin/env python3
"""Crossfit timestamp signs for original-f scores under four L2 variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from analyze_predicted_noise_probe12_sign_flips import rowwise_spearman, sign_matrix
from analyze_predicted_noise_probe8_choose4 import cache_group, load_target_data
from dataset_config import _prompt_tag
from run_expected_residual_jacobian_scores import load_query_bank, train_part_dir


VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")
TARGETS = ("endpoint_contarfactual", "traj_contarfactual")


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def lds(predictions: np.ndarray, endpoint: np.ndarray, trajectory: np.ndarray):
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    end = 100.0 * rowwise_spearman(values, endpoint)
    traj = 100.0 * rowwise_spearman(values, trajectory)
    return end, traj, 0.5 * (end + traj)


def sign_label(values: np.ndarray, timesteps: np.ndarray) -> str:
    return ",".join(
        f"{int(t)}:{'+' if value > 0 else '-'}"
        for t, value in zip(timesteps, values)
    )


def target_data(args, query_id: int, score_indices: np.ndarray):
    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    record = records[query_id]
    eval_root = (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "eval"
        / "prompted_solo"
        / f"query_{_prompt_tag(str(record['prompt']))}"
        / f"initial_seed_{int(record['initial_seed'])}"
    )
    return load_target_data(cache_group(eval_root), score_indices, targets=TARGETS)


def build_components(args, query_ids: list[int]):
    import jax
    import jax.numpy as jnp

    query_bank, metadata = load_query_bank(
        args, args.original_query_namespace, "trajectory_next_checkpoint_noise_mse"
    )
    query_bank = query_bank[query_ids]
    ckpts = np.asarray(metadata["ckpt_indices"], dtype=np.int32)
    timesteps_per_term = np.asarray(metadata["timesteps"], dtype=np.int32)
    timesteps = np.asarray(list(dict.fromkeys(int(x) for x in timesteps_per_term)))
    timestep_slot = {int(value): slot for slot, value in enumerate(timesteps)}
    lookup = {
        (int(ckpt), int(timestep)): term
        for term, (ckpt, timestep) in enumerate(zip(ckpts, timesteps_per_term))
    }
    components = {
        variant: np.zeros((len(query_ids), len(timesteps), 5000), dtype=np.float64)
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
            semantics = str(np.asarray(payload["train_feature_semantics"]).item())
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
                "query_train_l2": dots
                / (train_norm[:, None] * query_norm[None, :]),
            }
            slot = timestep_slot[int(timestep)]
            for variant, value in values.items():
                components[variant][:, slot] += float(weight) * np.asarray(
                    jax.device_get(value), dtype=np.float64
                ).T
            used_terms += 1
        print(f"[components] checkpoint={checkpoint + 1}/50 terms={used_terms}", flush=True)

    if used_terms != 490 or score_indices is None:
        raise ValueError(f"expected 490 original-f terms, found {used_terms}")
    return components, timesteps, score_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-ids", default="1,3,4,8")
    parser.add_argument("--num-checkpoints", type=int, default=50)
    parser.add_argument("--num-snapshots", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260916)
    parser.add_argument("--train-namespace", default="traj_tracin_loss_direction_residual_rms")
    parser.add_argument("--original-query-namespace", default="loss_direction_residual_rms_original_f")
    parser.add_argument(
        "--train-feature-semantics",
        default="unit_projected_expected_loss_gradient_times_matching_mc_residual_rms",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    query_ids = parse_ints(args.query_ids)
    args.skip_predicted = True
    args.shard_index = 0
    args.shard_count = 1
    args.run_id = "timestamp_crossfit"

    components, timesteps, score_indices = build_components(args, query_ids)
    signs = sign_matrix(len(timesteps)).astype(np.int8)
    summary_rows = []
    split_rows = []

    for qslot, query_id in enumerate(query_ids):
        # Use identical model folds for every query/variant comparison.
        rng = np.random.default_rng(args.random_seed)
        incidence, true = target_data(args, query_id, score_indices)
        endpoint = true[TARGETS[0]]
        trajectory = true[TARGETS[1]]
        folds = []
        for _ in range(args.repeats):
            permutation = rng.permutation(len(endpoint))
            folds.append((permutation[::2], permutation[1::2]))

        for variant in VARIANTS:
            # Existing cached LDS uses prediction_sign=-1.
            timestamp_predictions = -components[variant][qslot] @ incidence.T
            all_predictions = signs @ timestamp_predictions
            full_end, full_traj, full_joint = lds(all_predictions, endpoint, trajectory)
            full_index = int(np.argmax(full_joint))
            plus_end, plus_traj, plus_joint = lds(
                timestamp_predictions.sum(axis=0), endpoint, trajectory
            )
            minus_end, minus_traj, minus_joint = lds(
                -timestamp_predictions.sum(axis=0), endpoint, trajectory
            )

            repeat_values = []
            for repeat, pair in enumerate(folds):
                fold_values = []
                for train_fold in range(2):
                    train_ids = pair[train_fold]
                    test_ids = pair[1 - train_fold]
                    _, _, train_joint = lds(
                        signs @ timestamp_predictions[:, train_ids],
                        endpoint[train_ids],
                        trajectory[train_ids],
                    )
                    selected = int(np.argmax(train_joint))
                    heldout = signs[selected] @ timestamp_predictions[:, test_ids]
                    heldout_end, heldout_traj, heldout_joint = lds(
                        heldout, endpoint[test_ids], trajectory[test_ids]
                    )
                    value = (
                        float(heldout_end[0]),
                        float(heldout_traj[0]),
                        float(heldout_joint[0]),
                    )
                    fold_values.append(value)
                    split_rows.append(
                        {
                            "query": query_id,
                            "variant": variant,
                            "repeat": repeat,
                            "train_fold": train_fold,
                            "selected_mask": selected,
                            "selected_signs": sign_label(signs[selected], timesteps),
                            "train_cf_joint_percent": float(train_joint[selected]),
                            "heldout_endpoint_percent": value[0],
                            "heldout_trajectory_percent": value[1],
                            "heldout_cf_joint_percent": value[2],
                        }
                    )
                repeat_values.append(tuple(np.mean(fold_values, axis=0)))

            repeat_values = np.asarray(repeat_values)
            row = {
                "query": query_id,
                "variant": variant,
                "all_plus_cf_joint_percent": float(plus_joint[0]),
                "all_minus_cf_joint_percent": float(minus_joint[0]),
                "full_oracle_cf_joint_percent": float(full_joint[full_index]),
                "full_oracle_signs": sign_label(signs[full_index], timesteps),
                "crossfit_endpoint_mean_percent": float(repeat_values[:, 0].mean()),
                "crossfit_trajectory_mean_percent": float(repeat_values[:, 1].mean()),
                "crossfit_cf_joint_mean_percent": float(repeat_values[:, 2].mean()),
                "crossfit_cf_joint_std_percent": float(repeat_values[:, 2].std(ddof=1)),
                "crossfit_cf_joint_min_percent": float(repeat_values[:, 2].min()),
                "crossfit_cf_joint_max_percent": float(repeat_values[:, 2].max()),
                "crossfit_positive_repeat_fraction": float(
                    np.mean(repeat_values[:, 2] > 0)
                ),
            }
            summary_rows.append(row)
            print(
                f"Q{query_id} {variant:16s} plus={plus_joint[0]:+7.3f}% "
                f"minus={minus_joint[0]:+7.3f}% full={full_joint[full_index]:7.3f}% "
                f"CV={row['crossfit_cf_joint_mean_percent']:+7.3f}% "
                f"±{row['crossfit_cf_joint_std_percent']:.3f}% "
                f"CV>0={row['crossfit_positive_repeat_fraction']:.2f}",
                flush=True,
            )

    write_csv(args.out_dir / "summary.csv", summary_rows)
    write_csv(args.out_dir / "per_split.csv", split_rows)
    np.savez_compressed(
        args.out_dir / "timestamp_components.npz",
        query_ids=np.asarray(query_ids),
        timesteps=timesteps,
        score_indices=score_indices,
        **components,
    )
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
