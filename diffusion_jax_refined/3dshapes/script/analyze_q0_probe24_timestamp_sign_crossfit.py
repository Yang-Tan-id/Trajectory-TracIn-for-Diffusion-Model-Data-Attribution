#!/usr/bin/env python3
"""Exact repeated two-fold crossfit for Q0 per-probe timestamp signs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_predicted_noise_old_fresh_term_signs import write_csv
from analyze_predicted_noise_probe12_sign_flips import rowwise_spearman, sign_matrix
from analyze_predicted_noise_probe8_choose4 import cache_group, load_target_data


SHAPES_ROOT = Path(__file__).resolve().parents[1]
BANKS = ("old12", "fresh12")
CF_TARGETS = ("endpoint_contarfactual", "traj_contarfactual")


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


def lds_rows(predictions, endpoint, trajectory):
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    endpoint_lds = 100.0 * rowwise_spearman(values, endpoint)
    trajectory_lds = 100.0 * rowwise_spearman(values, trajectory)
    return endpoint_lds, trajectory_lds, 0.5 * (endpoint_lds + trajectory_lds)


def sign_label(values, timesteps):
    return ",".join(
        f"{int(timestep)}:{'+' if sign > 0 else '-'}"
        for timestep, sign in zip(timesteps, values)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--source-run-id", default="3506389")
    parser.add_argument("--query-id", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260916)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    source = (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "eval"
        / "predicted_noise_old_fresh12_per_probe_timestamp_orientation"
        / f"run_{args.source_run_id}"
        / "per_probe_timestamp_scores.npz"
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as payload:
        banks = {
            bank: np.asarray(payload[bank], dtype=np.float64) for bank in BANKS
        }
        query_ids = np.asarray(payload["query_ids"], dtype=np.int32)
        timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        score_indices = np.asarray(payload["score_indices"], dtype=np.int64)
    qslots = np.flatnonzero(query_ids == args.query_id)
    if len(qslots) != 1:
        raise ValueError(f"query {args.query_id} not uniquely present in {source}")
    qslot = int(qslots[0])
    expected = (12, 10, len(query_ids), len(score_indices))
    for bank, values in banks.items():
        if values.shape != expected:
            raise ValueError(f"{bank}: expected {expected}, got {values.shape}")

    incidence, true_values = target_data(args, score_indices)
    endpoint = true_values[CF_TARGETS[0]]
    trajectory = true_values[CF_TARGETS[1]]
    num_models = len(endpoint)
    signs = sign_matrix(len(timesteps)).astype(np.int8)
    rng = np.random.default_rng(args.random_seed)
    repeated_folds = []
    for _ in range(args.repeats):
        permutation = rng.permutation(num_models)
        repeated_folds.append((permutation[::2], permutation[1::2]))

    split_rows = []
    summary_rows = []
    print("Q0 INDIVIDUAL-PROBE TIMESTAMP-SIGN EXACT CROSSFIT")
    print(
        f"models={num_models} assignments={len(signs)} repeats={args.repeats} "
        f"two-fold heldout evaluations={2 * args.repeats}"
    )
    print(
        f"{'P':>2s} {'BANK':8s} {'FULL':>8s} {'CV END':>9s} {'CV TRAJ':>9s} "
        f"{'CV JOINT':>10s} {'STD':>8s} {'CV>0':>7s} {'GAP':>9s}"
    )
    print("-" * 92)

    for bank_slot, bank in enumerate(BANKS):
        for probe_slot in range(12):
            global_probe = bank_slot * 12 + probe_slot + 1
            components = banks[bank][probe_slot, :, qslot, :]
            timestamp_predictions = components @ incidence.T
            all_predictions = signs @ timestamp_predictions
            full_endpoint, full_trajectory, full_joint = lds_rows(
                all_predictions, endpoint, trajectory
            )
            full_index = int(np.argmax(full_joint))

            repeat_endpoint = []
            repeat_trajectory = []
            repeat_joint = []
            for repeat, folds in enumerate(repeated_folds):
                fold_values = []
                for train_fold in range(2):
                    train_indices = folds[train_fold]
                    test_indices = folds[1 - train_fold]
                    train_predictions = signs @ timestamp_predictions[:, train_indices]
                    _, _, train_joint = lds_rows(
                        train_predictions,
                        endpoint[train_indices],
                        trajectory[train_indices],
                    )
                    selected = int(np.argmax(train_joint))
                    heldout_prediction = (
                        signs[selected].astype(np.float64)
                        @ timestamp_predictions[:, test_indices]
                    )
                    heldout_endpoint, heldout_trajectory, heldout_joint = lds_rows(
                        heldout_prediction,
                        endpoint[test_indices],
                        trajectory[test_indices],
                    )
                    fold_values.append(
                        (
                            float(heldout_endpoint[0]),
                            float(heldout_trajectory[0]),
                            float(heldout_joint[0]),
                        )
                    )
                    split_rows.append(
                        {
                            "global_probe": global_probe,
                            "bank": bank,
                            "probe_in_bank": probe_slot + 1,
                            "repeat": repeat,
                            "train_fold": train_fold,
                            "train_models": len(train_indices),
                            "heldout_models": len(test_indices),
                            "selected_mask": selected,
                            "selected_signs": sign_label(signs[selected], timesteps),
                            "train_cf_joint_lds_percent": float(train_joint[selected]),
                            "heldout_endpoint_lds_percent": float(heldout_endpoint[0]),
                            "heldout_traj_lds_percent": float(heldout_trajectory[0]),
                            "heldout_cf_joint_lds_percent": float(heldout_joint[0]),
                        }
                    )
                repeat_endpoint.append(float(np.mean([value[0] for value in fold_values])))
                repeat_trajectory.append(float(np.mean([value[1] for value in fold_values])))
                repeat_joint.append(float(np.mean([value[2] for value in fold_values])))

            row = {
                "global_probe": global_probe,
                "bank": bank,
                "probe_in_bank": probe_slot + 1,
                "full_oracle_cf_joint_lds_percent": float(full_joint[full_index]),
                "full_oracle_mask": full_index,
                "full_oracle_signs": sign_label(signs[full_index], timesteps),
                "crossfit_endpoint_mean_percent": float(np.mean(repeat_endpoint)),
                "crossfit_trajectory_mean_percent": float(np.mean(repeat_trajectory)),
                "crossfit_cf_joint_mean_percent": float(np.mean(repeat_joint)),
                "crossfit_cf_joint_std_percent": float(np.std(repeat_joint, ddof=1)),
                "crossfit_cf_joint_min_percent": float(np.min(repeat_joint)),
                "crossfit_cf_joint_max_percent": float(np.max(repeat_joint)),
                "crossfit_positive_repeat_fraction": float(np.mean(np.asarray(repeat_joint) > 0)),
                "oracle_minus_crossfit_gap_percent": float(
                    full_joint[full_index] - np.mean(repeat_joint)
                ),
            }
            summary_rows.append(row)
            print(
                f"{global_probe:2d} {bank:8s} {full_joint[full_index]:7.3f}% "
                f"{np.mean(repeat_endpoint):8.3f}% {np.mean(repeat_trajectory):8.3f}% "
                f"{np.mean(repeat_joint):9.3f}% {np.std(repeat_joint, ddof=1):7.3f}% "
                f"{np.mean(np.asarray(repeat_joint) > 0):7.3f} "
                f"{full_joint[full_index] - np.mean(repeat_joint):8.3f}%",
                flush=True,
            )

    ranking = sorted(
        summary_rows,
        key=lambda row: float(row["crossfit_cf_joint_mean_percent"]),
        reverse=True,
    )
    out_dir = args.out_dir or (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "eval"
        / "q0_probe24_timestamp_sign_crossfit"
        / f"source_run_{args.source_run_id}"
    )
    write_csv(out_dir / "per_split.csv", split_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "ranking.csv", ranking)
    print("\nTOP HELD-OUT TIMESTAMP-SIGN CROSSFIT")
    for rank, row in enumerate(ranking[:10], start=1):
        print(
            f"{rank:2d}. P{int(row['global_probe']):02d} "
            f"CV={float(row['crossfit_cf_joint_mean_percent']):+.3f}% "
            f"±{float(row['crossfit_cf_joint_std_percent']):.3f}% "
            f"positive={float(row['crossfit_positive_repeat_fraction']):.2f} "
            f"full={float(row['full_oracle_cf_joint_lds_percent']):.3f}%"
        )
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
