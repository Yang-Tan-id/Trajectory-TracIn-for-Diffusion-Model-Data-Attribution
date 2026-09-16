#!/usr/bin/env python3
"""Flip P10 terms whose projected direction opposes flipped P7, then audit LDS."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT / "script") not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT / "script"))

from analyze_predicted_noise_old_fresh_term_signs import write_csv
from analyze_q0_probe500_component_sign_oracle import cf_lds, target_data


def parse_sign_label(text: str) -> dict[int, int]:
    result = {}
    for item in text.split(","):
        timestep, sign = item.split(":", 1)
        result[int(timestep)] = 1 if sign.strip() == "+" else -1
    return result


def load_timestamp_signs(args, probe: int, timesteps: np.ndarray) -> np.ndarray:
    path = (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "eval"
        / "q0_probe24_timestamp_sign_crossfit"
        / f"source_run_{args.source_run_id}"
        / "per_split.csv"
    )
    with path.open(newline="") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if int(row["global_probe"]) == probe
            and int(row["repeat"]) == args.repeat
            and int(row["train_fold"]) == args.train_fold
        ]
    if len(matches) != 1:
        raise ValueError(f"expected one sign row for P{probe}; found {len(matches)}")
    lookup = parse_sign_label(matches[0]["selected_signs"])
    return np.asarray([lookup[int(value)] for value in timesteps], dtype=np.int8)


def load_term_predictions(args):
    root = (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "eval"
        / "q0_probe500_component_sign_oracle"
        / f"source_run_{args.source_run_id}"
    )
    payloads = []
    for shard in range(2):
        path = root / f"term_prediction_shard_{shard}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        payloads.append(np.load(path, allow_pickle=False))
    used = sum(np.asarray(payload["used"], dtype=np.int8) for payload in payloads)
    if not np.all(used == 1):
        raise ValueError(f"each term must occur once; counts={np.unique(used)}")
    predictions = sum(
        np.asarray(payload["predictions"], dtype=np.float64) for payload in payloads
    )
    score_indices = np.asarray(payloads[0]["score_indices"], dtype=np.int64)
    ckpt_indices = np.asarray(payloads[0]["ckpt_indices"], dtype=np.int32)
    timesteps = np.asarray(payloads[0]["timesteps"], dtype=np.int32)
    for payload in payloads[1:]:
        if not np.array_equal(score_indices, payload["score_indices"]):
            raise ValueError("score indices differ between shards")
    return predictions, score_indices, ckpt_indices, timesteps


def load_flipped_pair_cosines(args) -> np.ndarray:
    path = (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "eval"
        / "q0_three_probe_raw_projected_overlap"
        / f"source_run_{args.source_run_id}"
        / f"repeat_{args.repeat}_fold_{args.train_fold}_p7_p10_p12"
        / "per_term.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["term"]))
    if len(rows) != 500:
        raise ValueError(f"expected 500 cosine rows, got {len(rows)}")
    return np.asarray(
        [float(row["projected_p7_p10_cosine"]) for row in rows],
        dtype=np.float64,
    )


def repeated_split(args, num_models: int):
    rng = np.random.default_rng(args.random_seed)
    folds = None
    for _ in range(args.repeat + 1):
        permutation = rng.permutation(num_models)
        folds = (permutation[::2], permutation[1::2])
    assert folds is not None
    return folds[args.train_fold], folds[1 - args.train_fold]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--query-id", type=int, default=0)
    parser.add_argument("--source-run-id", default="3506389")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--train-fold", type=int, choices=(0, 1), default=0)
    parser.add_argument("--random-seed", type=int, default=20260916)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.query_id != 0:
        raise ValueError("this focused audit expects Q0")

    predictions, score_indices, ckpt_indices, timesteps = load_term_predictions(args)
    _, truth = target_data(args, score_indices)
    endpoint = truth["endpoint_contarfactual"]
    trajectory = truth["traj_contarfactual"]
    c_indices, d_indices = repeated_split(args, len(endpoint))

    p7_signs = load_timestamp_signs(args, 7, timesteps)
    p10_signs = load_timestamp_signs(args, 10, timesteps)
    flipped_cosines = load_flipped_pair_cosines(args)
    oppose = flipped_cosines < 0.0
    align_signs = p10_signs * np.where(oppose, -1, 1).astype(np.int8)

    methods = {
        "p10_all_plus": np.ones(500, dtype=np.int8),
        "p10_c_timestamp_signs": p10_signs,
        "p10_align_to_flipped_p7": align_signs,
        "p10_opposite_flipped_p7": -align_signs,
        "p7_c_timestamp_signs_reference": p7_signs,
    }
    probe_for_method = {
        name: 6 if name.startswith("p7_") else 9 for name in methods
    }
    splits = {
        "all192": np.arange(len(endpoint)),
        "c_train96": c_indices,
        "d_heldout96": d_indices,
    }
    rows = []
    print("Q0 P10 COMPONENT ALIGNMENT TO FLIPPED P7 — BOTH-L2")
    print(
        f"negative projected-cosine terms={int(oppose.sum())}/500; "
        f"repeat={args.repeat} train_fold={args.train_fold}"
    )
    print(f"{'METHOD':34s} {'SPLIT':12s} {'ENDPOINT':>10s} {'TRAJ':>10s} {'CF JOINT':>10s}")
    print("-" * 84)
    for method, signs in methods.items():
        prediction = signs.astype(np.float64) @ predictions[probe_for_method[method]]
        for split, indices in splits.items():
            end, traj, joint = cf_lds(
                prediction[indices], endpoint[indices], trajectory[indices]
            )
            row = {
                "method": method,
                "split": split,
                "endpoint_lds_percent": float(end[0]),
                "traj_lds_percent": float(traj[0]),
                "cf_joint_lds_percent": float(joint[0]),
                "positive_term_fraction": float(np.mean(signs > 0)),
            }
            rows.append(row)
            print(
                f"{method:34s} {split:12s} {end[0]:9.3f}% "
                f"{traj[0]:9.3f}% {joint[0]:9.3f}%"
            )

    term_rows = []
    for term in range(500):
        term_rows.append(
            {
                "term": term,
                "checkpoint_ordinal": term // 10 + 1,
                "checkpoint_index": int(ckpt_indices[term]),
                "timestep": int(timesteps[term]),
                "flipped_p7_p10_projected_cosine": float(flipped_cosines[term]),
                "negative_cosine": int(oppose[term]),
                "p7_timestamp_sign": int(p7_signs[term]),
                "p10_timestamp_sign": int(p10_signs[term]),
                "p10_aligned_sign": int(align_signs[term]),
            }
        )

    out_dir = args.out_dir or (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "eval"
        / "q0_p10_align_negative_p7_components"
        / f"source_run_{args.source_run_id}"
        / f"repeat_{args.repeat}_fold_{args.train_fold}"
    )
    write_csv(out_dir / "lds.csv", rows)
    write_csv(out_dir / "per_term_signs.csv", term_rows)
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
