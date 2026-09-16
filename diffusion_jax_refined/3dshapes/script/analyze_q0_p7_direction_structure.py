#!/usr/bin/env python3
"""Analyze Q0 P7 pullback-direction structure across 50 checkpoints x 10 timestamps."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT / "script") not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT / "script"))

from analyze_predicted_noise_old_fresh_term_signs import BANKS, write_csv
from run_predicted_noise_jvp_l2_squared import query_artifact_path


def parse_sign_label(text: str) -> dict[int, int]:
    result = {}
    for item in text.split(","):
        timestep, sign = item.split(":", 1)
        result[int(timestep)] = 1 if sign.strip() == "+" else -1
    return result


def load_signs(args, timesteps):
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
            if int(row["global_probe"]) == args.probe
            and int(row["repeat"]) == args.repeat
            and int(row["train_fold"]) == args.train_fold
        ]
    if len(matches) != 1:
        raise ValueError(f"expected one sign row, found {len(matches)} in {path}")
    lookup = parse_sign_label(matches[0]["selected_signs"])
    return np.asarray([lookup[int(value)] for value in timesteps], dtype=np.int8)


def vector_stats(cosines):
    values = np.asarray(cosines, dtype=np.float64)
    return {
        "mean_cosine": float(values.mean()),
        "std_cosine": float(values.std()),
        "mean_abs_cosine": float(np.abs(values).mean()),
        "positive_fraction": float(np.mean(values > 0.0)),
        "minimum_cosine": float(values.min()),
        "maximum_cosine": float(values.max()),
    }


def offdiagonal(matrix):
    mask = ~np.eye(len(matrix), dtype=bool)
    return matrix[mask]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-id", type=int, default=0)
    parser.add_argument("--probe", type=int, default=7)
    parser.add_argument("--source-run-id", default="3506389")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--train-fold", type=int, choices=(0, 1), default=0)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.query_id != 0 or not 1 <= args.probe <= 12:
        raise ValueError("this focused audit expects Q0 and an old12 probe")

    artifact = query_artifact_path(
        args.experiment,
        args.train_seed,
        args.epochs,
        args.query_id,
        num_probes=12,
        probe_index=args.probe - 1,
        query_namespace_pattern=BANKS["old12"]["query_pattern"],
    )
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    with np.load(artifact, allow_pickle=False) as payload:
        directions = np.asarray(payload["query_features"], dtype=np.float64)
        ckpt_indices = np.asarray(payload["ckpt_indices"], dtype=np.int32)
        timesteps_flat = np.asarray(payload["timesteps"], dtype=np.int32)
    directions = directions / np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
    )
    signs = load_signs(args, timesteps_flat)
    oriented = directions * signs[:, None]

    checkpoints = list(dict.fromkeys(int(value) for value in ckpt_indices))
    timesteps = list(dict.fromkeys(int(value) for value in timesteps_flat))
    if len(checkpoints) != 50 or len(timesteps) != 10:
        raise ValueError(
            f"expected 50 checkpoints x 10 timestamps; got {len(checkpoints)} x {len(timesteps)}"
        )
    lookup = {
        (int(checkpoint), int(timestep)): term
        for term, (checkpoint, timestep) in enumerate(zip(ckpt_indices, timesteps_flat))
    }
    grid = np.stack(
        [
            np.stack([oriented[lookup[(checkpoint, timestep)]] for timestep in timesteps])
            for checkpoint in checkpoints
        ]
    )

    timestamp_rows = []
    for tslot, timestep in enumerate(timesteps):
        values = grid[:, tslot, :]
        adjacent = np.einsum("cd,cd->c", values[:-1], values[1:])
        gram = values @ values.T
        resultant = float(np.linalg.norm(values.mean(axis=0)))
        row = {
            "timestep": timestep,
            **{f"adjacent_checkpoint_{key}": value for key, value in vector_stats(adjacent).items()},
            **{f"all_checkpoint_{key}": value for key, value in vector_stats(offdiagonal(gram)).items()},
            "checkpoint_resultant": resultant,
            "first_last_cosine": float(values[0] @ values[-1]),
        }
        for lag in (1, 2, 5, 10, 20):
            lag_values = np.einsum("cd,cd->c", values[:-lag], values[lag:])
            row[f"lag_{lag}_mean_cosine"] = float(lag_values.mean())
            row[f"lag_{lag}_mean_abs_cosine"] = float(np.abs(lag_values).mean())
        timestamp_rows.append(row)

    checkpoint_rows = []
    for cslot, checkpoint in enumerate(checkpoints):
        values = grid[cslot]
        adjacent = np.einsum("td,td->t", values[:-1], values[1:])
        gram = values @ values.T
        checkpoint_rows.append(
            {
                "checkpoint_ordinal": cslot + 1,
                "checkpoint_index": checkpoint,
                "epoch": (cslot + 1) * 4,
                **{f"adjacent_timestamp_{key}": value for key, value in vector_stats(adjacent).items()},
                **{f"all_timestamp_{key}": value for key, value in vector_stats(offdiagonal(gram)).items()},
                "timestamp_resultant": float(np.linalg.norm(values.mean(axis=0))),
                "t999_t0_cosine": float(values[0] @ values[-1]),
            }
        )

    adjacent_checkpoint_rows = []
    for cslot in range(49):
        for tslot, timestep in enumerate(timesteps):
            adjacent_checkpoint_rows.append(
                {
                    "left_checkpoint_ordinal": cslot + 1,
                    "right_checkpoint_ordinal": cslot + 2,
                    "left_epoch": (cslot + 1) * 4,
                    "right_epoch": (cslot + 2) * 4,
                    "timestep": timestep,
                    "cosine": float(grid[cslot, tslot] @ grid[cslot + 1, tslot]),
                }
            )

    gram = oriented @ oriented.T
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    fractions = eigenvalues / total
    effective_rank = float(total * total / np.sum(eigenvalues * eigenvalues))
    full_offdiag = offdiagonal(gram)
    full_resultant = float(np.linalg.norm(oriented.mean(axis=0)))

    out_dir = args.out_dir or (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "eval"
        / "q0_p7_direction_structure"
        / f"source_run_{args.source_run_id}"
        / f"repeat_{args.repeat}_fold_{args.train_fold}"
    )
    write_csv(out_dir / "per_timestamp.csv", timestamp_rows)
    write_csv(out_dir / "per_checkpoint.csv", checkpoint_rows)
    write_csv(out_dir / "adjacent_checkpoints.csv", adjacent_checkpoint_rows)
    write_csv(
        out_dir / "spectrum.csv",
        [
            {
                "rank": rank + 1,
                "eigenvalue": float(value),
                "explained_fraction": float(fractions[rank]),
                "cumulative_fraction": float(fractions[: rank + 1].sum()),
            }
            for rank, value in enumerate(eigenvalues)
        ],
    )

    print("Q0 P7 FLIPPED PULLBACK-DIRECTION STRUCTURE")
    print(f"artifact={artifact}")
    print(
        f"full 500-term resultant={full_resultant:.6f} "
        f"random baseline={1/np.sqrt(500):.6f}"
    )
    full_stats = vector_stats(full_offdiag)
    print(
        f"full offdiag cosine mean={full_stats['mean_cosine']:+.6f} "
        f"|cos|={full_stats['mean_abs_cosine']:.6f} "
        f"positive={full_stats['positive_fraction']:.3f}"
    )
    print(
        f"effective rank={effective_rank:.2f}; "
        f"top1={fractions[0]*100:.2f}% top5={fractions[:5].sum()*100:.2f}% "
        f"top10={fractions[:10].sum()*100:.2f}%"
    )
    print("\nFIXED TIMESTAMP, ACROSS CHECKPOINTS")
    print(f"{'T':>4s} {'ADJ COS':>9s} {'ADJ |C|':>9s} {'ADJ+':>7s} {'ALL COS':>9s} {'ALL |C|':>9s} {'R50':>8s} {'FIRST~LAST':>11s}")
    print("-" * 86)
    for row in timestamp_rows:
        print(
            f"{int(row['timestep']):4d} "
            f"{float(row['adjacent_checkpoint_mean_cosine']):+9.4f} "
            f"{float(row['adjacent_checkpoint_mean_abs_cosine']):9.4f} "
            f"{float(row['adjacent_checkpoint_positive_fraction']):7.3f} "
            f"{float(row['all_checkpoint_mean_cosine']):+9.4f} "
            f"{float(row['all_checkpoint_mean_abs_cosine']):9.4f} "
            f"{float(row['checkpoint_resultant']):8.4f} "
            f"{float(row['first_last_cosine']):+11.4f}"
        )
    print("\nSELECTED CHECKPOINTS, ACROSS TIMESTAMPS")
    print(f"{'CKPT':>4s} {'EPOCH':>5s} {'ADJ COS':>9s} {'ADJ |C|':>9s} {'ADJ+':>7s} {'ALL COS':>9s} {'R10':>8s} {'999~0':>9s}")
    print("-" * 78)
    for ordinal in (1, 10, 20, 30, 40, 50):
        row = checkpoint_rows[ordinal - 1]
        print(
            f"{ordinal:4d} {ordinal*4:5d} "
            f"{float(row['adjacent_timestamp_mean_cosine']):+9.4f} "
            f"{float(row['adjacent_timestamp_mean_abs_cosine']):9.4f} "
            f"{float(row['adjacent_timestamp_positive_fraction']):7.3f} "
            f"{float(row['all_timestamp_mean_cosine']):+9.4f} "
            f"{float(row['timestamp_resultant']):8.4f} "
            f"{float(row['t999_t0_cosine']):+9.4f}"
        )
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
