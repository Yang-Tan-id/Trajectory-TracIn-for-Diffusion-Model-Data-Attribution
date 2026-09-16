#!/usr/bin/env python3
"""Compare a crossfit-oriented probe with next/reference noise deltas."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT / "script") not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT / "script"))

from analyze_nearest_train_probe_predicted_noise_relation import (
    combine_probe_query_features_mc,
)
from analyze_predicted_noise_old_fresh_term_signs import write_csv
from analyze_predicted_noise_probe24_output_alignment import load_bank
from analyze_predicted_noise_probe24_term_winners import (
    FRESH_PATTERN,
    ORIGINAL_PATTERN,
    load_features,
)


DELTA_SPECS = {
    "next_delta": (
        "cosine_to_next_predicted_noise_delta",
        "projection_on_next_predicted_noise_delta",
    ),
    "reference_delta": (
        "cosine_to_reference_predicted_noise_delta",
        "projection_on_reference_predicted_noise_delta",
    ),
    "reference_direction_delta": (
        "cosine_to_reference_predicted_noise_direction_delta",
        "projection_on_reference_predicted_noise_direction_delta",
    ),
}


def unit(vector):
    value = np.asarray(vector, dtype=np.float64)
    return value / max(float(np.linalg.norm(value)), 1e-12)


def parse_sign_label(text: str) -> dict[int, int]:
    result = {}
    for item in text.split(","):
        timestep, sign = item.split(":", 1)
        result[int(timestep)] = 1 if sign.strip() == "+" else -1
    return result


def load_probe_signs(args):
    path = (
        args.crossfit_dir / "per_split.csv"
        if args.crossfit_dir is not None
        else SHAPES_ROOT
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
            if int(row["global_probe"]) == args.global_probe
            and int(row["repeat"]) == args.repeat
            and int(row["train_fold"]) == args.train_fold
        ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one P{args.global_probe} sign row in {path}, "
            f"found {len(matches)}"
        )
    return parse_sign_label(matches[0]["selected_signs"])


def collect_alignment(alignments, checkpoint, timestep):
    rows = []
    for global_probe in range(24):
        bank = "original" if global_probe < 12 else "fresh"
        local_probe = global_probe + 1 if global_probe < 12 else global_probe - 11
        rows.append(alignments[bank][(checkpoint, timestep, local_probe)])
    return rows


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "mean_abs": float(np.abs(array).mean()),
        "positive_fraction": float(np.mean(array > 0.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-id", type=int, default=0)
    parser.add_argument("--global-probe", type=int, default=7)
    parser.add_argument("--source-run-id", default="3506389")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--train-fold", type=int, choices=(0, 1), default=0)
    parser.add_argument("--crossfit-dir", type=Path)
    parser.add_argument(
        "--original-namespace",
        default="predicted_noise_output_reference_delta_original12",
    )
    parser.add_argument(
        "--fresh-namespace",
        default="predicted_noise_output_reference_delta_fresh12",
    )
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if not 1 <= args.global_probe <= 24:
        raise ValueError("--global-probe must be in [1, 24]")
    probe_slot = args.global_probe - 1
    args.query_ids = [args.query_id]
    args.checkpoint_direction = "next"

    print("[phase 1/3] load 24 saved query pullbacks", flush=True)
    original, metadata = load_features(args, ORIGINAL_PATTERN, range(12))
    fresh, fresh_metadata = load_features(args, FRESH_PATTERN, range(12))
    for key, value in metadata.items():
        if not np.array_equal(value, fresh_metadata[key]):
            raise ValueError(f"old/fresh metadata mismatch: {key}")
    query_features = np.concatenate((original[:, 0], fresh[:, 0]), axis=0).astype(
        np.float64
    )
    if query_features.shape != (24, 500, 4096):
        raise ValueError(f"unexpected query feature shape: {query_features.shape}")

    print("[phase 2/3] load cached output-space delta projections", flush=True)
    alignments = {
        "original": load_bank(args, args.query_id, "original"),
        "fresh": load_bank(args, args.query_id, "fresh"),
    }
    probe_signs = load_probe_signs(args)
    term_lookup = {
        (int(checkpoint) + 1, int(timestep)): term
        for term, (checkpoint, timestep) in enumerate(
            zip(metadata["ckpt_indices"], metadata["timesteps"])
        )
    }

    rows = []
    for checkpoint in range(1, 50):
        for timestep in list(dict.fromkeys(int(x) for x in metadata["timesteps"])):
            term = term_lookup[(checkpoint, timestep)]
            values = collect_alignment(alignments, checkpoint, timestep)
            sign = probe_signs[timestep]
            selected_probe = sign * unit(query_features[probe_slot, term])
            row = {
                "checkpoint": checkpoint,
                "epoch": checkpoint * 4,
                "timestep": timestep,
                "probe_timestamp_sign": sign,
            }
            for name, (cosine_key, scalar_key) in DELTA_SPECS.items():
                if cosine_key not in values[0] or scalar_key not in values[0]:
                    continue
                scalars = np.asarray(
                    [entry[scalar_key] for entry in values], dtype=np.float64
                )
                full = combine_probe_query_features_mc(
                    query_features[:, term], scalars
                )
                keep = np.arange(24) != probe_slot
                leave_probe_out = combine_probe_query_features_mc(
                    query_features[keep, term], scalars[keep]
                )
                row[f"output_cosine_{name}"] = sign * float(
                    values[probe_slot][cosine_key]
                )
                row[f"pullback_cosine_{name}_mc24"] = float(
                    selected_probe @ unit(full)
                )
                row[f"pullback_cosine_{name}_mc23_leave_probe_out"] = float(
                    selected_probe @ unit(leave_probe_out)
                )
            rows.append(row)

    if len(rows) != 490:
        raise ValueError(f"expected 490 next-checkpoint terms, got {len(rows)}")

    summary_rows = []
    timestamp_rows = []
    metric_keys = [
        key
        for key in rows[0]
        if key.endswith("mc24")
        or "leave_probe_out" in key
        or key.startswith("output_cosine_")
    ]
    for key in metric_keys:
        stats = summarize([row[key] for row in rows])
        summary_rows.append({"metric": key, **stats})
        for timestep in sorted({int(row["timestep"]) for row in rows}, reverse=True):
            selected = [row[key] for row in rows if int(row["timestep"]) == timestep]
            timestamp_rows.append({"metric": key, "timestep": timestep, **summarize(selected)})

    out_dir = args.out_dir or (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "eval"
        / f"q{args.query_id}_p{args.global_probe}_query_specific_directions"
        / f"source_run_{args.source_run_id}"
        / f"repeat_{args.repeat}_fold_{args.train_fold}"
    )
    write_csv(out_dir / "per_term.csv", rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "per_timestamp.csv", timestamp_rows)

    print(
        f"[phase 3/3] Q{args.query_id} P{args.global_probe} "
        "VS QUERY-SPECIFIC NOISE DIRECTIONS"
    )
    print(f"{'METRIC':58s} {'MEAN':>9s} {'|COS|':>9s} {'COS+':>7s}")
    print("-" * 88)
    for row in summary_rows:
        print(
            f"{str(row['metric']):58s} {float(row['mean']):+9.5f} "
            f"{float(row['mean_abs']):9.5f} {float(row['positive_fraction']):7.3f}"
        )

    print(f"\nLEAVE-P{args.global_probe}-OUT PULLBACK COSINE BY TIMESTAMP")
    print(f"{'T':>4s} {'NEXT':>9s} {'|NEXT|':>9s} {'N+':>6s} {'REF':>9s} {'|REF|':>9s} {'R+':>6s}")
    print("-" * 68)
    next_key = "pullback_cosine_next_delta_mc23_leave_probe_out"
    reference_key = "pullback_cosine_reference_delta_mc23_leave_probe_out"
    if next_key not in rows[0]:
        raise ValueError(f"required next-delta metric is unavailable: {next_key}")
    reference_available = reference_key in rows[0]
    for timestep in sorted({int(row["timestep"]) for row in rows}, reverse=True):
        next_values = [row[next_key] for row in rows if row["timestep"] == timestep]
        ns = summarize(next_values)
        rs = (
            summarize(
                [row[reference_key] for row in rows if row["timestep"] == timestep]
            )
            if reference_available
            else {"mean": float("nan"), "mean_abs": float("nan"), "positive_fraction": float("nan")}
        )
        print(
            f"{timestep:4d} {ns['mean']:+9.5f} {ns['mean_abs']:9.5f} "
            f"{ns['positive_fraction']:6.3f} {rs['mean']:+9.5f} "
            f"{rs['mean_abs']:9.5f} {rs['positive_fraction']:6.3f}"
        )

    print(f"\nSELECTED CHECKPOINTS — LEAVE-P{args.global_probe}-OUT")
    print(f"{'CKPT':>4s} {'EPOCH':>5s} {'T':>4s} {'NEXT':>9s} {'REF':>9s}")
    print("-" * 44)
    for checkpoint in (1, 10, 20, 30, 40):
        for row in rows:
            if row["checkpoint"] == checkpoint:
                print(
                    f"{checkpoint:4d} {checkpoint*4:5d} {int(row['timestep']):4d} "
                    f"{float(row[next_key]):+9.5f} "
                    f"{float(row.get(reference_key, float('nan'))):+9.5f}"
                )
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
