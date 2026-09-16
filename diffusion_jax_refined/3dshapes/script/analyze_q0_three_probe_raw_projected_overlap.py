#!/usr/bin/env python3
"""Compare three flipped probes before and after the saved query pullback."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT / "script") not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT / "script"))

from analyze_predicted_noise_old_fresh_term_signs import BANKS, write_csv
from analyze_predicted_noise_probe_independence import predicted_noise_probe_key
from run_predicted_noise_jvp_l2_squared import query_artifact_path


def parse_probes(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if len(values) != 3 or len(set(values)) != 3:
        raise argparse.ArgumentTypeError("expected three distinct probes")
    if any(value < 1 or value > 12 for value in values):
        raise argparse.ArgumentTypeError("this audit expects three old12 probes")
    return values


def parse_sign_label(text: str) -> dict[int, int]:
    result = {}
    for item in text.split(","):
        timestep, sign = item.split(":", 1)
        result[int(timestep)] = 1 if sign.strip() == "+" else -1
    return result


def load_selected_signs(args) -> dict[int, dict[int, int]]:
    path = (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "eval"
        / "q0_probe24_timestamp_sign_crossfit"
        / f"source_run_{args.source_run_id}"
        / "per_split.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    selected = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            probe = int(row["global_probe"])
            if (
                probe in args.probes
                and int(row["repeat"]) == args.repeat
                and int(row["train_fold"]) == args.train_fold
            ):
                selected[probe] = parse_sign_label(row["selected_signs"])
    if set(selected) != set(args.probes):
        raise ValueError(f"missing selected signs for probes {args.probes}: {path}")
    return selected


def load_projected(args):
    values = []
    paths = []
    reference = None
    for probe in args.probes:
        path = query_artifact_path(
            args.experiment,
            args.train_seed,
            args.epochs,
            args.query_id,
            num_probes=12,
            probe_index=probe - 1,
            query_namespace_pattern=BANKS["old12"]["query_pattern"],
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            feature = np.asarray(payload["query_features"], dtype=np.float64)
            metadata = {
                "ckpt_indices": np.asarray(payload["ckpt_indices"], dtype=np.int32),
                "timesteps": np.asarray(payload["timesteps"], dtype=np.int32),
                "snapshot_positions": np.asarray(
                    payload["snapshot_positions"], dtype=np.int32
                ),
            }
        feature /= np.maximum(np.linalg.norm(feature, axis=1, keepdims=True), 1e-12)
        values.append(feature)
        paths.append(str(path))
        if reference is None:
            reference = metadata
        else:
            for key, expected in reference.items():
                if not np.array_equal(metadata[key], expected):
                    raise ValueError(f"metadata differ for {path}:{key}")
    assert reference is not None
    projected = np.stack(values, axis=0)
    if projected.shape != (3, 500, 4096):
        raise ValueError(f"expected projected features (3,500,4096), got {projected.shape}")
    return projected, reference, paths


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "mean_abs": float(np.abs(values).mean()),
        "positive_fraction": float(np.mean(values > 0.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-id", type=int, default=0)
    parser.add_argument("--source-run-id", default="3506389")
    parser.add_argument("--probes", type=parse_probes, default=parse_probes("7,10,12"))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--train-fold", type=int, choices=(0, 1), default=0)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    signs = load_selected_signs(args)
    projected, metadata, artifact_paths = load_projected(args)
    dimension = 64 * 64 * 3
    raw = np.empty((3, 500, dimension), dtype=np.float32)
    generate = jax.jit(
        jax.vmap(lambda key: jax.random.normal(key, (1, 64, 64, 3), dtype=jnp.float32))
    )
    for term, (checkpoint, timestep, position) in enumerate(
        zip(
            metadata["ckpt_indices"],
            metadata["timesteps"],
            metadata["snapshot_positions"],
        )
    ):
        keys = jnp.stack(
            [
                predicted_noise_probe_key(
                    args.train_seed,
                    int(checkpoint),
                    int(timestep),
                    int(position),
                    probe - 1,
                )
                for probe in args.probes
            ]
        )
        generated = np.asarray(jax.device_get(generate(keys)), dtype=np.float32).reshape(
            3, dimension
        )
        generated /= np.maximum(np.linalg.norm(generated, axis=1, keepdims=True), 1e-12)
        raw[:, term] = generated
        if (term + 1) % 50 == 0:
            print(f"[raw probes] terms={term + 1}/500", flush=True)

    term_signs = np.asarray(
        [
            [signs[probe][int(timestep)] for timestep in metadata["timesteps"]]
            for probe in args.probes
        ],
        dtype=np.float64,
    )
    raw_flipped = raw.astype(np.float64) * term_signs[:, :, None]
    projected_flipped = projected * term_signs[:, :, None]
    pairs = list(combinations(range(3), 2))
    raw_pair = {
        pair: np.einsum("td,td->t", raw_flipped[pair[0]], raw_flipped[pair[1]])
        for pair in pairs
    }
    projected_pair = {
        pair: np.einsum(
            "td,td->t", projected_flipped[pair[0]], projected_flipped[pair[1]]
        )
        for pair in pairs
    }
    raw_resultant = np.linalg.norm(raw_flipped.mean(axis=0), axis=1)
    projected_resultant = np.linalg.norm(projected_flipped.mean(axis=0), axis=1)

    per_term = []
    checkpoint_values = list(dict.fromkeys(int(x) for x in metadata["ckpt_indices"]))
    checkpoint_slots = {value: slot for slot, value in enumerate(checkpoint_values)}
    for term, (checkpoint, timestep, position) in enumerate(
        zip(
            metadata["ckpt_indices"],
            metadata["timesteps"],
            metadata["snapshot_positions"],
        )
    ):
        row = {
            "term": term,
            "checkpoint_ordinal": checkpoint_slots[int(checkpoint)] + 1,
            "checkpoint_index": int(checkpoint),
            "timestep": int(timestep),
            "snapshot_position": int(position),
            "raw_three_probe_resultant": float(raw_resultant[term]),
            "projected_three_probe_resultant": float(projected_resultant[term]),
        }
        for left, right in pairs:
            label = f"p{args.probes[left]}_p{args.probes[right]}"
            row[f"raw_{label}_cosine"] = float(raw_pair[(left, right)][term])
            row[f"projected_{label}_cosine"] = float(
                projected_pair[(left, right)][term]
            )
        per_term.append(row)

    per_timestamp = []
    timesteps = list(dict.fromkeys(int(x) for x in metadata["timesteps"]))
    for timestep in timesteps:
        mask = metadata["timesteps"] == timestep
        for left, right in pairs:
            label = f"P{args.probes[left]}/P{args.probes[right]}"
            row = {
                "timestep": timestep,
                "pair": label,
                **{f"raw_{key}": value for key, value in stats(raw_pair[(left, right)][mask]).items()},
                **{
                    f"projected_{key}": value
                    for key, value in stats(projected_pair[(left, right)][mask]).items()
                },
                "raw_three_probe_resultant_mean": float(raw_resultant[mask].mean()),
                "projected_three_probe_resultant_mean": float(
                    projected_resultant[mask].mean()
                ),
            }
            per_timestamp.append(row)

    summary = []
    for left, right in pairs:
        summary.append(
            {
                "pair": f"P{args.probes[left]}/P{args.probes[right]}",
                **{f"raw_{key}": value for key, value in stats(raw_pair[(left, right)]).items()},
                **{
                    f"projected_{key}": value
                    for key, value in stats(projected_pair[(left, right)]).items()
                },
            }
        )

    if args.out_dir is None:
        out_dir = (
            SHAPES_ROOT
            / "result"
            / args.experiment
            / "eval"
            / "q0_three_probe_raw_projected_overlap"
            / f"source_run_{args.source_run_id}"
            / f"repeat_{args.repeat}_fold_{args.train_fold}_p{'_p'.join(str(value) for value in args.probes)}"
        )
    else:
        out_dir = args.out_dir
    write_csv(out_dir / "per_term.csv", per_term)
    write_csv(out_dir / "per_timestamp.csv", per_timestamp)
    write_csv(out_dir / "summary.csv", summary)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "query": args.query_id,
                "probes": args.probes,
                "repeat": args.repeat,
                "train_fold": args.train_fold,
                "signs": signs,
                "query_artifacts": artifact_paths,
                "raw_dimension": dimension,
                "projected_dimension": int(projected.shape[-1]),
                "raw_random_cosine_sd": 1.0 / math.sqrt(dimension),
                "projected_isotropic_cosine_sd": 1.0 / math.sqrt(projected.shape[-1]),
                "independent_three_probe_resultant_baseline": 1.0 / math.sqrt(3.0),
            },
            indent=2,
        )
    )

    print("Q0 THREE-PROBE OVERLAP — C->D TIMESTAMP SIGNS")
    print(f"probes={args.probes} repeat={args.repeat} train_fold={args.train_fold}")
    print(f"{'PAIR':9s} {'RAW MEAN':>10s} {'RAW |COS|':>11s} {'PJ MEAN':>10s} {'PJ |COS|':>10s} {'PJ+':>7s}")
    print("-" * 68)
    for row in summary:
        print(
            f"{row['pair']:9s} {row['raw_mean']:+10.6f} {row['raw_mean_abs']:11.6f} "
            f"{row['projected_mean']:+10.6f} {row['projected_mean_abs']:10.6f} "
            f"{row['projected_positive_fraction']:7.3f}"
        )
    print("\nPER TIMESTAMP")
    print(f"{'T':>4s} {'PAIR':9s} {'RAW':>9s} {'|RAW|':>9s} {'PJ':>9s} {'|PJ|':>9s} {'PJ+':>6s} {'R3 RAW':>9s} {'R3 PJ':>9s}")
    print("-" * 92)
    for row in per_timestamp:
        print(
            f"{row['timestep']:4d} {row['pair']:9s} "
            f"{row['raw_mean']:+9.5f} {row['raw_mean_abs']:9.5f} "
            f"{row['projected_mean']:+9.5f} {row['projected_mean_abs']:9.5f} "
            f"{row['projected_positive_fraction']:6.3f} "
            f"{row['raw_three_probe_resultant_mean']:9.5f} "
            f"{row['projected_three_probe_resultant_mean']:9.5f}"
        )
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
