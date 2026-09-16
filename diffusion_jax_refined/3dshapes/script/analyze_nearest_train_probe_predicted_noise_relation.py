#!/usr/bin/env python3
"""Analyze output-space noise alignment of per-datapoint nearest probes."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from analyze_predicted_noise_probe24_output_alignment import load_bank
from analyze_predicted_noise_probe24_term_winners import (
    FRESH_PATTERN,
    ORIGINAL_PATTERN,
    load_features,
    unit_rows,
)
from analyze_predicted_noise_probe8_choose4 import spearman
from run_predicted_noise_jvp_l2_squared import train_part_dir


ALIGNMENT_KEYS = (
    "cosine_to_current_predicted_noise",
    "cosine_to_next_predicted_noise",
    "cosine_to_next_predicted_noise_delta",
)
METHODS = ("nearest_direction", "nearest_axis_signed")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def selected_probe_indices(scores: np.ndarray, method: str) -> np.ndarray:
    if method == "nearest_direction":
        return np.argmax(scores, axis=1)
    if method == "nearest_axis_signed":
        return np.argmax(np.abs(scores), axis=1)
    raise ValueError(method)


def probe_alignment_matrix(
    alignments: dict[tuple[int, str], dict[tuple[int, int, int], dict[str, float]]],
    query_id: int,
    checkpoint: int,
    timestep: int,
) -> dict[str, np.ndarray]:
    result = {key: [] for key in ALIGNMENT_KEYS}
    for probe in range(24):
        bank = "original" if probe < 12 else "fresh"
        bank_probe = probe + 1 if probe < 12 else probe - 11
        values = alignments[(query_id, bank)][(checkpoint, timestep, bank_probe)]
        for key in ALIGNMENT_KEYS:
            result[key].append(values[key])
    return {key: np.asarray(values, dtype=np.float64) for key, values in result.items()}


def analyze_shard(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp

    original, meta = load_features(args, ORIGINAL_PATTERN, range(12))
    fresh, fresh_meta = load_features(args, FRESH_PATTERN, range(12))
    for key in meta:
        if not np.array_equal(meta[key], fresh_meta[key]):
            raise ValueError(f"original/fresh metadata mismatch for {key}")
    probes = np.concatenate((original, fresh), axis=0)
    probe_lookup = {
        (int(ckpt), int(timestep)): term
        for term, (ckpt, timestep) in enumerate(
            zip(meta["ckpt_indices"], meta["timesteps"])
        )
    }
    alignments = {
        (query_id, bank): load_bank(args, query_id, bank)
        for query_id in args.query_ids
        for bank in ("original", "fresh")
    }

    rows: list[dict[str, object]] = []
    for ckpt_i in range(args.shard_index, 49, args.shard_count):
        path = train_part_dir(args.experiment, args.train_seed) / f"ckpt_{ckpt_i:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            train = np.asarray(payload["train_features"], dtype=np.float32)
            ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            timesteps = np.asarray(payload["timesteps"], dtype=np.int32)

        for local_term, (ckpt, timestep) in enumerate(zip(ckpts, timesteps)):
            term = probe_lookup[(int(ckpt), int(timestep))]
            train_unit = unit_rows(train[local_term])
            query_matrix = np.concatenate(
                [probes[:, qslot, term, :] for qslot in range(len(args.query_ids))],
                axis=0,
            )
            query_unit = unit_rows(query_matrix)
            scores_all = np.asarray(
                jax.device_get(jnp.asarray(train_unit) @ jnp.asarray(query_unit).T),
                dtype=np.float64,
            )

            for qslot, query_id in enumerate(args.query_ids):
                scores = scores_all[:, qslot * 24 : (qslot + 1) * 24]
                output = probe_alignment_matrix(
                    alignments, query_id, int(ckpt) + 1, int(timestep)
                )
                for method in METHODS:
                    selected = selected_probe_indices(scores, method)
                    counts = np.bincount(selected, minlength=24).astype(np.float64)
                    selected_train_cosine = scores[np.arange(len(scores)), selected]
                    row: dict[str, object] = {
                        "query": query_id,
                        "checkpoint": int(ckpt) + 1,
                        "epoch": (int(ckpt) + 1) * 4,
                        "timestamp": int(timestep),
                        "method": method,
                        "num_datapoints": len(scores),
                        "mean_selected_train_query_cosine": float(
                            np.mean(selected_train_cosine)
                        ),
                        "selected_train_query_positive_fraction": float(
                            np.mean(selected_train_cosine > 0.0)
                        ),
                        "selection_entropy_normalized": float(
                            -np.sum(
                                np.where(
                                    counts > 0,
                                    (counts / counts.sum())
                                    * np.log(counts / counts.sum()),
                                    0.0,
                                )
                            )
                            / np.log(24.0)
                        ),
                    }
                    for key, values in output.items():
                        selected_values = values[selected]
                        short = key.removeprefix("cosine_to_")
                        row[f"selected_mean_{short}"] = float(
                            np.mean(selected_values)
                        )
                        row[f"selected_mean_abs_{short}"] = float(
                            np.mean(np.abs(selected_values))
                        )
                        row[f"selected_positive_fraction_{short}"] = float(
                            np.mean(selected_values > 0.0)
                        )
                        row[f"uniform_mean_{short}"] = float(np.mean(values))
                        row[f"uniform_mean_abs_{short}"] = float(
                            np.mean(np.abs(values))
                        )
                        row[f"selection_frequency_vs_{short}_spearman"] = float(
                            spearman(counts, values)
                        )
                        row[f"selection_frequency_vs_abs_{short}_spearman"] = float(
                            spearman(counts, np.abs(values))
                        )
                    delta_abs = np.abs(output["cosine_to_next_predicted_noise_delta"])
                    ranks = np.argsort(np.argsort(delta_abs, kind="mergesort"), kind="mergesort")
                    percentiles = 100.0 * (ranks + 1) / 24.0
                    row["selected_mean_abs_delta_percentile"] = float(
                        np.mean(percentiles[selected])
                    )
                    rows.append(row)
            print(
                f"[nearest-output shard {args.shard_index}/{args.shard_count}] "
                f"checkpoint={int(ckpt) + 1}/49 timestamp={int(timestep)}",
                flush=True,
            )
    write_csv(args.out_dir / f"per_term_shard_{args.shard_index:02d}.csv", rows)


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def aggregate(rows: list[dict[str, str]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    identity = set(keys) | {"epoch", "timestamp", "num_datapoints"}
    numeric = [key for key in rows[0] if key not in identity]
    result = []
    for group, items in sorted(groups.items()):
        record: dict[str, object] = dict(zip(keys, group))
        record["num_terms"] = len(items)
        for key in numeric:
            record[key] = statistics.mean(float(item[key]) for item in items)
        result.append(record)
    return result


def merge(args: argparse.Namespace) -> None:
    rows = read_rows(
        [args.out_dir / f"per_term_shard_{i:02d}.csv" for i in range(args.shard_count)]
    )
    rows.sort(
        key=lambda row: (
            int(row["query"]),
            row["method"],
            int(row["checkpoint"]),
            int(row["timestamp"]),
        )
    )
    write_csv(args.out_dir / "per_term.csv", rows)
    per_checkpoint = aggregate(rows, ("query", "method", "checkpoint"))
    summary = aggregate(rows, ("query", "method"))
    write_csv(args.out_dir / "per_checkpoint.csv", per_checkpoint)
    write_csv(args.out_dir / "query_summary.csv", summary)

    print("NEAREST TRAIN-GRADIENT PROBE VS PREDICTED NOISE — 490 TERMS")
    print(
        f"{'Q':>2s} {'METHOD':20s} {'TRAIN COS':>10s} {'EPS_c':>9s} "
        f"{'EPS_n':>9s} {'DELTA':>9s} {'|DELTA|':>9s} {'DELTA+':>7s} "
        f"{'|D|PCTL':>9s} {'FREQ~D':>8s} {'FREQ~|D|':>10s}"
    )
    print("-" * 116)
    for row in summary:
        print(
            f"{int(row['query']):2d} {str(row['method']):20s} "
            f"{float(row['mean_selected_train_query_cosine']):+10.5f} "
            f"{float(row['selected_mean_current_predicted_noise']):+9.5f} "
            f"{float(row['selected_mean_next_predicted_noise']):+9.5f} "
            f"{float(row['selected_mean_next_predicted_noise_delta']):+9.5f} "
            f"{float(row['selected_mean_abs_next_predicted_noise_delta']):9.5f} "
            f"{float(row['selected_positive_fraction_next_predicted_noise_delta']):7.3f} "
            f"{float(row['selected_mean_abs_delta_percentile']):8.2f}% "
            f"{100.0 * float(row['selection_frequency_vs_next_predicted_noise_delta_spearman']):7.2f}% "
            f"{100.0 * float(row['selection_frequency_vs_abs_next_predicted_noise_delta_spearman']):9.2f}%"
        )
    print(f"[saved] {args.out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("analyze-shard", "merge"))
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-ids", default="2,3")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--original-namespace", default="predicted_noise_output_next_original12"
    )
    parser.add_argument(
        "--fresh-namespace", default="predicted_noise_output_next_fresh12"
    )
    args = parser.parse_args()
    args.query_ids = tuple(int(value) for value in args.query_ids.split(","))
    args.out_dir = args.out_dir or (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "eval"
        / "probe24_nearest_train_predicted_noise"
        / f"run_{args.run_id}"
    )
    if args.command == "analyze-shard":
        analyze_shard(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
