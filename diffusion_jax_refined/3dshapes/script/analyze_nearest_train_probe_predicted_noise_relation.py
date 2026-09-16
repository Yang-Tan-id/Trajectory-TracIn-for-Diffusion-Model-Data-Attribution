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
    target_data,
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
OUTPUT_SELECTIONS = {
    "current_noise_direction": "cosine_to_current_predicted_noise",
    "next_noise_direction": "cosine_to_next_predicted_noise",
    "delta_noise_direction": "cosine_to_next_predicted_noise_delta",
    "reference_l2_oriented_delta": "reference_l2_oriented_delta",
    "reference_cosine_oriented_delta": "reference_cosine_oriented_delta",
}


def enabled_output_selections(args: argparse.Namespace) -> tuple[str, ...]:
    base = (
        "current_noise_direction",
        "next_noise_direction",
        "delta_noise_direction",
    )
    if not args.include_reference_oriented:
        return base
    return base + (
        "reference_l2_oriented_delta",
        "reference_cosine_oriented_delta",
    )


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
    reference_keys = (
        "current_to_reference_predicted_noise_l2",
        "next_to_reference_predicted_noise_l2",
        "current_to_reference_predicted_noise_cosines",
        "next_to_reference_predicted_noise_cosines",
    )
    result = {key: [] for key in ALIGNMENT_KEYS + reference_keys}
    for probe in range(24):
        bank = "original" if probe < 12 else "fresh"
        bank_probe = probe + 1 if probe < 12 else probe - 11
        values = alignments[(query_id, bank)][(checkpoint, timestep, bank_probe)]
        for key in result:
            if key in values:
                result[key].append(values[key])
    return {key: np.asarray(values, dtype=np.float64) for key, values in result.items()}


def output_selection_values(output: dict[str, np.ndarray], method: str) -> np.ndarray:
    key = OUTPUT_SELECTIONS[method]
    if key in output:
        return output[key]
    delta = output["cosine_to_next_predicted_noise_delta"]
    if method == "reference_l2_oriented_delta":
        current = output["current_to_reference_predicted_noise_l2"]
        following = output["next_to_reference_predicted_noise_l2"]
        orientation = 1.0 if following[0] <= current[0] else -1.0
        return orientation * delta
    if method == "reference_cosine_oriented_delta":
        current = output["current_to_reference_predicted_noise_cosines"]
        following = output["next_to_reference_predicted_noise_cosines"]
        orientation = 1.0 if following[0] >= current[0] else -1.0
        return orientation * delta
    raise ValueError(method)


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
    output_selection_rows: list[dict[str, object]] = []
    output_selected_totals = None
    score_indices_ref = None
    for ckpt_i in range(args.shard_index, 49, args.shard_count):
        path = train_part_dir(args.experiment, args.train_seed) / f"ckpt_{ckpt_i:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            train = np.asarray(payload["train_features"], dtype=np.float32)
            score_indices = np.asarray(payload["score_indices"], dtype=np.int64)
            ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            term_weights = np.asarray(payload["term_weights"], dtype=np.float64)
        if score_indices_ref is None:
            score_indices_ref = score_indices
            output_selected_totals = {
                method: np.zeros(
                    (len(args.query_ids), len(score_indices)), dtype=np.float64
                )
                for method in enabled_output_selections(args)
            }
        elif not np.array_equal(score_indices_ref, score_indices):
            raise ValueError(f"score indices differ in {path}")

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
                assert output_selected_totals is not None
                for method in enabled_output_selections(args):
                    selection_values = output_selection_values(output, method)
                    selected_probe = int(np.argmax(selection_values))
                    selected_scores = scores[:, selected_probe]
                    output_selected_totals[method][qslot] += (
                        float(term_weights[local_term]) * selected_scores
                    )
                    output_selection_rows.append(
                        {
                            "query": query_id,
                            "checkpoint": int(ckpt) + 1,
                            "epoch": (int(ckpt) + 1) * 4,
                            "timestamp": int(timestep),
                            "method": method,
                            "selected_probe": selected_probe + 1,
                            "selected_bank": (
                                "original" if selected_probe < 12 else "fresh"
                            ),
                            "selected_bank_probe": (
                                selected_probe + 1
                                if selected_probe < 12
                                else selected_probe - 11
                            ),
                            "selection_cosine": float(
                                selection_values[selected_probe]
                            ),
                            "mean_selected_train_query_cosine": float(
                                np.mean(selected_scores)
                            ),
                            "selected_train_query_positive_fraction": float(
                                np.mean(selected_scores > 0.0)
                            ),
                        }
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
    write_csv(
        args.out_dir / f"output_selection_shard_{args.shard_index:02d}.csv",
        output_selection_rows,
    )
    assert output_selected_totals is not None and score_indices_ref is not None
    np.savez_compressed(
        args.out_dir / f"output_selected_scores_shard_{args.shard_index:02d}.npz",
        score_indices=score_indices_ref,
        **output_selected_totals,
    )


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

    output_selection_rows = read_rows(
        [
            args.out_dir / f"output_selection_shard_{i:02d}.csv"
            for i in range(args.shard_count)
        ]
    )
    output_selection_rows.sort(
        key=lambda row: (
            int(row["query"]),
            row["method"],
            int(row["checkpoint"]),
            int(row["timestamp"]),
        )
    )
    write_csv(
        args.out_dir / "output_direction_selection_per_term.csv",
        output_selection_rows,
    )

    output_totals = None
    output_indices = None
    for shard_index in range(args.shard_count):
        path = args.out_dir / f"output_selected_scores_shard_{shard_index:02d}.npz"
        with np.load(path, allow_pickle=False) as payload:
            indices = np.asarray(payload["score_indices"], dtype=np.int64)
            shard_totals = {
                method: np.asarray(payload[method], dtype=np.float64)
                for method in enabled_output_selections(args)
            }
        if output_indices is None:
            output_indices = indices
            output_totals = {
                method: np.zeros_like(values)
                for method, values in shard_totals.items()
            }
        elif not np.array_equal(output_indices, indices):
            raise ValueError(f"score indices differ in {path}")
        assert output_totals is not None
        for method, values in shard_totals.items():
            output_totals[method] += values

    assert output_totals is not None and output_indices is not None
    targets = target_data(args, output_indices)
    output_lds_rows = []
    for qslot, query_id in enumerate(args.query_ids):
        incidence, true_values = targets[query_id]
        for method, values in output_totals.items():
            for prediction_sign in (1.0, -1.0):
                prediction = prediction_sign * values[qslot] @ incidence.T
                for target, truth in true_values.items():
                    output_lds_rows.append(
                        {
                            "query": query_id,
                            "method": method,
                            "prediction_sign": (
                                "p1" if prediction_sign > 0.0 else "m1"
                            ),
                            "target": target,
                            "lds_percent": 100.0 * spearman(prediction, truth),
                        }
                    )
    write_csv(args.out_dir / "output_direction_selected_lds.csv", output_lds_rows)
    output_lds_summary = []
    grouped_output_lds: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in output_lds_rows:
        grouped_output_lds[
            (
                str(row["method"]),
                str(row["prediction_sign"]),
                str(row["target"]),
            )
        ].append(float(row["lds_percent"]))
    for (method, sign, target), values in sorted(grouped_output_lds.items()):
        output_lds_summary.append(
            {
                "method": method,
                "prediction_sign": sign,
                "target": target,
                "num_queries": len(values),
                "mean_lds_percent": statistics.mean(values),
                "std_lds_percent": (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                ),
            }
        )
    write_csv(
        args.out_dir / "output_direction_selected_lds_summary.csv",
        output_lds_summary,
    )

    by_query_method_sign: dict[tuple[int, str, str], dict[str, float]] = defaultdict(dict)
    for row in output_lds_rows:
        by_query_method_sign[
            (
                int(row["query"]),
                str(row["method"]),
                str(row["prediction_sign"]),
            )
        ][str(row["target"])] = float(row["lds_percent"])
    cf_joint_rows = []
    for (query_id, method, sign), values in sorted(by_query_method_sign.items()):
        endpoint = values["endpoint_contarfactual"]
        trajectory = values["traj_contarfactual"]
        cf_joint_rows.append(
            {
                "query": query_id,
                "method": method,
                "prediction_sign": sign,
                "endpoint_lds_percent": endpoint,
                "traj_lds_percent": trajectory,
                "cf_joint_lds_percent": 0.5 * (endpoint + trajectory),
            }
        )
    write_csv(args.out_dir / "output_direction_cf_joint.csv", cf_joint_rows)

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
    print("\nOUTPUT-DIRECTION-SELECTED PROBE LDS — BOTH-L2")
    print(f"{'Q':>2s} {'METHOD':24s} {'SIGN':>4s} {'TARGET':24s} {'LDS':>9s}")
    print("-" * 72)
    for row in output_lds_rows:
        print(
            f"{int(row['query']):2d} {str(row['method']):24s} "
            f"{str(row['prediction_sign']):>4s} {str(row['target']):24s} "
            f"{float(row['lds_percent']):8.3f}%"
        )
    print("\nOUTPUT-DIRECTION-SELECTED LDS — QUERY MEAN")
    print(
        f"{'METHOD':24s} {'SIGN':>4s} {'TARGET':24s} "
        f"{'N':>3s} {'MEAN':>9s} {'STD':>9s}"
    )
    print("-" * 82)
    for row in output_lds_summary:
        print(
            f"{str(row['method']):24s} {str(row['prediction_sign']):>4s} "
            f"{str(row['target']):24s} {int(row['num_queries']):3d} "
            f"{float(row['mean_lds_percent']):8.3f}% "
            f"{float(row['std_lds_percent']):8.3f}%"
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
    parser.add_argument("--include-reference-oriented", action="store_true")
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
