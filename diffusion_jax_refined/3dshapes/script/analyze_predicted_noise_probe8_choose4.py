#!/usr/bin/env python3
"""Evaluate every four-probe subset of a saved eight-probe linear score run."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    "endpoint_contarfactual",
    "noise_trajectory",
    "simple_loss",
    "traj_contarfactual",
)
VARIANTS = {
    "raw": "sums_score",
    "query_l2": "sums_score_query_normalized",
    "train_l2": "sums_score_train_l2_normalized",
    "query_train_l2": "sums_score_query_train_l2_normalized",
}


def average_tie_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = average_tie_ranks(np.asarray(x, dtype=np.float64))
    ry = average_tie_ranks(np.asarray(y, dtype=np.float64))
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    return float(np.sum(rx * ry) / denominator) if denominator else float("nan")


def load_probe_scores(shard_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    shard_paths = sorted(shard_dir.glob("shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"no score shards under {shard_dir}")
    totals: dict[str, np.ndarray] = {}
    score_indices = None
    terms = 0
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as payload:
            for variant, key in VARIANTS.items():
                values = np.asarray(payload[key], dtype=np.float64)
                if values.shape != (8, 10, 5000):
                    raise ValueError(f"{path}:{key} expected (8,10,5000), got {values.shape}")
                if variant not in totals:
                    totals[variant] = np.zeros_like(values)
                totals[variant] += values
            indices = np.asarray(payload["score_indices"], dtype=np.int64)
            terms += int(payload["used_terms"])
        if score_indices is None:
            score_indices = indices
        elif not np.array_equal(score_indices, indices):
            raise ValueError(f"score indices differ in {path}")
    if terms != 500:
        raise ValueError(f"expected 500 accumulated terms, found {terms}")
    assert score_indices is not None
    return totals, score_indices


def cache_group(eval_root: Path) -> Path:
    parent = eval_root / "lds_target_cache" / "ddim_eta0"
    groups = sorted(path for path in parent.iterdir() if path.is_dir()) if parent.is_dir() else []
    if len(groups) != 1:
        raise RuntimeError(f"expected one target-cache group under {parent}; found {groups}")
    return groups[0]


def load_target_data(
    group: Path,
    score_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    reference_paths = sorted((group / TARGETS[0]).glob("target_*.json"))
    if len(reference_paths) != 192:
        raise RuntimeError(f"expected 192 cached models under {group}; found {len(reference_paths)}")
    index_to_column = {int(index): column for column, index in enumerate(score_indices)}
    incidence = np.zeros((len(reference_paths), len(score_indices)), dtype=np.float64)
    for row, path in enumerate(reference_paths):
        payload = json.loads(path.read_text())
        kept = np.asarray(
            np.load(Path(payload["subset_dir"]) / "kept_attribution_indices.npy"),
            dtype=np.int64,
        )
        columns = [index_to_column[int(index)] for index in kept if int(index) in index_to_column]
        incidence[row, columns] = 1.0

    true_values = {}
    for target in TARGETS:
        paths = sorted((group / target).glob("target_*.json"))
        if len(paths) != len(reference_paths):
            raise RuntimeError(f"target {target} has {len(paths)} cached models")
        true_values[target] = np.asarray(
            [float(json.loads(path.read_text())["true_f"]) for path in paths],
            dtype=np.float64,
        )
    return incidence, true_values


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prediction-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    args = parser.parse_args()

    result_root = SHAPES_ROOT / "result" / args.experiment
    shard_dir = (
        result_root
        / "stream_score"
        / "traj_tracin_predicted_noise_jvp_final_post_square_probe8"
        / f"train_seed_{args.train_seed}"
        / f"run_{args.run_id}"
        / "shards"
    )
    probe_scores, score_indices = load_probe_scores(shard_dir)
    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    combinations = list(itertools.combinations(range(8), 4))
    rows: list[dict[str, object]] = []

    for query_id, record in enumerate(records):
        prompt = str(record["prompt"])
        prompt_tag = prompt.replace(",", "_")
        eval_root = (
            result_root
            / "eval"
            / "prompted_solo"
            / f"query_{prompt_tag}"
            / f"initial_seed_{int(record['initial_seed'])}"
        )
        incidence, true_values = load_target_data(cache_group(eval_root), score_indices)
        for variant, values in probe_scores.items():
            per_probe_predictions = (
                args.prediction_sign * values[:, query_id, :] @ incidence.T
            )
            for combination in combinations:
                prediction = per_probe_predictions[list(combination)].mean(axis=0)
                combination_tag = "".join(str(index) for index in combination)
                for target in TARGETS:
                    rows.append(
                        {
                            "combination": combination_tag,
                            "query": query_id,
                            "target": target,
                            "variant": variant,
                            "lds_percent": 100.0 * spearman(prediction, true_values[target]),
                            "prompt": prompt_tag,
                        }
                    )
        print(f"[query {query_id}/9] evaluated 70 combinations", flush=True)

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["combination"]), str(row["target"]), str(row["variant"]))].append(
            float(row["lds_percent"])
        )
    means = [
        {
            "combination": combination,
            "target": target,
            "variant": variant,
            "mean_lds_percent": statistics.mean(values),
        }
        for (combination, target, variant), values in sorted(grouped.items())
    ]
    output_dir = result_root / "eval" / "probe8_choose4_linear" / f"run_{args.run_id}"
    write_csv(output_dir / "per_query.csv", rows)
    write_csv(output_dir / "ten_query_means.csv", means)

    mean_lookup = {
        (str(row["combination"]), str(row["target"]), str(row["variant"])): float(
            row["mean_lds_percent"]
        )
        for row in means
    }
    print("\nALL 70 COMBINATIONS — BOTH-L2 10-QUERY MEAN")
    print(f"{'PROBES':>7s} {'ENDPOINT':>10s} {'NOISE':>10s} {'SIMPLE':>10s} {'TRAJ':>10s}")
    print("-" * 63)
    for combination in combinations:
        tag = "".join(str(index) for index in combination)
        values = [mean_lookup[(tag, target, "query_train_l2")] for target in TARGETS]
        print(f"{tag:>7s} " + " ".join(f"{value:9.3f}%" for value in values))

    print("\nDISTRIBUTION ACROSS 70 COMBINATIONS")
    print(
        f"{'TARGET':24s} {'VARIANT':15s} {'MEAN':>9s} {'STD':>9s} "
        f"{'MIN':>9s} {'MAX':>9s} {'BEST':>7s}"
    )
    print("-" * 94)
    for target in TARGETS:
        for variant in VARIANTS:
            values = [(mean_lookup[("".join(map(str, combo)), target, variant)], combo) for combo in combinations]
            best_value, best_combo = max(values)
            only_values = [value for value, _ in values]
            print(
                f"{target:24s} {variant:15s} {statistics.mean(only_values):8.3f}% "
                f"{statistics.stdev(only_values):8.3f}% {min(only_values):8.3f}% "
                f"{best_value:8.3f}% {''.join(map(str, best_combo)):>7s}"
            )

    ranked = []
    for combination in combinations:
        tag = "".join(str(index) for index in combination)
        endpoint = mean_lookup[(tag, "endpoint_contarfactual", "query_train_l2")]
        trajectory = mean_lookup[(tag, "traj_contarfactual", "query_train_l2")]
        ranked.append((0.5 * (endpoint + trajectory), tag, endpoint, trajectory))
    print("\nTOP 10 — BOTH-L2 COUNTERFACTUAL MEAN")
    print(f"{'RANK':>4s} {'PROBES':>7s} {'CF-MEAN':>10s} {'ENDPOINT':>10s} {'TRAJ':>10s}")
    print("-" * 50)
    for rank, (mean_value, tag, endpoint, trajectory) in enumerate(
        sorted(ranked, reverse=True)[:10], start=1
    ):
        print(f"{rank:4d} {tag:>7s} {mean_value:9.3f}% {endpoint:9.3f}% {trajectory:9.3f}%")

    print(f"\n[saved] {output_dir / 'per_query.csv'}")
    print(f"[saved] {output_dir / 'ten_query_means.csv'}")


if __name__ == "__main__":
    main()
