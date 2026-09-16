#!/usr/bin/env python3
"""Relate per-term 24-probe LDS winners to current/next predicted noise."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]


def records() -> list[dict[str, object]]:
    import json

    return json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]


def artifact_path(experiment: str, train_seed: int, epochs: int, query_id: int, namespace: str) -> Path:
    record = records()[query_id]
    prompt = str(record["prompt"]).replace(",", "_")
    seed = int(record["initial_seed"])
    return (
        SHAPES_ROOT
        / "result"
        / experiment
        / "sample_ddim_eta0_1000"
        / "cifar"
        / f"prompt_{prompt}"
        / f"model_prompted_solo__ckpt_seed_{train_seed}_epoch_{epochs:04d}"
        / f"seed_{seed:06d}_query_gradient_{namespace}"
        / "traj_tracin"
        / "query_gradient_artifact.npz"
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = rank(left)
    right_rank = rank(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def load_bank(args: argparse.Namespace, query_id: int, bank: str) -> dict[tuple[int, int, int], dict[str, float]]:
    namespace = (
        args.original_namespace if bank == "original" else args.fresh_namespace
    )
    path = artifact_path(
        args.experiment, args.train_seed, args.epochs, query_id, namespace
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint_direction = getattr(args, "checkpoint_direction", "next")
    if checkpoint_direction not in ("next", "previous"):
        raise ValueError(f"unsupported checkpoint direction: {checkpoint_direction}")
    adjacent = checkpoint_direction
    delta_prefix = (
        "next_checkpoint" if checkpoint_direction == "next" else "previous_checkpoint"
    )
    with np.load(path, allow_pickle=False) as payload:
        ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
        timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        current = np.asarray(payload["probe_cosines"], dtype=np.float64)
        following = np.asarray(
            payload[f"{adjacent}_predicted_noise_probe_cosines"], dtype=np.float64
        )
        delta = np.asarray(
            payload[f"{delta_prefix}_delta_probe_cosines"], dtype=np.float64
        )
        delta_scalar = np.asarray(
            payload[f"{delta_prefix}_delta_probe_scalars"], dtype=np.float64
        )
        reference_metrics = {}
        reference_key_pairs = (
            (
                "current_to_reference_predicted_noise_l2",
                "current_to_reference_predicted_noise_l2",
            ),
            (
                f"{adjacent}_to_reference_predicted_noise_l2",
                "next_to_reference_predicted_noise_l2",
            ),
            (
                "current_to_reference_predicted_noise_cosines",
                "current_to_reference_predicted_noise_cosines",
            ),
            (
                f"{adjacent}_to_reference_predicted_noise_cosines",
                "next_to_reference_predicted_noise_cosines",
            ),
        )
        for payload_key, canonical_key in reference_key_pairs:
            if payload_key in payload:
                reference_metrics[canonical_key] = np.asarray(
                    payload[payload_key], dtype=np.float64
                )
    result = {}
    for term, (ckpt, timestep) in enumerate(zip(ckpts, timesteps)):
        for probe in range(current.shape[0]):
            values = {
                "cosine_to_current_predicted_noise": float(current[probe, term]),
                "cosine_to_next_predicted_noise": float(following[probe, term]),
                "cosine_to_next_predicted_noise_delta": float(delta[probe, term]),
                "projection_on_next_predicted_noise_delta": float(
                    delta_scalar[probe, term]
                ),
            }
            values.update(
                {key: float(metric[term]) for key, metric in reference_metrics.items()}
            )
            result[(int(ckpt) + 1, int(timestep), probe + 1)] = values
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-ids", default="2,3")
    parser.add_argument("--winner-dir", type=Path, required=True)
    parser.add_argument(
        "--original-namespace", default="predicted_noise_output_next_original12"
    )
    parser.add_argument(
        "--fresh-namespace", default="predicted_noise_output_next_fresh12"
    )
    args = parser.parse_args()
    query_ids = tuple(int(value) for value in args.query_ids.split(","))

    alignments = {
        (query_id, bank): load_bank(args, query_id, bank)
        for query_id in query_ids
        for bank in ("original", "fresh")
    }
    rows = read_csv(args.winner_dir / "all_probe_term_lds.csv")
    enriched: list[dict[str, object]] = []
    for row in rows:
        query_id = int(row["query"])
        if query_id not in query_ids:
            continue
        key = (int(row["checkpoint"]), int(row["timestamp"]), int(row["bank_probe"]))
        output = alignments[(query_id, row["bank"])][key]
        enriched.append(dict(row) | output)
    write_csv(args.winner_dir / "all_probe_term_output_alignment.csv", enriched)

    grouped: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in enriched:
        grouped[(int(row["query"]), int(row["checkpoint"]), int(row["timestamp"]))].append(row)

    selected: list[dict[str, object]] = []
    term_relations: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (query_id, _checkpoint, _timestep), term_rows in grouped.items():
        term_rows.sort(key=lambda row: int(row["probe"]))
        lds = np.asarray([float(row["cf_joint_lds_percent"]) for row in term_rows])
        winner = int(np.argmax(lds))
        chosen = dict(term_rows[winner])
        delta = np.asarray(
            [float(row["cosine_to_next_predicted_noise_delta"]) for row in term_rows]
        )
        current = np.asarray(
            [float(row["cosine_to_current_predicted_noise"]) for row in term_rows]
        )
        following = np.asarray(
            [float(row["cosine_to_next_predicted_noise"]) for row in term_rows]
        )
        chosen["delta_cosine_percentile_among_24"] = float(
            100.0 * np.mean(delta <= delta[winner])
        )
        chosen["abs_delta_cosine_percentile_among_24"] = float(
            100.0 * np.mean(np.abs(delta) <= abs(delta[winner]))
        )
        selected.append(chosen)
        term_relations[query_id]["delta"].append(100.0 * spearman(lds, delta))
        term_relations[query_id]["abs_delta"].append(
            100.0 * spearman(lds, np.abs(delta))
        )
        term_relations[query_id]["current"].append(100.0 * spearman(lds, current))
        term_relations[query_id]["next"].append(100.0 * spearman(lds, following))

    selected.sort(
        key=lambda row: (int(row["query"]), int(row["checkpoint"]), int(row["timestamp"]))
    )
    write_csv(args.winner_dir / "selected_term_winners_output_alignment.csv", selected)

    summaries = []
    for query_id in query_ids:
        chosen = [row for row in selected if int(row["query"]) == query_id]
        values = lambda key: [float(row[key]) for row in chosen]
        relation = term_relations[query_id]
        summaries.append(
            {
                "query": query_id,
                "num_terms": len(chosen),
                "winner_mean_cosine_to_current_predicted_noise": statistics.mean(
                    values("cosine_to_current_predicted_noise")
                ),
                "winner_mean_cosine_to_next_predicted_noise": statistics.mean(
                    values("cosine_to_next_predicted_noise")
                ),
                "winner_mean_cosine_to_next_predicted_noise_delta": statistics.mean(
                    values("cosine_to_next_predicted_noise_delta")
                ),
                "winner_mean_abs_cosine_to_next_predicted_noise_delta": statistics.mean(
                    [abs(value) for value in values("cosine_to_next_predicted_noise_delta")]
                ),
                "winner_delta_positive_fraction": statistics.mean(
                    [value > 0.0 for value in values("cosine_to_next_predicted_noise_delta")]
                ),
                "winner_mean_abs_delta_percentile": statistics.mean(
                    values("abs_delta_cosine_percentile_among_24")
                ),
                "mean_lds_vs_delta_cosine_spearman_percent": statistics.mean(
                    relation["delta"]
                ),
                "mean_lds_vs_abs_delta_cosine_spearman_percent": statistics.mean(
                    relation["abs_delta"]
                ),
                "mean_lds_vs_current_cosine_spearman_percent": statistics.mean(
                    relation["current"]
                ),
                "mean_lds_vs_next_cosine_spearman_percent": statistics.mean(
                    relation["next"]
                ),
            }
        )
    write_csv(args.winner_dir / "output_alignment_summary.csv", summaries)

    print("24-PROBE WINNERS VS RAW OUTPUT-SPACE PREDICTED NOISE")
    print(
        f"{'Q':>2s} {'TERMS':>5s} {'COS EPS_c':>10s} {'COS EPS_n':>10s} "
        f"{'COS DELTA':>10s} {'|DELTA|':>9s} {'DELTA+':>7s} "
        f"{'|D| PCTL':>9s} {'LDS~D':>8s} {'LDS~|D|':>9s}"
    )
    print("-" * 105)
    for row in summaries:
        print(
            f"{int(row['query']):2d} {int(row['num_terms']):5d} "
            f"{float(row['winner_mean_cosine_to_current_predicted_noise']):+10.5f} "
            f"{float(row['winner_mean_cosine_to_next_predicted_noise']):+10.5f} "
            f"{float(row['winner_mean_cosine_to_next_predicted_noise_delta']):+10.5f} "
            f"{float(row['winner_mean_abs_cosine_to_next_predicted_noise_delta']):9.5f} "
            f"{float(row['winner_delta_positive_fraction']):7.3f} "
            f"{float(row['winner_mean_abs_delta_percentile']):8.2f}% "
            f"{float(row['mean_lds_vs_delta_cosine_spearman_percent']):7.2f}% "
            f"{float(row['mean_lds_vs_abs_delta_cosine_spearman_percent']):8.2f}%"
        )
    print(f"[saved] {args.winner_dir}")


if __name__ == "__main__":
    main()
