#!/usr/bin/env python3
"""Explain query-specific linear LDS through effective projected probe directions."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT / "script") not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT / "script"))

from analyze_predicted_noise_probe24_output_alignment import artifact_path
from run_predicted_noise_jvp_l2_squared import query_artifact_path, train_part_dir


BANKS = {
    "old12": {
        "query_pattern": "loss_direction_residual_rms_predicted_noise_probe4_r{probe_index}",
        "alignment_namespace": "predicted_noise_output_next_original12",
        "score_suffix": "",
    },
    "fresh12": {
        "query_pattern": (
            "loss_direction_residual_rms_predicted_noise_fresh_seed20260915_"
            "r{probe_index}"
        ),
        "alignment_namespace": "predicted_noise_output_next_fresh12",
        "score_suffix": "_fresh_seed20260915",
    },
}

REFERENCE_SCALARS = {
    "current": "probe_scalars",
    "next": "next_predicted_noise_probe_scalars",
    "next_delta": "next_checkpoint_delta_probe_scalars",
    "reference_delta": "reference_checkpoint_delta_probe_scalars",
    "reference_direction_delta": "reference_direction_delta_probe_scalars",
    "future_mean_delta": "future_mean_delta_probe_scalars",
    "future_lr_delta": "future_lr_weighted_delta_probe_scalars",
}


def parse_query_ids(text: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values or any(value < 0 or value > 9 for value in values):
        raise argparse.ArgumentTypeError("query ids must be a nonempty subset of 0,...,9")
    return values


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.einsum("td,td->t", left, right, optimize=True)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / max(denominator, 1e-12))


def load_query_features(
    args: argparse.Namespace, query_id: int, pattern: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    features = []
    reference = None
    for probe_index in range(args.num_probes):
        path = query_artifact_path(
            args.experiment,
            args.train_seed,
            args.epochs,
            query_id,
            num_probes=args.num_probes,
            probe_index=probe_index,
            query_namespace_pattern=pattern,
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            features.append(np.asarray(payload["query_features"], dtype=np.float32))
            meta = {
                "ckpt_indices": np.asarray(payload["ckpt_indices"], dtype=np.int32),
                "timesteps": np.asarray(payload["timesteps"], dtype=np.int32),
            }
        if reference is None:
            reference = meta
        else:
            for key, value in meta.items():
                if not np.array_equal(value, reference[key]):
                    raise ValueError(f"query metadata mismatch for {path}:{key}")
    assert reference is not None
    return np.stack(features, axis=0), reference


def load_alignment_scalars(
    args: argparse.Namespace, query_id: int, namespace: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    path = artifact_path(
        args.experiment, args.train_seed, args.epochs, query_id, namespace
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    result = {}
    with np.load(path, allow_pickle=False) as payload:
        meta = {
            "ckpt_indices": np.asarray(payload["ckpt_indices"], dtype=np.int32),
            "timesteps": np.asarray(payload["timesteps"], dtype=np.int32),
        }
        for name, key in REFERENCE_SCALARS.items():
            if key in payload:
                values = np.asarray(payload[key], dtype=np.float64)
                if values.shape != (args.num_probes, len(meta["ckpt_indices"])):
                    raise ValueError(f"{path}:{key} has unexpected shape {values.shape}")
                result[name] = values
    return result, meta


@lru_cache(maxsize=None)
def load_weight_lookup(experiment: str, train_seed: int) -> dict[tuple[int, int], float]:
    lookup = {}
    for ckpt_i in range(50):
        path = train_part_dir(experiment, train_seed) / f"ckpt_{ckpt_i:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            weights = np.asarray(payload["term_weights"], dtype=np.float64)
        for ckpt, timestep, weight in zip(ckpts, timesteps, weights):
            lookup[(int(ckpt), int(timestep))] = float(weight)
    return lookup


def load_term_weights(args: argparse.Namespace, meta: dict[str, np.ndarray]) -> np.ndarray:
    lookup = load_weight_lookup(args.experiment, args.train_seed)
    return np.asarray(
        [
            lookup[(int(ckpt), int(timestep))]
            for ckpt, timestep in zip(meta["ckpt_indices"], meta["timesteps"])
        ],
        dtype=np.float64,
    )


def load_cf_joint_lds(args: argparse.Namespace, query_id: int, suffix: str) -> float:
    record = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"][query_id]
    eval_root = (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "eval"
        / "prompted_solo"
        / f"query_{str(record['prompt']).replace(',', '_')}"
        / f"initial_seed_{int(record['initial_seed'])}"
        / "lds"
    )
    namespace = f"traj_tracin_predicted_noise_jvp_final_linear_mean_probe12{suffix}"
    values = []
    for target in ("endpoint_contarfactual", "traj_contarfactual"):
        matches = list(
            (
                eval_root
                / f"{namespace}_query_train_l2"
                / target
                / "pred_kept_sign_p1"
            ).glob("*/lds_summary.json")
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one linear LDS summary for Q={query_id}, target={target}, "
                f"bank suffix={suffix!r}; found {len(matches)}"
            )
        values.append(float(json.loads(matches[0].read_text())["lds_percent"]))
    return statistics.mean(values)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_bank(
    args: argparse.Namespace,
    query_id: int,
    bank: str,
    config: dict[str, str],
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    query, meta = load_query_features(args, query_id, config["query_pattern"])
    # The linear Both-L2 score averages unit query pullbacks over probes.
    query_unit = unit_rows(query)
    effective = query_unit.mean(axis=0)
    weights = load_term_weights(args, meta)
    aggregate = np.einsum("t,td->d", weights, effective, optimize=True)
    scalars, alignment_meta = load_alignment_scalars(
        args, query_id, config["alignment_namespace"]
    )
    query_lookup = {
        (int(ckpt), int(timestep)): term
        for term, (ckpt, timestep) in enumerate(
            zip(meta["ckpt_indices"], meta["timesteps"])
        )
    }
    aligned_terms = np.asarray(
        [
            query_lookup[(int(ckpt), int(timestep))]
            for ckpt, timestep in zip(
                alignment_meta["ckpt_indices"], alignment_meta["timesteps"]
            )
        ],
        dtype=np.int64,
    )
    effective_aligned = effective[aligned_terms]
    weights_aligned = weights[aligned_terms]

    row: dict[str, object] = {
        "query": query_id,
        "bank": bank,
        "cf_joint_lds_percent": load_cf_joint_lds(
            args, query_id, config["score_suffix"]
        ),
        "resultant_norm_mean": float(np.linalg.norm(effective, axis=-1).mean()),
        "resultant_norm_std": float(np.linalg.norm(effective, axis=-1).std()),
    }
    for reference, projection_scalars in scalars.items():
        pulled = np.mean(
            projection_scalars[:, :, None] * query_unit[:, aligned_terms, :], axis=0
        )
        term_cosines = cosine_rows(effective_aligned, pulled)
        pulled_aggregate = np.einsum(
            "t,td->d", weights_aligned, pulled, optimize=True
        )
        effective_aligned_aggregate = np.einsum(
            "t,td->d", weights_aligned, effective_aligned, optimize=True
        )
        row[f"cos_{reference}_term_mean"] = float(term_cosines.mean())
        row[f"cos_{reference}_term_positive_fraction"] = float(
            np.mean(term_cosines > 0.0)
        )
        row[f"cos_{reference}_trajectory"] = cosine(
            effective_aligned_aggregate, pulled_aggregate
        )
    return row, effective, aggregate, weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num-probes", type=int, default=12)
    parser.add_argument("--query-ids", type=parse_query_ids, default=parse_query_ids("0,1,2,3,4,5,6,7,8,9"))
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.num_probes != 12:
        raise ValueError("this comparison expects the two saved 12-probe banks")

    rows = []
    comparisons = []
    for query_id in args.query_ids:
        bank_results = {}
        for bank, config in BANKS.items():
            row, effective, aggregate, weights = analyze_bank(
                args, query_id, bank, config
            )
            rows.append(row)
            bank_results[bank] = (effective, aggregate, weights)
        old_effective, old_aggregate, old_weights = bank_results["old12"]
        fresh_effective, fresh_aggregate, fresh_weights = bank_results["fresh12"]
        if not np.array_equal(old_weights, fresh_weights):
            raise ValueError(f"old/fresh term weights differ for query {query_id}")
        term_cosines = cosine_rows(old_effective, fresh_effective)
        comparisons.append(
            {
                "query": query_id,
                "old_fresh_projected_term_cos_mean": float(term_cosines.mean()),
                "old_fresh_projected_term_cos_std": float(term_cosines.std()),
                "old_fresh_projected_term_positive_fraction": float(
                    np.mean(term_cosines > 0.0)
                ),
                "old_fresh_projected_trajectory_cosine": cosine(
                    old_aggregate, fresh_aggregate
                ),
            }
        )

    out_dir = args.out_dir or (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "eval"
        / "predicted_noise_linear_bank_direction_analysis"
    )
    write_csv(out_dir / "per_query_bank.csv", rows)
    write_csv(out_dir / "old_fresh_comparison.csv", comparisons)

    print("LINEAR BOTH-L2 EFFECTIVE PROJECTED DIRECTIONS")
    print(
        f"{'Q':>2s} {'BANK':8s} {'CF LDS':>8s} {'|MEAN Q|':>9s} "
        f"{'CURR':>8s} {'NEXT':>8s} {'DELTA':>8s} {'REF-DIR':>8s}"
    )
    print("-" * 80)
    for row in rows:
        def trajectory_value(name: str) -> str:
            value = row.get(f"cos_{name}_trajectory")
            return "     n/a" if value is None else f"{float(value):+8.3f}"

        print(
            f"{int(row['query']):2d} {str(row['bank']):8s} "
            f"{float(row['cf_joint_lds_percent']):7.3f}% "
            f"{float(row['resultant_norm_mean']):9.4f} "
            f"{trajectory_value('current')} {trajectory_value('next')} "
            f"{trajectory_value('next_delta')} "
            f"{trajectory_value('reference_direction_delta')}"
        )

    print("\nOLD12 vs FRESH12 — EFFECTIVE PROJECTED DIRECTION")
    print(f"{'Q':>2s} {'TERM COS':>10s} {'TERM +FRAC':>11s} {'TRAJ COS':>10s}")
    print("-" * 43)
    for row in comparisons:
        print(
            f"{int(row['query']):2d} "
            f"{float(row['old_fresh_projected_term_cos_mean']):+10.4f} "
            f"{float(row['old_fresh_projected_term_positive_fraction']):10.3f} "
            f"{float(row['old_fresh_projected_trajectory_cosine']):+10.4f}"
        )
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
