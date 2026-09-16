#!/usr/bin/env python3
"""Compare old/fresh 12-probe linear Both-L2 signs for every trajectory term."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT / "script") not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT / "script"))

from run_predicted_noise_jvp_l2_squared import (
    query_artifact_path,
    records,
    result_root,
    train_part_dir,
)


BANKS = {
    "old12": {
        "query_pattern": "loss_direction_residual_rms_predicted_noise_probe4_r{probe_index}",
        "score_namespace": "traj_tracin_predicted_noise_jvp_final_linear_mean_probe12",
    },
    "fresh12": {
        "query_pattern": (
            "loss_direction_residual_rms_predicted_noise_fresh_seed20260915_"
            "r{probe_index}"
        ),
        "score_namespace": (
            "traj_tracin_predicted_noise_jvp_final_linear_mean_probe12_"
            "fresh_seed20260915"
        ),
    },
}


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_query_ids(text: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values or any(value < 0 or value > 9 for value in values):
        raise argparse.ArgumentTypeError("query ids must be a nonempty subset of 0,...,9")
    return values


def load_effective_query_bank(
    args: argparse.Namespace, pattern: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return mean_r unit(P J^T v_r), with shape (queries, terms, dim)."""
    total = None
    reference = None
    for probe_index in range(args.num_probes):
        per_query = []
        for query_id in args.query_ids:
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
                feature = np.asarray(payload["query_features"], dtype=np.float32)
                meta = {
                    "ckpt_indices": np.asarray(payload["ckpt_indices"], dtype=np.int32),
                    "timesteps": np.asarray(payload["timesteps"], dtype=np.int32),
                }
            norm = np.linalg.norm(feature, axis=-1, keepdims=True)
            per_query.append(feature / np.maximum(norm, 1e-8))
            if reference is None:
                reference = meta
            else:
                for key, value in meta.items():
                    if not np.array_equal(value, reference[key]):
                        raise ValueError(f"query metadata mismatch for {path}:{key}")
        probe = np.stack(per_query, axis=0)
        total = probe if total is None else total + probe
        print(
            f"[query bank] pattern={pattern} probe={probe_index + 1}/{args.num_probes}",
            flush=True,
        )
    assert total is not None and reference is not None
    return (total / float(args.num_probes)).astype(np.float32), reference


def paired_moments(old: np.ndarray, fresh: np.ndarray) -> tuple[np.ndarray, ...]:
    """Sufficient statistics over the leading term axis."""
    return (
        old.sum(axis=0, dtype=np.float64),
        fresh.sum(axis=0, dtype=np.float64),
        np.square(old, dtype=np.float64).sum(axis=0),
        np.square(fresh, dtype=np.float64).sum(axis=0),
        np.multiply(old, fresh, dtype=np.float64).sum(axis=0),
    )


def score_path(args: argparse.Namespace, query_id: int, namespace: str) -> Path:
    record = records()[query_id]
    prompt = str(record["prompt"]).replace(",", "_")
    return (
        result_root(args.experiment)
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{args.train_seed}"
        / f"query_{prompt}"
        / f"initial_seed_{int(record['initial_seed'])}"
        / namespace
        / "score_query_train_l2_normalized"
        / "scores.npy"
    )


def shard(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp

    output = args.out_dir / f"shard_{args.shard_index}.npz"
    term_csv = args.out_dir / f"per_term_shard_{args.shard_index}.csv"
    if output.is_file() and term_csv.is_file():
        print(f"[skip] shard exists: {output}", flush=True)
        return

    old_query, meta = load_effective_query_bank(args, BANKS["old12"]["query_pattern"])
    fresh_query, fresh_meta = load_effective_query_bank(
        args, BANKS["fresh12"]["query_pattern"]
    )
    for key in meta:
        if not np.array_equal(meta[key], fresh_meta[key]):
            raise ValueError(f"old/fresh query metadata mismatch for {key}")
    lookup = {
        (int(ckpt), int(timestep)): term
        for term, (ckpt, timestep) in enumerate(
            zip(meta["ckpt_indices"], meta["timesteps"])
        )
    }

    signs_old = []
    signs_fresh = []
    ckpt_values = []
    timestep_values = []
    weight_values = []
    rows: list[dict[str, object]] = []
    score_indices = None
    final_old = None
    final_fresh = None
    moment_sums = None

    for ckpt_i in range(args.shard_index, 50, args.shard_count):
        path = train_part_dir(args.experiment, args.train_seed) / f"ckpt_{ckpt_i:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            train = np.asarray(payload["train_features"], dtype=np.float32)
            indices = np.asarray(payload["score_indices"], dtype=np.int64)
            ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            weights = np.asarray(payload["term_weights"], dtype=np.float64)
        if score_indices is None:
            score_indices = indices
            shape = (len(args.query_ids), len(indices))
            final_old = np.zeros(shape, dtype=np.float64)
            final_fresh = np.zeros(shape, dtype=np.float64)
            moment_sums = [np.zeros(shape, dtype=np.float64) for _ in range(5)]
        elif not np.array_equal(score_indices, indices):
            raise ValueError(f"score indices differ in {path}")

        for local_term, (ckpt, timestep, weight) in enumerate(
            zip(ckpts, timesteps, weights)
        ):
            term = lookup[(int(ckpt), int(timestep))]
            train_device = jax.device_put(jnp.asarray(train[local_term]))
            train_unit = train_device / jnp.maximum(
                jnp.linalg.norm(train_device, axis=1, keepdims=True), 1e-8
            )
            combined_query = np.concatenate(
                (old_query[:, term], fresh_query[:, term]), axis=0
            )
            values = np.asarray(
                jax.device_get(train_unit @ jnp.asarray(combined_query).T),
                dtype=np.float32,
            ).T
            old = values[: len(args.query_ids)]
            fresh = values[len(args.query_ids) :]

            signs_old.append(np.where(old > 0.0, 1, -1).astype(np.int8))
            signs_fresh.append(np.where(fresh > 0.0, 1, -1).astype(np.int8))
            ckpt_values.append(int(ckpt))
            timestep_values.append(int(timestep))
            weight_values.append(float(weight))
            assert final_old is not None and final_fresh is not None
            final_old += float(weight) * old
            final_fresh += float(weight) * fresh
            assert moment_sums is not None
            for destination, contribution in zip(
                moment_sums, paired_moments(old[None], fresh[None])
            ):
                destination += contribution

            for qslot, query_id in enumerate(args.query_ids):
                old_pos = old[qslot] > 0.0
                fresh_pos = fresh[qslot] > 0.0
                rows.append(
                    {
                        "query": query_id,
                        "checkpoint": int(ckpt) + 1,
                        "epoch": (int(ckpt) + 1) * 4,
                        "timestep": int(timestep),
                        "old_positive_fraction": float(old_pos.mean()),
                        "fresh_positive_fraction": float(fresh_pos.mean()),
                        "both_positive_fraction": float(np.mean(old_pos & fresh_pos)),
                        "old_positive_fresh_negative_fraction": float(
                            np.mean(old_pos & ~fresh_pos)
                        ),
                        "old_negative_fresh_positive_fraction": float(
                            np.mean(~old_pos & fresh_pos)
                        ),
                        "both_negative_fraction": float(np.mean(~old_pos & ~fresh_pos)),
                        "same_sign_fraction": float(np.mean(old_pos == fresh_pos)),
                        "old_mean": float(old[qslot].mean()),
                        "fresh_mean": float(fresh[qslot].mean()),
                        "old_std": float(old[qslot].std()),
                        "fresh_std": float(fresh[qslot].std()),
                        "datapoint_pearson": float(
                            np.corrcoef(old[qslot], fresh[qslot])[0, 1]
                        ),
                    }
                )
        print(f"[term signs] checkpoint={ckpt_i + 1}/50", flush=True)

    assert score_indices is not None and final_old is not None and final_fresh is not None
    assert moment_sums is not None
    atomic_savez(
        output,
        old_sign=np.stack(signs_old, axis=1),
        fresh_sign=np.stack(signs_fresh, axis=1),
        query_ids=np.asarray(args.query_ids, dtype=np.int32),
        ckpt_indices=np.asarray(ckpt_values, dtype=np.int32),
        timesteps=np.asarray(timestep_values, dtype=np.int32),
        term_weights=np.asarray(weight_values, dtype=np.float64),
        score_indices=score_indices,
        final_old=final_old,
        final_fresh=final_fresh,
        sum_old=moment_sums[0],
        sum_fresh=moment_sums[1],
        sum_old_square=moment_sums[2],
        sum_fresh_square=moment_sums[3],
        sum_cross=moment_sums[4],
    )
    write_csv(term_csv, rows)
    print(f"[saved] {output}", flush=True)
    print(f"[saved] {term_csv}", flush=True)


def merge(args: argparse.Namespace) -> None:
    payloads = []
    term_rows = []
    for shard_index in range(args.shard_count):
        path = args.out_dir / f"shard_{shard_index}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        payloads.append(np.load(path, allow_pickle=False))
        with (args.out_dir / f"per_term_shard_{shard_index}.csv").open(newline="") as handle:
            term_rows.extend(csv.DictReader(handle))

    query_ids = np.asarray(payloads[0]["query_ids"], dtype=np.int32)
    score_indices = np.asarray(payloads[0]["score_indices"], dtype=np.int64)
    for payload in payloads[1:]:
        if not np.array_equal(query_ids, payload["query_ids"]):
            raise ValueError("query ids differ across shards")
        if not np.array_equal(score_indices, payload["score_indices"]):
            raise ValueError("score indices differ across shards")

    ckpts = np.concatenate([payload["ckpt_indices"] for payload in payloads])
    timesteps = np.concatenate([payload["timesteps"] for payload in payloads])
    order = np.lexsort((timesteps, ckpts))
    old_sign = np.concatenate([payload["old_sign"] for payload in payloads], axis=1)[:, order]
    fresh_sign = np.concatenate([payload["fresh_sign"] for payload in payloads], axis=1)[:, order]
    ckpts = ckpts[order]
    timesteps = timesteps[order]
    weights = np.concatenate([payload["term_weights"] for payload in payloads])[order]
    if old_sign.shape[1:] != (500, 5000):
        raise ValueError(f"unexpected merged sign shape: {old_sign.shape}")

    final_old = sum(np.asarray(payload["final_old"], dtype=np.float64) for payload in payloads)
    final_fresh = sum(
        np.asarray(payload["final_fresh"], dtype=np.float64) for payload in payloads
    )
    moment_keys = (
        "sum_old",
        "sum_fresh",
        "sum_old_square",
        "sum_fresh_square",
        "sum_cross",
    )
    moments = {
        key: sum(np.asarray(payload[key], dtype=np.float64) for payload in payloads)
        for key in moment_keys
    }

    count = float(old_sign.shape[1])
    mean_old = moments["sum_old"] / count
    mean_fresh = moments["sum_fresh"] / count
    covariance = moments["sum_cross"] / count - mean_old * mean_fresh
    variance_old = moments["sum_old_square"] / count - np.square(mean_old)
    variance_fresh = moments["sum_fresh_square"] / count - np.square(mean_fresh)
    term_pearson = covariance / np.maximum(
        np.sqrt(np.maximum(variance_old, 0.0) * np.maximum(variance_fresh, 0.0)),
        1e-15,
    )

    datapoint_rows = []
    summary_rows = []
    print("OLD12 vs FRESH12 LINEAR BOTH-L2 TERM SIGNS")
    print(
        f"{'Q':>2s} {'OLD+':>8s} {'FRESH+':>8s} {'++':>8s} {'+-':>8s} "
        f"{'-+':>8s} {'--':>8s} {'SAME':>8s} {'TERM r':>9s}"
    )
    print("-" * 91)
    for qslot, query_id in enumerate(query_ids):
        old_pos = old_sign[qslot] > 0
        fresh_pos = fresh_sign[qslot] > 0
        old_fraction = old_pos.mean(axis=0)
        fresh_fraction = fresh_pos.mean(axis=0)
        pp = np.mean(old_pos & fresh_pos, axis=0)
        pn = np.mean(old_pos & ~fresh_pos, axis=0)
        np_ = np.mean(~old_pos & fresh_pos, axis=0)
        nn = np.mean(~old_pos & ~fresh_pos, axis=0)
        same = pp + nn
        for datapoint_slot, score_index in enumerate(score_indices):
            datapoint_rows.append(
                {
                    "query": int(query_id),
                    "score_index": int(score_index),
                    "old_positive_fraction_over_500_terms": float(
                        old_fraction[datapoint_slot]
                    ),
                    "fresh_positive_fraction_over_500_terms": float(
                        fresh_fraction[datapoint_slot]
                    ),
                    "both_positive_fraction": float(pp[datapoint_slot]),
                    "old_positive_fresh_negative_fraction": float(pn[datapoint_slot]),
                    "old_negative_fresh_positive_fraction": float(np_[datapoint_slot]),
                    "both_negative_fraction": float(nn[datapoint_slot]),
                    "same_sign_fraction": float(same[datapoint_slot]),
                    "old_fresh_term_pearson": float(term_pearson[qslot, datapoint_slot]),
                }
            )
        summary = {
            "query": int(query_id),
            "old_positive_fraction": float(old_pos.mean()),
            "fresh_positive_fraction": float(fresh_pos.mean()),
            "both_positive_fraction": float(np.mean(old_pos & fresh_pos)),
            "old_positive_fresh_negative_fraction": float(
                np.mean(old_pos & ~fresh_pos)
            ),
            "old_negative_fresh_positive_fraction": float(
                np.mean(~old_pos & fresh_pos)
            ),
            "both_negative_fraction": float(np.mean(~old_pos & ~fresh_pos)),
            "same_sign_fraction": float(np.mean(old_pos == fresh_pos)),
            "mean_datapoint_term_pearson": float(term_pearson[qslot].mean()),
        }
        summary_rows.append(summary)
        print(
            f"{int(query_id):2d} "
            f"{summary['old_positive_fraction']:8.3f} "
            f"{summary['fresh_positive_fraction']:8.3f} "
            f"{summary['both_positive_fraction']:8.3f} "
            f"{summary['old_positive_fresh_negative_fraction']:8.3f} "
            f"{summary['old_negative_fresh_positive_fraction']:8.3f} "
            f"{summary['both_negative_fraction']:8.3f} "
            f"{summary['same_sign_fraction']:8.3f} "
            f"{summary['mean_datapoint_term_pearson']:+9.4f}"
        )

        for bank, reconstructed in (("old12", final_old), ("fresh12", final_fresh)):
            expected = np.asarray(
                np.load(score_path(args, int(query_id), BANKS[bank]["score_namespace"])),
                dtype=np.float64,
            )
            error = float(np.max(np.abs(reconstructed[qslot] - expected)))
            print(f"    validate {bank}: max_abs_error={error:.3e}")
            if error > 2e-7:
                raise ValueError(f"{bank} Q{query_id} reconstruction error {error}")

    term_rows.sort(
        key=lambda row: (
            int(row["query"]),
            int(row["checkpoint"]),
            int(row["timestep"]),
        )
    )
    write_csv(args.out_dir / "per_term.csv", term_rows)
    write_csv(args.out_dir / "per_datapoint.csv", datapoint_rows)
    write_csv(args.out_dir / "per_query_summary.csv", summary_rows)
    atomic_savez(
        args.out_dir / "term_signs.npz",
        old_sign=old_sign,
        fresh_sign=fresh_sign,
        query_ids=query_ids,
        ckpt_indices=ckpts,
        timesteps=timesteps,
        term_weights=weights,
        score_indices=score_indices,
    )
    manifest = {
        "definition": (
            "sign(mean_probe(dot(unit(train_projected_gradient), "
            "unit(query_projected_gradient))))"
        ),
        "shape": [len(query_ids), 500, 5000],
        "axes": ["query", "checkpoint_timestamp_term", "datapoint"],
        "banks": BANKS,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    print(f"[saved] {args.out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("shard", "merge"):
        child = subparsers.add_parser(command)
        child.add_argument("--experiment", default="experiment1")
        child.add_argument("--train-seed", type=int, default=42)
        child.add_argument("--epochs", type=int, default=200)
        child.add_argument("--num-probes", type=int, default=12)
        child.add_argument(
            "--query-ids",
            type=parse_query_ids,
            default=parse_query_ids("0,1,2,3,4,5,6,7,8,9"),
        )
        child.add_argument("--shard-count", type=int, default=2)
        child.add_argument("--out-dir", type=Path, required=True)
        if command == "shard":
            child.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args()
    if args.num_probes != 12:
        raise ValueError("this analysis expects old12 and fresh12")
    if args.command == "shard":
        shard(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
