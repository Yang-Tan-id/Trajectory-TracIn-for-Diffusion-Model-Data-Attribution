#!/usr/bin/env python3
"""Compare fixed-reference timestamp-shared probes with checkpoint noise deltas."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
PROBE_SEEDS = (
    314159265, 271828183, 161803399, 141421357,
    538102947, 794615203, 126937481, 682450719,
    905173624, 417286953, 263590817, 849031576,
)


def probe_key(seed: int, timestep: int):
    key = jax.random.PRNGKey(seed)
    for value in (0x54535042, timestep):
        key = jax.random.fold_in(key, int(value))
    return key


def records() -> list[dict[str, object]]:
    return json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]


def artifact_path(args: argparse.Namespace, query: int) -> Path:
    record = records()[query]
    prompt = str(record["prompt"]).replace(",", "_")
    seed = int(record["initial_seed"])
    return (
        SHAPES_ROOT / "result" / args.experiment / "sample_ddim_eta0_1000" / "cifar"
        / f"prompt_{prompt}"
        / f"model_prompted_solo__ckpt_seed_{args.train_seed}_epoch_{args.epochs:04d}"
        / f"seed_{seed:06d}_query_gradient_{args.geometry_namespace}"
        / "traj_tracin" / "query_gradient_artifact.npz"
    )


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    offset = 0
    for path in paths:
        with path.open(newline="") as handle:
            current = list(csv.DictReader(handle))
        individual = [row for row in current if row["kind"] == "individual"]
        local_count = max(int(row["probe"]) for row in individual)
        for row in individual:
            copied = dict(row)
            copied["probe"] = str(int(row["probe"]) + offset)
            rows.append(copied)
        offset += local_count
    if offset != len(PROBE_SEEDS):
        raise ValueError(f"expected 12 probes across LDS files, found {offset}")
    return rows


def classifications(rows: list[dict[str, str]], threshold: float) -> dict[tuple[int, int, str], str]:
    grouped: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["target"] != "endpoint_contarfactual":
            continue
        if row["reduction"] not in ("square", "absolute") or row["sign"] != "p1":
            continue
        key = (int(row["query"]), int(row["probe"]), row["reduction"])
        grouped[key].append(float(row["lds_percent"]))
    result: dict[tuple[int, int, str], str] = {}
    for key, values in grouped.items():
        if len(values) != 4:
            raise ValueError(f"expected four variants for {key}, got {len(values)}")
        if min(values) > 0.0 and values[-1] >= threshold:
            result[key] = "strong_positive"
        elif max(values) < 0.0 and values[-1] <= -threshold:
            result[key] = "strong_negative"
        else:
            result[key] = "other"
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--geometry-namespace", default="predicted_noise_output_next_original12")
    parser.add_argument("--lds-csv", type=Path, action="append", required=True)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    labels = classifications(read_rows(args.lds_csv), args.threshold)
    pair_rows: list[dict[str, object]] = []
    timestamp_rows: list[dict[str, object]] = []

    for query in range(10):
        path = artifact_path(args, query)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path}\nRun a collect-all next-checkpoint probe-alignment artifact first."
            )
        with np.load(path, allow_pickle=False) as payload:
            deltas = np.asarray(payload["checkpoint_next_predicted_noise_deltas"], dtype=np.float64)
            flat_timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            term_weights = np.asarray(payload["term_weights"], dtype=np.float64)
        checkpoint_count, timestamp_count = deltas.shape[:2]
        timesteps = flat_timesteps[:timestamp_count]
        weights = term_weights.reshape(checkpoint_count, timestamp_count)
        axes = tuple(range(2, deltas.ndim))
        delta_norm = np.sqrt(np.sum(np.square(deltas), axis=axes))
        dimension = int(np.prod(deltas.shape[2:]))

        for probe, seed in enumerate(PROBE_SEEDS, start=1):
            cosines = np.empty((checkpoint_count, timestamp_count), dtype=np.float64)
            standardized_projection = np.empty_like(cosines)
            for slot, timestep in enumerate(timesteps):
                v = np.asarray(
                    jax.random.normal(probe_key(seed, int(timestep)), deltas.shape[2:]),
                    dtype=np.float64,
                )
                v_norm = np.linalg.norm(v)
                dots = np.sum(deltas[:, slot] * v, axis=tuple(range(1, deltas[:, slot].ndim)))
                cosines[:, slot] = dots / np.maximum(v_norm * delta_norm[:, slot], 1e-12)
                standardized_projection[:, slot] = np.abs(dots) / np.maximum(delta_norm[:, slot], 1e-12)
            for reduction in ("square", "absolute"):
                label = labels[(query, probe, reduction)]
                w = weights / np.maximum(weights.sum(), 1e-12)
                pair_rows.append({
                    "query": query, "probe": probe, "probe_seed": seed,
                    "reduction": reduction, "group": label,
                    "mean_cosine": float(np.sum(w * cosines)),
                    "mean_abs_cosine": float(np.sum(w * np.abs(cosines))),
                    "rms_cosine": float(np.sqrt(np.sum(w * np.square(cosines)))),
                    "sqrt_d_mean_abs_cosine": float(np.sqrt(dimension) * np.sum(w * np.abs(cosines))),
                    "mean_standardized_abs_projection": float(np.sum(w * standardized_projection)),
                })
                for slot, timestep in enumerate(timesteps):
                    tw = weights[:, slot] / np.maximum(weights[:, slot].sum(), 1e-12)
                    timestamp_rows.append({
                        "query": query, "probe": probe, "probe_seed": seed,
                        "reduction": reduction, "group": label, "timestep": int(timestep),
                        "mean_cosine": float(np.sum(tw * cosines[:, slot])),
                        "mean_abs_cosine": float(np.sum(tw * np.abs(cosines[:, slot]))),
                        "rms_cosine": float(np.sqrt(np.sum(tw * np.square(cosines[:, slot])))),
                        "sqrt_d_mean_abs_cosine": float(np.sqrt(dimension) * np.sum(tw * np.abs(cosines[:, slot]))),
                        "mean_standardized_abs_projection": float(np.sum(tw * standardized_projection[:, slot])),
                    })

    write_csv(args.out_dir / "per_query_probe.csv", pair_rows)
    write_csv(args.out_dir / "per_query_probe_timestamp.csv", timestamp_rows)

    print("FIXED-REFERENCE v[t] ALIGNMENT WITH NEXT-CHECKPOINT DELTA-EPS")
    print(f"{'REDUCTION':9s} {'GROUP':16s} {'N':>3s} {'|COS|':>9s} {'RMS-COS':>9s} {'sqrt(D)|COS|':>13s} {'|DOT|/DELTA':>12s}")
    print("-" * 82)
    for reduction in ("square", "absolute"):
        for group in ("strong_positive", "strong_negative", "other"):
            selected = [row for row in pair_rows if row["reduction"] == reduction and row["group"] == group]
            if not selected:
                continue
            mean = lambda key: float(np.mean([float(row[key]) for row in selected]))
            print(f"{reduction:9s} {group:16s} {len(selected):3d} {mean('mean_abs_cosine'):9.5f} "
                  f"{mean('rms_cosine'):9.5f} {mean('sqrt_d_mean_abs_cosine'):13.5f} "
                  f"{mean('mean_standardized_abs_projection'):12.5f}")

    print("\nPOSITIVE MINUS NEGATIVE BY TIMESTAMP")
    print(f"{'REDUCTION':9s} {'T':>4s} {'DELTA |COS|':>13s} {'DELTA RMS':>11s} {'DELTA sqrtD':>12s}")
    print("-" * 58)
    for reduction in ("square", "absolute"):
        for timestep in sorted({int(row["timestep"]) for row in timestamp_rows}, reverse=True):
            groups = {}
            for group in ("strong_positive", "strong_negative"):
                selected = [row for row in timestamp_rows if row["reduction"] == reduction and row["group"] == group and row["timestep"] == timestep]
                if selected:
                    groups[group] = {key: np.mean([float(row[key]) for row in selected]) for key in ("mean_abs_cosine", "rms_cosine", "sqrt_d_mean_abs_cosine")}
            if len(groups) == 2:
                pos, neg = groups["strong_positive"], groups["strong_negative"]
                print(f"{reduction:9s} {timestep:4d} {pos['mean_abs_cosine']-neg['mean_abs_cosine']:+13.6f} "
                      f"{pos['rms_cosine']-neg['rms_cosine']:+11.6f} "
                      f"{pos['sqrt_d_mean_abs_cosine']-neg['sqrt_d_mean_abs_cosine']:+12.6f}")
    print(f"\n[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
