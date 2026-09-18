#!/usr/bin/env python3
"""Test whether positive-LDS probe gradients share a timestamp-wise hemisphere."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
for root in (SHAPES_ROOT, REFINE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from analyze_q8_p7_probe_direction import (  # noqa: E402
    NAMESPACES,
    SEEDS,
    artifact_path,
)


# The score artifacts and per-query CSV use ``query_train_l2`` as the canonical
# key; presentation scripts may label the same variant as BOTH-L2.
VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")


def unit_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-30)


def centroid_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    """Return positive fraction, minimum/mean cosine, and concentration."""
    values = unit_rows(values)
    mean = np.mean(values, axis=0)
    concentration = float(np.linalg.norm(mean))
    if concentration <= 1e-30:
        return 0.0, -1.0, 0.0, concentration
    center = mean / concentration
    cosines = values @ center
    return (
        float(np.mean(cosines > 0)),
        float(np.min(cosines)),
        float(np.mean(cosines)),
        concentration,
    )


def load_positive_probes(
    path: Path,
    reduction: str,
    sign: str,
    target: str,
) -> dict[tuple[int, str], list[int]]:
    selected: dict[tuple[int, str], list[int]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["reduction"] != reduction
                or row["sign"] != sign
                or row["target"] != target
                or row["variant"] not in VARIANTS
            ):
                continue
            key = (int(row["query"]), row["variant"])
            if float(row["lds_percent"]) > 0:
                selected.setdefault(key, []).append(int(row["probe"]) - 1)
            else:
                selected.setdefault(key, [])
    return selected


def load_bank(
    result_root: Path,
    records: list[dict],
    train_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    per_probe = []
    reference_ckpts = reference_timesteps = None
    for probe, namespace in enumerate(NAMESPACES):
        per_query = []
        for query, record in enumerate(records):
            path = artifact_path(result_root, record, namespace, train_seed)
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as payload:
                features = np.asarray(payload["query_features"], dtype=np.float64)
                ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
                timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            if reference_ckpts is None:
                reference_ckpts, reference_timesteps = ckpts, timesteps
            elif not (
                np.array_equal(reference_ckpts, ckpts)
                and np.array_equal(reference_timesteps, timesteps)
            ):
                raise ValueError(f"metadata mismatch: P{probe + 1} Q{query}: {path}")
            per_query.append(features)
        per_probe.append(np.stack(per_query))
    assert reference_ckpts is not None and reference_timesteps is not None
    return np.stack(per_probe), reference_ckpts, reference_timesteps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--individual-results", type=Path, required=True)
    parser.add_argument("--reduction", default="root")
    parser.add_argument("--sign", default="p1")
    parser.add_argument("--target", default="endpoint_contarfactual")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    positive = load_positive_probes(
        args.individual_results, args.reduction, args.sign, args.target
    )
    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    bank, ckpts, timesteps = load_bank(
        SHAPES_ROOT / "result" / args.experiment, records, args.train_seed
    )

    # Gauge fixing: at each timestamp orient every probe against P1's Q0 vector
    # at Q0's first checkpoint. The same sign is then used for every query and
    # checkpoint at that timestamp.
    first_ckpt = int(np.min(ckpts))
    orientation: dict[tuple[int, int], float] = {}
    for timestep in np.unique(timesteps):
        anchor_mask = (ckpts == first_ckpt) & (timesteps == timestep)
        if np.sum(anchor_mask) != 1:
            raise ValueError(
                f"expected one anchor for checkpoint={first_ckpt}, timestep={timestep}"
            )
        anchor = bank[0, 0, anchor_mask][0]
        for probe in range(len(SEEDS)):
            candidate = bank[probe, 0, anchor_mask][0]
            orientation[(probe, int(timestep))] = (
                1.0 if float(np.dot(anchor, candidate)) >= 0 else -1.0
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "per_query_variant_timestamp.csv"
    fieldnames = [
        "query", "variant", "timestep", "positive_probes", "probe_count",
        "probe_mean_in_hemisphere", "probe_mean_min_cos", "probe_mean_mean_cos",
        "probe_mean_concentration", "all_ckpt_in_hemisphere", "all_ckpt_min_cos",
        "all_ckpt_mean_cos", "all_ckpt_concentration",
    ]
    rows = []

    print(
        "POSITIVE-LDS PROBE GRADIENTS — Q0/FIRST-CHECKPOINT/TIMESTAMP ORIENTED"
    )
    print(
        f"selection={args.reduction}/{args.sign}/{args.target}; "
        f"anchor=P1,Q0,checkpoint {first_ckpt}"
    )
    print(
        f"{'Q':>2} {'VARIANT':>10} {'T':>4} {'PROBES':>17} "
        f"{'PM-IN':>6} {'PM-MIN':>8} {'PM-CONC':>8} "
        f"{'ALL-IN':>7} {'ALL-MIN':>8} {'ALL-CONC':>9}"
    )
    print("-" * 102)
    for query in range(len(records)):
        for variant in VARIANTS:
            probes = sorted(set(positive.get((query, variant), [])))
            if not probes:
                continue
            probe_label = ",".join(f"P{probe + 1}" for probe in probes)
            for timestep in sorted(np.unique(timesteps), reverse=True):
                mask = timesteps == timestep
                oriented = np.stack(
                    [
                        orientation[(probe, int(timestep))] * bank[probe, query, mask]
                        for probe in probes
                    ]
                )
                normalized = unit_rows(oriented.reshape(-1, oriented.shape[-1])).reshape(
                    oriented.shape
                )
                probe_means = unit_rows(np.mean(normalized, axis=1))
                pm_in, pm_min, pm_mean, pm_conc = centroid_stats(probe_means)
                all_in, all_min, all_mean, all_conc = centroid_stats(
                    normalized.reshape(-1, normalized.shape[-1])
                )
                row = {
                    "query": query,
                    "variant": variant,
                    "timestep": int(timestep),
                    "positive_probes": probe_label,
                    "probe_count": len(probes),
                    "probe_mean_in_hemisphere": pm_in,
                    "probe_mean_min_cos": pm_min,
                    "probe_mean_mean_cos": pm_mean,
                    "probe_mean_concentration": pm_conc,
                    "all_ckpt_in_hemisphere": all_in,
                    "all_ckpt_min_cos": all_min,
                    "all_ckpt_mean_cos": all_mean,
                    "all_ckpt_concentration": all_conc,
                }
                rows.append(row)
                print(
                    f"{query:2d} {variant:>10} {int(timestep):4d} "
                    f"{probe_label:>17} {pm_in:6.3f} {pm_min:+8.4f} "
                    f"{pm_conc:8.4f} {all_in:7.3f} {all_min:+8.4f} "
                    f"{all_conc:9.4f}"
                )

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
