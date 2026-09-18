#!/usr/bin/env python3
"""Compare positive- and negative-LDS probe concentration by timestamp."""

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

from analyze_positive_probe_timestamp_hemisphere import (  # noqa: E402
    VARIANTS,
    load_bank,
    unit_rows,
)


def load_lds(
    path: Path, reduction: str, sign: str, target: str
) -> dict[tuple[int, str, int], float]:
    values: dict[tuple[int, str, int], float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["reduction"] == reduction
                and row["sign"] == sign
                and row["target"] == target
                and row["variant"] in VARIANTS
            ):
                values[(int(row["query"]), row["variant"], int(row["probe"]) - 1)] = float(
                    row["lds_percent"]
                )
    return values


def group_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    """Return pairwise mean/min cosine, concentration, and centroid hemisphere fraction."""
    values = unit_rows(values)
    center = np.mean(values, axis=0)
    concentration = float(np.linalg.norm(center))
    if len(values) < 2:
        pair_mean = pair_min = float("nan")
    else:
        gram = values @ values.T
        pair = gram[np.triu_indices(len(values), k=1)]
        pair_mean, pair_min = float(np.mean(pair)), float(np.min(pair))
    if concentration <= 1e-30:
        hemisphere = 0.0
    else:
        hemisphere = float(np.mean(values @ (center / concentration) > 0))
    return pair_mean, pair_min, concentration, hemisphere


def center_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_center = np.mean(unit_rows(left), axis=0)
    right_center = np.mean(unit_rows(right), axis=0)
    denom = np.linalg.norm(left_center) * np.linalg.norm(right_center)
    return float(np.dot(left_center, right_center) / max(float(denom), 1e-30))


def remove_common_pc1(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Remove PC1 of the eight probe-mean unit directions using an 8x8 Gram matrix."""
    values = unit_rows(values)
    gram = values @ values.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    coefficients = eigenvectors[:, -1]
    pc1 = coefficients @ values
    pc1 /= max(float(np.linalg.norm(pc1)), 1e-30)
    residual = values - (values @ pc1)[:, None] * pc1[None, :]
    explained = float(eigenvalues[-1] / max(float(np.sum(eigenvalues)), 1e-30))
    return unit_rows(residual), explained


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

    lds = load_lds(args.individual_results, args.reduction, args.sign, args.target)
    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    bank, ckpts, timesteps = load_bank(
        SHAPES_ROOT / "result" / args.experiment, records, args.train_seed
    )
    first_ckpt = int(np.min(ckpts))

    # Fix the otherwise arbitrary probe sign against P1 at Q0's first
    # checkpoint, separately for each timestamp.
    orientation: dict[tuple[int, int], float] = {}
    for timestep in np.unique(timesteps):
        mask = (ckpts == first_ckpt) & (timesteps == timestep)
        anchor = bank[0, 0, mask][0]
        for probe in range(bank.shape[0]):
            direction = bank[probe, 0, mask][0]
            orientation[(probe, int(timestep))] = (
                1.0 if float(np.dot(anchor, direction)) >= 0 else -1.0
            )

    rows: list[dict[str, object]] = []
    print("POSITIVE vs NEGATIVE ENDPOINT-LDS PROBE CONCENTRATION BY TIMESTAMP")
    print(
        f"selection={args.reduction}/{args.sign}/{args.target}; "
        f"gauge=P1,Q0,checkpoint {first_ckpt}"
    )
    print(
        f"{'Q':>2} {'VARIANT':>14} {'T':>4} {'N+':>3} {'N-':>3} "
        f"{'POS-COS':>8} {'NEG-COS':>8} {'DELTA':>8} {'+/−CTR':>8} "
        f"{'PC1%':>7} {'RES+':>8} {'RES-':>8} {'RESΔ':>8} {'RESCTR':>8}"
    )
    print("-" * 120)

    for query in range(len(records)):
        for variant in VARIANTS:
            probe_lds = np.asarray(
                [lds[(query, variant, probe)] for probe in range(bank.shape[0])],
                dtype=np.float64,
            )
            positive = np.flatnonzero(probe_lds > 0)
            negative = np.flatnonzero(probe_lds <= 0)
            if len(positive) == 0 or len(negative) == 0:
                continue
            for timestep in sorted(np.unique(timesteps), reverse=True):
                mask = timesteps == timestep
                probe_means = []
                for probe in range(bank.shape[0]):
                    directions = orientation[(probe, int(timestep))] * bank[probe, query, mask]
                    probe_means.append(np.mean(unit_rows(directions), axis=0))
                probe_means = unit_rows(np.stack(probe_means))
                residual, explained = remove_common_pc1(probe_means)

                pos = group_stats(probe_means[positive])
                neg = group_stats(probe_means[negative])
                cross = center_cosine(probe_means[positive], probe_means[negative])
                pos_res = group_stats(residual[positive])
                neg_res = group_stats(residual[negative])
                cross_res = center_cosine(residual[positive], residual[negative])
                delta = pos[0] - neg[0]
                residual_delta = pos_res[0] - neg_res[0]
                row = {
                    "query": query,
                    "variant": variant,
                    "timestep": int(timestep),
                    "positive_probes": ",".join(f"P{x + 1}" for x in positive),
                    "negative_probes": ",".join(f"P{x + 1}" for x in negative),
                    "positive_count": len(positive),
                    "negative_count": len(negative),
                    "positive_pair_mean_cos": pos[0],
                    "negative_pair_mean_cos": neg[0],
                    "positive_minus_negative_cos": delta,
                    "positive_negative_center_cos": cross,
                    "pc1_explained_fraction": explained,
                    "residual_positive_pair_mean_cos": pos_res[0],
                    "residual_negative_pair_mean_cos": neg_res[0],
                    "residual_positive_minus_negative_cos": residual_delta,
                    "residual_positive_negative_center_cos": cross_res,
                    "positive_concentration": pos[2],
                    "negative_concentration": neg[2],
                    "residual_positive_concentration": pos_res[2],
                    "residual_negative_concentration": neg_res[2],
                }
                rows.append(row)
                print(
                    f"{query:2d} {variant:>14} {int(timestep):4d} "
                    f"{len(positive):3d} {len(negative):3d} "
                    f"{pos[0]:+8.4f} {neg[0]:+8.4f} {delta:+8.4f} "
                    f"{cross:+8.4f} {100.0 * explained:6.2f}% "
                    f"{pos_res[0]:+8.4f} {neg_res[0]:+8.4f} "
                    f"{residual_delta:+8.4f} {cross_res:+8.4f}"
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "positive_negative_concentration.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
