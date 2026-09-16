#!/usr/bin/env python3
"""Measure projected probe-gradient angles using leave-one-probe-out references."""

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

from run_predicted_noise_jvp_l2_squared import load_query_bank
from analyze_predicted_noise_angle_oriented_scores import load_alignments, query_args


BANKS = {
    "1-4": (0, 1, 2, 3),
    "5-8": (4, 5, 6, 7),
    "9-12": (8, 9, 10, 11),
    "1-12": tuple(range(12)),
}


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.einsum("qtd,qtd->qt", left, right, optimize=True)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


def stats(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean_cosine": float(flat.mean()),
        "mean_abs_cosine": float(np.abs(flat).mean()),
        "positive_fraction": float(np.mean(flat > 0.0)),
        "std_cosine": float(flat.std()),
    }


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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num-probes", type=int, default=12)
    parser.add_argument("--source-run-id", default="3503654")
    parser.add_argument(
        "--query-namespace-pattern",
        default="loss_direction_residual_rms_predicted_noise_probe4_r{probe_index}",
    )
    parser.add_argument(
        "--alignment-namespace",
        default="predicted_noise_alignment_probe12",
    )
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.num_probes != 12:
        raise ValueError("the current saved bank is expected to contain 12 probes")

    query, meta = load_query_bank(query_args(args))
    scalars, output_cosines = load_alignments(args, meta)
    signs = np.where(scalars >= 0.0, 1.0, -1.0).astype(np.float32)

    # Each total is (query, term, projected_dimension). The probe under test is
    # removed below, avoiding the guaranteed positive self-dot-product.
    energy_total = np.einsum(
        "pqt,pqtd->qtd", scalars, query, optimize=True
    ).astype(np.float32)
    norm_total = np.einsum(
        "pqt,pqtd->qtd", signs, query, optimize=True
    ).astype(np.float32)

    metric_values: dict[tuple[int, str], np.ndarray] = {}
    per_query_rows = []
    probe_rows = []
    for probe_index in range(args.num_probes):
        q_probe = query[probe_index]
        oriented_probe = signs[probe_index, :, :, None] * q_probe
        energy_reference = (
            energy_total
            - scalars[probe_index, :, :, None] * q_probe
        ) / float(args.num_probes - 1)
        norm_reference = (
            norm_total
            - signs[probe_index, :, :, None] * q_probe
        ) / float(args.num_probes - 1)

        metrics = {
            "raw_to_energy_loo": cosine_rows(q_probe, energy_reference),
            "oriented_to_energy_loo": cosine_rows(
                oriented_probe, energy_reference
            ),
            "raw_to_norm_loo": cosine_rows(q_probe, norm_reference),
            "oriented_to_norm_loo": cosine_rows(oriented_probe, norm_reference),
        }
        for metric, values in metrics.items():
            metric_values[(probe_index, metric)] = values
            row = {"probe": probe_index + 1, "metric": metric, **stats(values)}
            probe_rows.append(row)
            for query_id in range(values.shape[0]):
                per_query_rows.append(
                    {
                        "probe": probe_index + 1,
                        "query": query_id,
                        "metric": metric,
                        **stats(values[query_id]),
                    }
                )

    bank_rows = []
    metric_names = sorted({metric for _, metric in metric_values})
    for bank_name, indices in BANKS.items():
        for metric in metric_names:
            values = np.stack(
                [metric_values[(probe_index, metric)] for probe_index in indices]
            )
            bank_rows.append({"bank": bank_name, "metric": metric, **stats(values)})

    result_root = SHAPES_ROOT / "result" / args.experiment
    out_dir = args.out_dir or (
        result_root
        / "eval"
        / "predicted_noise_angle_oriented_probe12"
        / f"run_{args.source_run_id}"
        / "projected_angle"
    )
    write_csv(out_dir / "per_probe.csv", probe_rows)
    write_csv(out_dir / "per_query.csv", per_query_rows)
    write_csv(out_dir / "by_bank.csv", bank_rows)
    np.savez_compressed(
        out_dir / "raw_output_angle_reference.npz",
        probe_cosines=output_cosines,
        probe_scalars=scalars,
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "definition": {
                    "energy_reference": "mean_{s != r}(f_s * P J^T v_s)",
                    "norm_reference": "mean_{s != r}(sign(f_s) * P J^T v_s)",
                    "f": "dot(v, predicted_noise) / sqrt(output_dimension)",
                },
                "leave_one_out": True,
                "num_probes": args.num_probes,
                "num_queries": 10,
                "num_terms": 500,
                "projected_dimension": 4096,
            },
            indent=2,
            sort_keys=True,
        )
    )

    print("PROJECTED QUERY-GRADIENT ANGLES — LEAVE ONE PROBE OUT")
    print(
        f"{'BANK':8s} {'METRIC':28s} {'MEAN COS':>10s} "
        f"{'MEAN |COS|':>11s} {'POS FRAC':>10s} {'STD':>10s}"
    )
    print("-" * 86)
    for row in bank_rows:
        print(
            f"{str(row['bank']):8s} {str(row['metric']):28s} "
            f"{float(row['mean_cosine']):+10.5f} "
            f"{float(row['mean_abs_cosine']):10.5f} "
            f"{float(row['positive_fraction']):10.4f} "
            f"{float(row['std_cosine']):10.5f}"
        )
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
