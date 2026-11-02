"""Find the two LDS subset models with the closest simple loss per query.

This is a cheap analysis over the saved [query, subset] observed arrays.  The
endpoint and trajectory columns here are the existing scalar true-f metrics
(deviation from the base query), not direct pairwise state-space MSE.  Use
308/309 for the direct pairwise endpoint/trajectory comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from exp_config import LDS_DIR


def nearest_pair(values: np.ndarray) -> tuple[int, int]:
    order = np.argsort(values, kind="stable")
    gaps = np.diff(values[order])
    position = int(np.argmin(gaps))
    return int(order[position]), int(order[position + 1])


def percentile_among_pair_gaps(values: np.ndarray, i: int, j: int) -> float:
    rows, cols = np.triu_indices(values.size, k=1)
    all_gaps = np.abs(values[rows] - values[cols])
    selected = abs(float(values[i]) - float(values[j]))
    return float(np.mean(all_gaps <= selected))


def pair_gap_correlation(x: np.ndarray, y: np.ndarray) -> float:
    rows, cols = np.triu_indices(x.size, k=1)
    x_gap = np.abs(x[rows] - x[cols])
    y_gap = np.abs(y[rows] - y[cols])
    x_rank = np.argsort(np.argsort(x_gap, kind="stable"), kind="stable")
    y_rank = np.argsort(np.argsort(y_gap, kind="stable"), kind="stable")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def load_metric(name: str) -> np.ndarray:
    path = LDS_DIR / f"observed_{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.load(path).astype(np.float64)
    if values.ndim != 2 or values.shape[1] != 192:
        raise ValueError(f"expected [queries, 192] at {path}, got {values.shape}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="ema,raw")
    parser.add_argument("--query-ids", default="0-99")
    args = parser.parse_args()

    if "-" in args.query_ids and "," not in args.query_ids:
        lo, hi = (int(value) for value in args.query_ids.split("-", 1))
        query_ids = list(range(lo, hi + 1))
    else:
        query_ids = [int(value) for value in args.query_ids.split(",")]

    output: dict[str, object] = {
        "definition": {
            "selection": "minimum absolute simple-loss gap among 192 subsets",
            "endpoint": "absolute gap in endpoint_deviation true-f scalar",
            "trajectory": "absolute gap in trajectory_state_mse true-f scalar",
        },
        "sources": {},
    }
    lines: list[str] = [
        "NEAREST SIMPLE-LOSS PAIR AMONG 192 LDS SUBSET MODELS",
        "Endpoint/trajectory below are gaps between existing true-f scalars.",
        "Direct pairwise state MSE is produced by scripts 308/309.",
        "",
    ]

    for source in [value.strip() for value in args.sources.split(",") if value.strip()]:
        loss_path = LDS_DIR / f"observed_simple_loss_{source}_100t_mc4.npy"
        if not loss_path.is_file():
            loss_path = LDS_DIR / f"observed_simple_loss_{source}.npy"
        loss = np.load(loss_path).astype(np.float64)
        endpoint = load_metric(f"endpoint_deviation_{source}")
        trajectory = load_metric(f"trajectory_state_mse_{source}")
        if loss.shape != endpoint.shape or loss.shape != trajectory.shape:
            raise ValueError(
                f"shape mismatch loss={loss.shape}, endpoint={endpoint.shape}, "
                f"trajectory={trajectory.shape}"
            )

        records = []
        lines.extend(
            [
                f"SOURCE: {source.upper()}  loss={loss_path.name}",
                "query pair      loss-a       loss-b      loss-gap    "
                "endpoint-gap endpoint-pct trajectory-gap trajectory-pct",
            ]
        )
        for query_id in query_ids:
            i, j = nearest_pair(loss[query_id])
            record = {
                "query_id": query_id,
                "subset_a": i,
                "subset_b": j,
                "loss_a": float(loss[query_id, i]),
                "loss_b": float(loss[query_id, j]),
                "loss_gap": abs(float(loss[query_id, i] - loss[query_id, j])),
                "endpoint_a": float(endpoint[query_id, i]),
                "endpoint_b": float(endpoint[query_id, j]),
                "endpoint_gap": abs(float(endpoint[query_id, i] - endpoint[query_id, j])),
                "endpoint_gap_percentile": percentile_among_pair_gaps(
                    endpoint[query_id], i, j
                ),
                "trajectory_a": float(trajectory[query_id, i]),
                "trajectory_b": float(trajectory[query_id, j]),
                "trajectory_gap": abs(
                    float(trajectory[query_id, i] - trajectory[query_id, j])
                ),
                "trajectory_gap_percentile": percentile_among_pair_gaps(
                    trajectory[query_id], i, j
                ),
                "all_pair_loss_endpoint_spearman": pair_gap_correlation(
                    loss[query_id], endpoint[query_id]
                ),
                "all_pair_loss_trajectory_spearman": pair_gap_correlation(
                    loss[query_id], trajectory[query_id]
                ),
            }
            records.append(record)
            lines.append(
                f"q{query_id:02d}   {i:03d}/{j:03d} "
                f"{record['loss_a']:12.6e} {record['loss_b']:12.6e} "
                f"{record['loss_gap']:11.3e} {record['endpoint_gap']:12.3e} "
                f"{100 * record['endpoint_gap_percentile']:10.2f}% "
                f"{record['trajectory_gap']:14.3e} "
                f"{100 * record['trajectory_gap_percentile']:12.2f}%"
            )

        summary = {
            key: float(np.mean([record[key] for record in records]))
            for key in (
                "loss_gap",
                "endpoint_gap",
                "endpoint_gap_percentile",
                "trajectory_gap",
                "trajectory_gap_percentile",
                "all_pair_loss_endpoint_spearman",
                "all_pair_loss_trajectory_spearman",
            )
        }
        output["sources"][source] = {"summary": summary, "queries": records}
        lines.extend(
            [
                "MEAN",
                f"  loss gap                    {summary['loss_gap']:.6e}",
                f"  endpoint scalar gap         {summary['endpoint_gap']:.6e}",
                f"  endpoint gap percentile     {100 * summary['endpoint_gap_percentile']:.2f}%",
                f"  trajectory scalar gap       {summary['trajectory_gap']:.6e}",
                f"  trajectory gap percentile   {100 * summary['trajectory_gap_percentile']:.2f}%",
                f"  all-pair loss/endpoint rho  {summary['all_pair_loss_endpoint_spearman']:+.4f}",
                f"  all-pair loss/traj rho      {summary['all_pair_loss_trajectory_spearman']:+.4f}",
                "",
            ]
        )

    txt_path = LDS_DIR / "nearest_simple_loss_pair_scalar_comparison.txt"
    json_path = LDS_DIR / "nearest_simple_loss_pair_scalar_comparison.json"
    txt_path.write_text("\n".join(lines) + "\n")
    json_path.write_text(json.dumps(output, indent=2) + "\n")
    print("\n".join(lines))
    print(f"[saved] {txt_path}")
    print(f"[saved] {json_path}")


if __name__ == "__main__":
    main()
