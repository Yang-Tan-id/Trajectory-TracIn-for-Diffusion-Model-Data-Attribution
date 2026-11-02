"""Print the five closest simple-loss subset pairs for selected queries."""

from __future__ import annotations

import argparse
import json

import numpy as np

from exp_config import LDS_DIR


DEFAULT_QUERY_IDS = (7, 30, 67, 71, 72, 82)


def load(name: str) -> np.ndarray:
    path = LDS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.load(path).astype(np.float64)
    if values.ndim != 2 or values.shape[1] != 192:
        raise ValueError(f"expected [queries, 192] at {path}, got {values.shape}")
    return values


def all_pair_gaps(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, right = np.triu_indices(values.size, k=1)
    return left, right, np.abs(values[left] - values[right])


def percentile(gaps: np.ndarray, selected: float) -> float:
    return float(np.mean(gaps <= selected))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("ema", "raw"), default="ema")
    parser.add_argument(
        "--query-ids",
        default=",".join(str(value) for value in DEFAULT_QUERY_IDS),
    )
    parser.add_argument("--top-pairs", type=int, default=5)
    args = parser.parse_args()
    if args.top_pairs <= 0:
        raise ValueError("--top-pairs must be positive")

    query_ids = [int(value) for value in args.query_ids.split(",")]
    loss_name = f"observed_simple_loss_{args.source}_100t_mc4.npy"
    if not (LDS_DIR / loss_name).is_file():
        loss_name = f"observed_simple_loss_{args.source}.npy"
    loss = load(loss_name)
    endpoint = load(f"observed_endpoint_deviation_{args.source}.npy")
    trajectory = load(f"observed_trajectory_state_mse_{args.source}.npy")
    if loss.shape != endpoint.shape or loss.shape != trajectory.shape:
        raise ValueError("observed array shapes do not match")

    output = {
        "source": args.source,
        "loss_file": loss_name,
        "top_pairs": args.top_pairs,
        "queries": {},
    }
    lines = [
        f"TOP-{args.top_pairs} NEAREST SIMPLE-LOSS PAIRS — {args.source.upper()}",
        f"loss={loss_name}",
        "Percentiles are among all C(192,2)=18,336 pairs for the same query.",
        "",
    ]

    for query_id in query_ids:
        left, right, loss_gaps = all_pair_gaps(loss[query_id])
        _, _, endpoint_gaps = all_pair_gaps(endpoint[query_id])
        _, _, trajectory_gaps = all_pair_gaps(trajectory[query_id])
        order = np.argsort(loss_gaps, kind="stable")[: args.top_pairs]
        records = []
        lines.extend(
            [
                f"QUERY q{query_id:02d}",
                "rank pair      loss-a       loss-b      loss-gap    "
                "endpoint-gap endpoint-pct trajectory-gap trajectory-pct",
            ]
        )
        for rank, pair_position in enumerate(order, start=1):
            subset_a = int(left[pair_position])
            subset_b = int(right[pair_position])
            record = {
                "rank": rank,
                "subset_a": subset_a,
                "subset_b": subset_b,
                "loss_a": float(loss[query_id, subset_a]),
                "loss_b": float(loss[query_id, subset_b]),
                "loss_gap": float(loss_gaps[pair_position]),
                "endpoint_gap": float(endpoint_gaps[pair_position]),
                "endpoint_percentile": percentile(
                    endpoint_gaps, float(endpoint_gaps[pair_position])
                ),
                "trajectory_gap": float(trajectory_gaps[pair_position]),
                "trajectory_percentile": percentile(
                    trajectory_gaps, float(trajectory_gaps[pair_position])
                ),
            }
            records.append(record)
            lines.append(
                f"{rank:>4d} {subset_a:03d}/{subset_b:03d} "
                f"{record['loss_a']:12.6e} {record['loss_b']:12.6e} "
                f"{record['loss_gap']:11.3e} {record['endpoint_gap']:12.3e} "
                f"{100 * record['endpoint_percentile']:10.2f}% "
                f"{record['trajectory_gap']:14.3e} "
                f"{100 * record['trajectory_percentile']:12.2f}%"
            )

        best_endpoint = max(records, key=lambda item: item["endpoint_gap"])
        best_trajectory = max(records, key=lambda item: item["trajectory_gap"])
        lines.extend(
            [
                f"  largest endpoint gap in top-{args.top_pairs}: "
                f"rank {best_endpoint['rank']} "
                f"({best_endpoint['subset_a']:03d}/{best_endpoint['subset_b']:03d})",
                f"  largest trajectory gap in top-{args.top_pairs}: "
                f"rank {best_trajectory['rank']} "
                f"({best_trajectory['subset_a']:03d}/{best_trajectory['subset_b']:03d})",
                "",
            ]
        )
        output["queries"][f"q{query_id:02d}"] = records

    stem = (
        f"top{args.top_pairs}_nearest_simple_loss_pairs_"
        f"{args.source}_q" + "_q".join(f"{query_id:02d}" for query_id in query_ids)
    )
    txt_path = LDS_DIR / f"{stem}.txt"
    json_path = LDS_DIR / f"{stem}.json"
    txt_path.write_text("\n".join(lines) + "\n")
    json_path.write_text(json.dumps(output, indent=2) + "\n")
    print("\n".join(lines))
    print(f"[saved] {txt_path}")
    print(f"[saved] {json_path}")


if __name__ == "__main__":
    main()
