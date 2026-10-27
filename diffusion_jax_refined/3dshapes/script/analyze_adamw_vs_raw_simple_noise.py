#!/usr/bin/env python3
"""Compare Shape3D raw-gradient and AdamW LDS on Simple/Noise targets."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from dataset_config import _prompt_tag


TARGETS = ("simple_loss", "noise_trajectory")
STANDARD_LDS_GROUP = (
    "m_64_k_2500_subset_seed_0__"
    "m_64_k_2500_subset_seed_1__"
    "m_64_k_2500_subset_seed_2"
)


@dataclass(frozen=True)
class Method:
    label: str
    raw_namespace: str
    adamw_namespace: str
    variant: str
    raw_sign: str
    adamw_sign: str
    raw_multiplier: float
    adamw_multiplier: float


# The cached linear TracIn/ReTrac LDS runs use m1 for raw gradients and p1 for
# AdamW. Raw-gradient values already have the reporting orientation; AdamW
# values are multiplied by -1 to remove the optimizer update's descent minus
# sign. Endpoint-Pollute is squared and uses p1 without a sign correction.
METHODS = (
    Method(
        label="PINT / RAW",
        raw_namespace=(
            "traj_tracin_recreate_raw_polluted_endpoint_delta_l2normalized_"
            "timestamp_sum_squared_q0_99"
        ),
        adamw_namespace=(
            "traj_tracin_recreate_adamw_full_polluted_endpoint_"
            "delta_l2normalized_timestamp_sum_squared_q0_99"
        ),
        variant="raw",
        raw_sign="p1",
        adamw_sign="p1",
        raw_multiplier=1.0,
        adamw_multiplier=1.0,
    ),
    Method(
        label="PINT / BOTH-L2",
        raw_namespace=(
            "traj_tracin_recreate_raw_polluted_endpoint_delta_l2normalized_"
            "timestamp_sum_squared_q0_99"
        ),
        adamw_namespace=(
            "traj_tracin_recreate_adamw_full_polluted_endpoint_"
            "delta_l2normalized_timestamp_sum_squared_q0_99"
        ),
        variant="query_train_l2",
        raw_sign="p1",
        adamw_sign="p1",
        raw_multiplier=1.0,
        adamw_multiplier=1.0,
    ),
    Method(
        label="TRACIN / RAW",
        raw_namespace="traj_tracin_endpoint_tracin_train100x1_query100x1_q0_99",
        adamw_namespace=(
            "traj_tracin_endpoint_tracin_adamw_full_"
            "train100x1_query100x1_q0_99"
        ),
        variant="raw",
        raw_sign="m1",
        adamw_sign="p1",
        raw_multiplier=1.0,
        adamw_multiplier=-1.0,
    ),
    Method(
        label="TRACIN / BOTH-L2",
        raw_namespace="traj_tracin_endpoint_tracin_train100x1_query100x1_q0_99",
        adamw_namespace=(
            "traj_tracin_endpoint_tracin_adamw_full_"
            "train100x1_query100x1_q0_99"
        ),
        variant="query_train_l2",
        raw_sign="m1",
        adamw_sign="p1",
        raw_multiplier=1.0,
        adamw_multiplier=-1.0,
    ),
    Method(
        label="RETRAC / BOTH-L2",
        raw_namespace="traj_tracin_paper_retrac_exact4_endpoint100x1_q0_99",
        adamw_namespace=(
            "traj_tracin_paper_retrac_adamw_full_exact4_"
            "endpoint100x1_q0_99"
        ),
        variant="query_train_l2",
        raw_sign="m1",
        adamw_sign="p1",
        raw_multiplier=1.0,
        adamw_multiplier=-1.0,
    ),
)


def parse_ids(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def summary_path(
    *,
    eval_root: Path,
    prompt: str,
    initial_seed: int,
    namespace: str,
    variant: str,
    target: str,
    sign: str,
) -> Path:
    return (
        eval_root
        / f"query_{_prompt_tag(prompt)}"
        / f"initial_seed_{initial_seed}"
        / "lds"
        / f"{namespace}_{variant}"
        / target
        / f"pred_kept_sign_{sign}"
        / STANDARD_LDS_GROUP
        / "lds_summary.json"
    )


def resolve_summary(path: Path) -> Path:
    if path.is_file():
        return path
    target_root = path.parents[1]
    matches = sorted(target_root.glob("*/lds_summary.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one LDS summary under {target_root}; "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def load_percent(path: Path) -> float:
    path = resolve_summary(path)
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid LDS summary JSON: {path}: {error}") from error
    value = float(payload["lds_percent"])
    if not np.isfinite(value):
        raise RuntimeError(f"non-finite LDS value in {path}: {value}")
    return value


def paired_label_swap(
    differences: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> tuple[float, float]:
    differences = np.asarray(differences, dtype=np.float64)
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    positive_extreme = 0
    absolute_extreme = 0
    completed = 0
    chunk_size = 20_000
    while completed < permutations:
        current = min(chunk_size, permutations - completed)
        signs = rng.integers(
            0, 2, size=(current, len(differences)), dtype=np.int8
        ).astype(np.float64)
        signs = signs * 2.0 - 1.0
        null_means = np.mean(signs * differences[None, :], axis=1)
        positive_extreme += int(np.count_nonzero(null_means >= observed))
        absolute_extreme += int(
            np.count_nonzero(np.abs(null_means) >= abs(observed))
        )
        completed += current
    denominator = permutations + 1.0
    return (
        (positive_extreme + 1.0) / denominator,
        (absolute_extreme + 1.0) / denominator,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument(
        "--query-file",
        type=Path,
        default=(
            SHAPES_ROOT
            / "queries_in_distribution_plus_zero_seed_100_219.json"
        ),
    )
    parser.add_argument("--query-ids", default=",".join(map(str, range(100))))
    parser.add_argument("--permutations", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.permutations <= 0:
        parser.error("--permutations must be positive")
    records = json.loads(args.query_file.read_text())["queries"]
    query_ids = parse_ids(args.query_ids)
    if not query_ids or len(query_ids) != len(set(query_ids)):
        parser.error("--query-ids must contain unique query IDs")
    if min(query_ids) < 0 or max(query_ids) >= len(records):
        parser.error(f"query IDs must be within [0, {len(records) - 1}]")

    result_root = SHAPES_ROOT / "result" / args.experiment
    eval_root = result_root / "eval" / "prompted_solo"
    output_dir = args.output_dir or (
        result_root / "analysis" / "adamw_vs_raw_simple_noise_q0_99"
    )

    per_query_rows: list[dict[str, object]] = []
    values: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for method in METHODS:
        for target in TARGETS:
            raw_values = []
            adamw_values = []
            for query_id in query_ids:
                record = records[query_id]
                prompt = str(record["prompt"])
                initial_seed = int(record.get("initial_seed", record.get("seed")))
                raw_path = summary_path(
                    eval_root=eval_root,
                    prompt=prompt,
                    initial_seed=initial_seed,
                    namespace=method.raw_namespace,
                    variant=method.variant,
                    target=target,
                    sign=method.raw_sign,
                )
                adamw_path = summary_path(
                    eval_root=eval_root,
                    prompt=prompt,
                    initial_seed=initial_seed,
                    namespace=method.adamw_namespace,
                    variant=method.variant,
                    target=target,
                    sign=method.adamw_sign,
                )
                raw_stored_value = load_percent(raw_path)
                adamw_stored_value = load_percent(adamw_path)
                raw_value = method.raw_multiplier * raw_stored_value
                adamw_value = method.adamw_multiplier * adamw_stored_value
                raw_values.append(raw_value)
                adamw_values.append(adamw_value)
                per_query_rows.append(
                    {
                        "query_id": query_id,
                        "method": method.label,
                        "target": target,
                        "raw_sign": method.raw_sign,
                        "adamw_sign": method.adamw_sign,
                        "raw_multiplier": method.raw_multiplier,
                        "adamw_multiplier": method.adamw_multiplier,
                        "raw_stored_lds_percent": raw_stored_value,
                        "adamw_stored_lds_percent": adamw_stored_value,
                        "raw_lds_percent": raw_value,
                        "adamw_lds_percent": adamw_value,
                        "delta_percent_points": adamw_value - raw_value,
                    }
                )
            values[(method.label, target)] = (
                np.asarray(raw_values, dtype=np.float64),
                np.asarray(adamw_values, dtype=np.float64),
            )

    lines = [
        "SHAPE3D Q0-Q99 — ADAMW-FULL VS RAW-GRADIENT LDS",
        "Difference = AdamW-full minus Raw-gradient; larger LDS is better.",
        (
            "Cached signs: Pollute raw/AdamW=p1/p1; "
            "TracIn and ReTrac raw/AdamW=m1/p1."
        ),
        (
            "Reporting multipliers: Pollute raw/AdamW=+1/+1; "
            "TracIn and ReTrac raw/AdamW=+1/-1."
        ),
        f"Paired label-swap Monte Carlo: {args.permutations:,} permutations.",
        "",
    ]
    summary_rows: list[dict[str, object]] = []
    for target_index, target in enumerate(TARGETS):
        lines.extend(
            [
                "=" * 120,
                target.upper(),
                "=" * 120,
                (
                    f"{'METHOD':<24s}{'N':>6s}{'RAW MEAN':>14s}"
                    f"{'ADAMW MEAN':>14s}{'DELTA':>14s}{'WINS':>10s}"
                    f"{'P1':>14s}{'P2':>14s}"
                ),
                "-" * 120,
            ]
        )
        for method_index, method in enumerate(METHODS):
            raw, adamw = values[(method.label, target)]
            difference = adamw - raw
            wins = int(np.count_nonzero(difference > 0.0))
            ties = int(np.count_nonzero(difference == 0.0))
            p1, p2 = paired_label_swap(
                difference,
                permutations=args.permutations,
                seed=args.seed + target_index * len(METHODS) + method_index,
            )
            wins_text = f"{wins}/{len(difference)}"
            if ties:
                wins_text += f" T{ties}"
            lines.append(
                f"{method.label:<24s}{len(difference):>6d}"
                f"{raw.mean():>+13.3f}%{adamw.mean():>+13.3f}%"
                f"{difference.mean():>+13.3f}%{wins_text:>10s}"
                f"{p1:>14.6g}{p2:>14.6g}"
            )
            summary_rows.append(
                {
                    "target": target,
                    "method": method.label,
                    "query_count": len(difference),
                    "raw_mean_percent": float(raw.mean()),
                    "adamw_mean_percent": float(adamw.mean()),
                    "delta_percent_points": float(difference.mean()),
                    "wins": wins,
                    "ties": ties,
                    "p_one_sided_improvement": p1,
                    "p_two_sided": p2,
                }
            )
        lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "adamw_vs_raw_report.txt"
    report_path.write_text("\n".join(lines) + "\n")
    write_csv(output_dir / "adamw_vs_raw_summary.csv", summary_rows)
    write_csv(output_dir / "adamw_vs_raw_per_query.csv", per_query_rows)
    print("\n".join(lines))
    print(f"[saved] {report_path}")
    print(f"[saved] {output_dir / 'adamw_vs_raw_summary.csv'}")
    print(f"[saved] {output_dir / 'adamw_vs_raw_per_query.csv'}")


if __name__ == "__main__":
    main()
