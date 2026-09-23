#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys


SCHEMES = {
    "traj_tracin_adamw_residual_aligned10x10_own100q_from100t": "residual",
    "traj_tracin_adamw_full_aligned10x10_own100q_from100t": "full",
}
VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")
TARGETS = (
    "endpoint_contarfactual",
    "traj_contarfactual",
    "simple_loss",
    "noise_trajectory",
)


def parse_args() -> argparse.Namespace:
    shapes_root = Path(__file__).resolve().parents[1]
    default_eval_root = shapes_root / "result" / "experiment1" / "eval"
    parser = argparse.ArgumentParser(
        description="Summarize aligned 10x10 AdamW LDS across 100 own-trajectory queries."
    )
    parser.add_argument("--eval-root", type=Path, default=default_eval_root)
    parser.add_argument(
        "--query-file",
        type=Path,
        default=shapes_root / "queries_seed_0_99.json",
    )
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--print-per-query", action="store_true")
    return parser.parse_args()


def load_rows(
    eval_root: Path, query_file: Path, num_queries: int, workers: int
) -> list[dict[str, object]]:
    shapes_root = Path(__file__).resolve().parents[1]
    if str(shapes_root) not in sys.path:
        sys.path.insert(0, str(shapes_root))
    from dataset_config import _prompt_tag

    records = json.loads(query_file.read_text())["queries"]
    if len(records) < num_queries:
        raise ValueError(
            f"Query manifest contains {len(records)} records, fewer than {num_queries}"
        )
    tasks: list[tuple[int, str, str, str, Path]] = []
    for query_id, record in enumerate(records[:num_queries]):
        prompt_tag = _prompt_tag(str(record["prompt"]))
        seed = int(record["initial_seed"])
        query_root = (
            eval_root
            / "prompted_solo"
            / f"query_{prompt_tag}"
            / f"initial_seed_{seed}"
            / "lds"
        )
        for base, short_name in SCHEMES.items():
            for variant in VARIANTS:
                for target in TARGETS:
                    target_root = (
                        query_root
                        / f"{base}_{variant}"
                        / target
                        / "pred_kept_sign_p1"
                    )
                    tasks.append((query_id, target, short_name, variant, target_root))

    def load_one(task: tuple[int, str, str, str, Path]) -> dict[str, object]:
        query_id, target, scheme, variant, target_root = task
        matches = list(target_root.glob("*/lds_summary.json"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one LDS summary for query={query_id}, target={target}, "
                f"scheme={scheme}, variant={variant}; found {len(matches)} under {target_root}"
            )
        path = matches[0]
        payload = json.loads(path.read_text())
        return {
            "query": query_id,
            "target": target,
            "scheme": scheme,
            "variant": variant,
            "lds_percent": float(payload["lds_percent"]),
            "path": str(path),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(load_one, tasks))
    rows.sort(
        key=lambda row: (
            int(row["query"]),
            str(row["target"]),
            str(row["scheme"]),
            str(row["variant"]),
        )
    )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.num_queries <= 0:
        raise ValueError("--num-queries must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    eval_root = args.eval_root.resolve()
    rows = load_rows(
        eval_root,
        args.query_file.resolve(),
        args.num_queries,
        args.workers,
    )
    expected = args.num_queries * 4 * len(SCHEMES) * len(VARIANTS)
    print(f"Loaded {len(rows)}/{expected} LDS results")
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} results but found {len(rows)}")

    out_root = eval_root / "aligned10x10_own100q_from100t_summary"
    out_root.mkdir(parents=True, exist_ok=True)
    long_path = out_root / "per_query_long.csv"
    write_csv(long_path, list(rows[0]), rows)

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = float(row["lds_percent"])
        if math.isfinite(value):
            grouped[
                (str(row["target"]), str(row["scheme"]), str(row["variant"]))
            ].append(value)

    summary_rows: list[dict[str, object]] = []
    for (target, scheme, variant), values in sorted(grouped.items()):
        summary_rows.append(
            {
                "target": target,
                "scheme": scheme,
                "variant": variant,
                "n": len(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        )
    summary_path = out_root / "summary.csv"
    write_csv(
        summary_path,
        ["target", "scheme", "variant", "n", "mean", "std", "min", "max"],
        summary_rows,
    )

    column_keys = sorted(
        {
            (str(row["scheme"]), str(row["variant"]), str(row["target"]))
            for row in rows
        }
    )
    lookup = {
        (
            int(row["query"]),
            str(row["scheme"]),
            str(row["variant"]),
            str(row["target"]),
        ): float(row["lds_percent"])
        for row in rows
    }
    wide_fields = ["query"] + ["__".join(key) for key in column_keys]
    wide_rows: list[dict[str, object]] = []
    for query in range(args.num_queries):
        wide_row: dict[str, object] = {"query": query}
        for scheme, variant, target in column_keys:
            wide_row[f"{scheme}__{variant}__{target}"] = lookup[
                (query, scheme, variant, target)
            ]
        wide_rows.append(wide_row)
    wide_path = out_root / "per_query_wide.csv"
    write_csv(wide_path, wide_fields, wide_rows)

    print(
        f"{'target':22s} {'scheme':9s} {'variant':15s} "
        f"{'n':>4s} {'mean':>9s} {'std':>9s} {'min':>9s} {'max':>9s}"
    )
    print("-" * 92)
    for row in summary_rows:
        print(
            f"{str(row['target']):22s} {str(row['scheme']):9s} "
            f"{str(row['variant']):15s} {int(row['n']):4d} "
            f"{float(row['mean']):+9.3f} {float(row['std']):9.3f} "
            f"{float(row['min']):+9.3f} {float(row['max']):+9.3f}"
        )
    if args.print_per_query:
        print()
        for row in rows:
            print(
                f"q={int(row['query']):02d} {str(row['target']):22s} "
                f"{str(row['scheme']):9s} {str(row['variant']):15s} "
                f"{float(row['lds_percent']):+9.3f}"
            )
    print(f"Saved summary: {summary_path}")
    print(f"Saved long per-query table: {long_path}")
    print(f"Saved wide per-query table: {wide_path}")


if __name__ == "__main__":
    main()
