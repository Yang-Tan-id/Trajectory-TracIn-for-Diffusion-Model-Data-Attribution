#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


CELEBA_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = CELEBA_ROOT.parent
for path in (CELEBA_ROOT, REFINE_ROOT, REFINE_ROOT / "legacy_jax"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_config_attr40 import _prompt_path_tag
from dataset_config_das_attr40 import DAS_DAMPING_SWEEP_VALUES
from LDS.DM_cifar_lds import spearman_corr


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.replace(",", " ").split() if value]


def lambda_tag(value: float) -> str:
    return f"{float(value):g}".replace("+", "").replace("-", "neg_").replace(".", "p")


def rank_membership(
    score_indices: np.ndarray,
    subset_rows: list[dict[str, str]],
    lds_model_root: Path,
) -> np.ndarray:
    score_indices = np.asarray(score_indices, dtype=np.int64).reshape(-1)
    if len(np.unique(score_indices)) != len(score_indices):
        raise ValueError("DAS score indices contain duplicates")
    order = np.argsort(score_indices)
    sorted_indices = score_indices[order]
    membership = np.zeros((len(subset_rows), len(score_indices)), dtype=np.float64)
    for row_number, row in enumerate(subset_rows):
        source = Path(row["subset_dir"])
        if not source.is_dir():
            parts = source.parts
            if "lds_model" not in parts:
                raise FileNotFoundError(source)
            source = lds_model_root.joinpath(*parts[parts.index("lds_model") + 1 :])
        kept = np.asarray(
            np.load(source / "kept_attribution_indices.npy"), dtype=np.int64
        ).reshape(-1)
        positions = np.searchsorted(sorted_indices, kept)
        valid = positions < len(sorted_indices)
        positions = positions[valid]
        kept = kept[valid]
        matched = sorted_indices[positions] == kept
        membership[row_number, order[positions[matched]]] = 1.0
    return membership


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CelebA DAS lambda-sweep LDS from cached 192-model true-f values."
    )
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", default=",".join(str(i) for i in range(100)))
    parser.add_argument("--namespace", default="das_50x1_mc1")
    parser.add_argument("--lambdas", default="")
    parser.add_argument("--true-f-csv", type=Path, default=None)
    parser.add_argument("--target-functions", default="")
    parser.add_argument("--prediction-sign", type=float, default=-1.0)
    parser.add_argument("--expected-points", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    query_ids = parse_ints(args.query_ids)
    lambdas = (
        parse_floats(args.lambdas)
        if args.lambdas.strip()
        else [float(value) for value in DAS_DAMPING_SWEEP_VALUES]
    )
    requested_targets = {
        value for value in args.target_functions.replace(",", " ").split() if value
    }
    records = json.loads(args.query_file.read_text())["queries"]
    query_records = {int(record.get("query_id", i)): record for i, record in enumerate(records)}
    missing_queries = sorted(set(query_ids) - set(query_records))
    if missing_queries:
        raise ValueError(f"query manifest is missing IDs: {missing_queries}")

    result_root = CELEBA_ROOT / "result" / args.experiment
    true_f_csv = args.true_f_csv or (
        result_root / "query100_lds_csv_ddim50"
        / "celeba_attr40_query100_lds_true_f_long.csv"
    )
    if not true_f_csv.is_file():
        raise FileNotFoundError(true_f_csv)
    with true_f_csv.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    grouped: dict[tuple[int, str], list[dict[str, str]]] = {}
    for row in source_rows:
        qid = int(row["query_id"])
        if qid not in query_ids:
            continue
        target = row["target_function"]
        if requested_targets and target not in requested_targets:
            continue
        grouped.setdefault((qid, target), []).append(row)
    for key, rows in grouped.items():
        rows.sort(key=lambda row: int(row["global_subset_id"]))
        if [int(row["global_subset_id"]) for row in rows] != list(range(192)):
            raise ValueError(f"{key} has {len(rows)}/192 ordered subset rows")
    targets = sorted({target for _, target in grouped})
    for qid in query_ids:
        absent = [target for target in targets if (qid, target) not in grouped]
        if absent:
            raise ValueError(f"Q{qid} is missing cached true-f targets: {absent}")
    if not targets:
        raise ValueError("no cached target functions selected")

    output = args.output or (
        result_root
        / f"celeba_{args.namespace}_lambda_sweep_lds_q{min(query_ids)}_{max(query_ids)}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, object]] = []
    membership_cache: dict[bytes, np.ndarray] = {}
    lds_model_root = result_root / "lds_model"
    score_root = (
        result_root / "attribution_score" / "prompted_solo"
        / f"train_seed_{args.train_seed}"
    )

    for query_number, qid in enumerate(query_ids, start=1):
        record = query_records[qid]
        prompt_tag = _prompt_path_tag(str(record["prompt"]))
        seed = int(record["initial_seed"])
        base = (
            score_root / f"query_{prompt_tag}" / f"initial_seed_{seed}"
            / args.namespace / "score"
        )
        first_target_rows = grouped[(qid, targets[0])]
        for damping in lambdas:
            directory = base / f"lambda_{lambda_tag(damping)}"
            scores_path = directory / "scores.npy"
            indices_path = directory / "score_indices.npy"
            if not scores_path.is_file() or not indices_path.is_file():
                raise FileNotFoundError(directory)
            scores = np.asarray(np.load(scores_path), dtype=np.float64).reshape(-1)
            indices = np.asarray(np.load(indices_path), dtype=np.int64).reshape(-1)
            if scores.shape != indices.shape or len(scores) != args.expected_points:
                raise ValueError(
                    f"expected {args.expected_points} aligned DAS scores in {directory}"
                )
            key = indices.tobytes()
            if key not in membership_cache:
                membership_cache[key] = rank_membership(
                    indices, first_target_rows, lds_model_root
                )
            prediction = args.prediction_sign * (membership_cache[key] @ scores)
            for target in targets:
                rows = grouped[(qid, target)]
                true_f = np.asarray([float(row["true_f"]) for row in rows], dtype=np.float64)
                lds = float(spearman_corr(prediction, true_f))
                output_rows.append(
                    {
                        "query_id": qid,
                        "initial_seed": seed,
                        "lambda": damping,
                        "target_function": target,
                        "num_subsets": len(rows),
                        "lds_spearman": lds,
                        "lds_percent": 100.0 * lds,
                        "score_dir": str(directory),
                    }
                )
        print(
            f"[LDS] query {query_number}/{len(query_ids)} Q{qid} "
            f"lambdas={len(lambdas)} targets={len(targets)}",
            flush=True,
        )

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    report = output.with_suffix(".txt")
    lines = [
        "CELEBA DAS 50x1 (50 DDIM TIMESTEPS x MC1) LAMBDA SWEEP",
        "=" * 72,
        f"queries={len(query_ids)} range={min(query_ids)}..{max(query_ids)}",
        f"attribution points={args.expected_points}; LDS subset models=192",
        f"prediction sign={args.prediction_sign:g}",
        "",
    ]
    for target in targets:
        lines.append(target.upper())
        lines.append("lambda       mean LDS     median       std       queries")
        summaries = []
        for damping in lambdas:
            values = np.asarray(
                [
                    float(row["lds_percent"])
                    for row in output_rows
                    if row["target_function"] == target and row["lambda"] == damping
                ],
                dtype=np.float64,
            )
            summaries.append((float(np.mean(values)), damping))
            lines.append(
                f"{damping:<11g} {np.mean(values):+10.3f}% "
                f"{np.median(values):+10.3f}% {np.std(values):9.3f}% {len(values):7d}"
            )
        best_mean, best_lambda = max(summaries)
        lines.append(f"BEST MEAN: lambda={best_lambda:g} LDS={best_mean:+.3f}%")
        lines.append("")
    report.write_text("\n".join(lines) + "\n")
    print(f"SAVED CSV: {output}")
    print(f"SAVED TXT: {report}")
    print(f"RECORDS: {len(output_rows)}")


if __name__ == "__main__":
    main()

