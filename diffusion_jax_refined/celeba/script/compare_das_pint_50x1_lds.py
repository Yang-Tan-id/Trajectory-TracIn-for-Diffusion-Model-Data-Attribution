#!/usr/bin/env python3
"""Compare CelebA DAS lambda-sweep LDS with aligned 50x1 PINT scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


CELEBA_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = CELEBA_ROOT.parent
REPO_ROOT = REFINE_ROOT.parent
for path in (CELEBA_ROOT, REFINE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_config_attr40 import _prompt_path_tag
from diffusion_jax_refined.script.export_pint_lds_txt import method_configs
from diffusion_jax_refined.celeba.script.evaluate_five_method_simple_loss_50x1 import (
    VARIANT_SCORE_DIR,
)
from diffusion_jax_refined.celeba.script.run_endpoint_cached_lds import (
    build_membership,
    load_targets,
    spearman,
)


def parse_ids(text: str) -> list[int]:
    values: list[int] = []
    for token in text.replace(",", " ").split():
        if "-" in token:
            left, right = token.split("-", 1)
            values.extend(range(int(left), int(right) + 1))
        else:
            values.append(int(token))
    if not values or len(values) != len(set(values)):
        raise ValueError("query IDs must be nonempty and unique")
    return values


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument(
        "--query-file",
        type=Path,
        default=CELEBA_ROOT / "queries_attr40_val25_x4.json",
    )
    parser.add_argument("--query-ids", default="0-71")
    parser.add_argument("--das-csv", type=Path, default=None)
    parser.add_argument("--true-f-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    query_ids = parse_ids(args.query_ids)
    result_root = CELEBA_ROOT / "result" / args.experiment
    das_csv = args.das_csv or (
        result_root / "celeba_das_50x1_mc1_lambda_sweep_lds_q0_71.csv"
    )
    true_f_csv = args.true_f_csv or (
        result_root / "query100_lds_csv_ddim50"
        / "celeba_attr40_query100_lds_true_f_long.csv"
    )
    output = args.output or (
        result_root / "celeba_das_vs_pint_50x1_lds_q0_71.csv"
    )

    das_rows = [
        row for row in read_csv(das_csv) if int(row["query_id"]) in query_ids
    ]
    if not das_rows:
        raise ValueError("DAS CSV contains no requested queries")
    targets = sorted({row["target_function"] for row in das_rows})
    lambdas = sorted({float(row["lambda"]) for row in das_rows})
    best_lambda: dict[str, float] = {}
    for target in targets:
        candidates = []
        for damping in lambdas:
            values = [
                float(row["lds_percent"])
                for row in das_rows
                if row["target_function"] == target
                and float(row["lambda"]) == damping
            ]
            if len(values) == len(query_ids):
                candidates.append((float(np.mean(values)), damping))
        if not candidates:
            raise ValueError(f"DAS has no complete lambda for {target}")
        best_lambda[target] = max(candidates)[1]

    records = json.loads(args.query_file.read_text())["queries"]
    records_by_id = {
        int(record.get("query_id", fallback)): record
        for fallback, record in enumerate(records)
    }
    missing = sorted(set(query_ids) - set(records_by_id))
    if missing:
        raise ValueError(f"query manifest is missing IDs: {missing}")

    targets_by_key, subset_dirs = load_targets(true_f_csv)
    score_root = (
        result_root / "attribution_score" / "prompted_solo"
        / f"train_seed_{args.train_seed}"
    )
    lds_model_root = result_root / "lds_model"
    membership_cache: dict[str, np.ndarray] = {}
    output_rows: list[dict[str, object]] = []

    for row in das_rows:
        target = row["target_function"]
        if float(row["lambda"]) != best_lambda[target]:
            continue
        output_rows.append(
            {
                "query_id": int(row["query_id"]),
                "target_function": target,
                "method": "DAS",
                "transform": "raw_gradient",
                "variant": "lambda_sweep_best_mean",
                "lambda": best_lambda[target],
                "lds_percent": float(row["lds_percent"]),
            }
        )

    pint_methods = method_configs("celeba")[:2]
    for qid in query_ids:
        record = records_by_id[qid]
        prompt_tag = _prompt_path_tag(str(record["prompt"]))
        seed = int(record.get("initial_seed", record.get("seed")))
        query_root = score_root / f"query_{prompt_tag}" / f"initial_seed_{seed}"
        for method in pint_methods:
            for transform, namespace, sign_tag in (
                ("raw_gradient", method.raw_namespace, method.raw_sign),
                ("adamw_full", method.adamw_namespace, method.adamw_sign),
            ):
                if namespace is None:
                    continue
                score_dir = query_root / namespace / VARIANT_SCORE_DIR[method.variant]
                scores_path = score_dir / "scores.npy"
                indices_path = score_dir / "score_indices.npy"
                if not scores_path.is_file() or not indices_path.is_file():
                    raise FileNotFoundError(score_dir)
                scores = np.asarray(np.load(scores_path), dtype=np.float64).reshape(-1)
                indices = np.asarray(np.load(indices_path), dtype=np.int64).reshape(-1)
                if scores.shape != indices.shape:
                    raise ValueError(f"score/index shape mismatch: {score_dir}")
                digest = hashlib.sha256(indices.tobytes()).hexdigest()
                if digest not in membership_cache:
                    membership_cache[digest] = build_membership(
                        indices, subset_dirs, lds_model_root
                    )
                sign = 1.0 if sign_tag == "p1" else -1.0
                prediction = sign * (membership_cache[digest] @ scores)
                for target in targets:
                    true_rows = targets_by_key.get((qid, target))
                    if true_rows is None or len(true_rows) != 192:
                        found = 0 if true_rows is None else len(true_rows)
                        raise ValueError(f"Q{qid} {target} has {found}/192 true-f rows")
                    true_f = np.asarray(
                        [float(item["true_f"]) for item in true_rows],
                        dtype=np.float64,
                    )
                    output_rows.append(
                        {
                            "query_id": qid,
                            "target_function": target,
                            "method": "PINT",
                            "transform": transform,
                            "variant": method.variant,
                            "lambda": "",
                            "lds_percent": 100.0 * spearman(prediction, true_f),
                        }
                    )
        print(f"[PINT] Q{qid} complete", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    report = output.with_suffix(".txt")
    lines = [
        "CELEBA DAS VS PINT — ALIGNED 50x1 LDS",
        "=" * 72,
        f"queries={len(query_ids)} range={min(query_ids)}..{max(query_ids)}",
        "LDS subset models=192; identical cached true-f targets",
        "DAS lambda is selected by mean LDS separately for each target.",
        "",
    ]
    for target in targets:
        lines.extend([target.upper(), "method                                  mean      median         std   DAS wins"])
        target_rows = [row for row in output_rows if row["target_function"] == target]
        das_by_query = {
            int(row["query_id"]): float(row["lds_percent"])
            for row in target_rows
            if row["method"] == "DAS"
        }
        groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
        for row in target_rows:
            key = (str(row["method"]), str(row["transform"]), str(row["variant"]))
            groups.setdefault(key, []).append(row)
        ordered = sorted(groups, key=lambda key: (key[0] != "DAS", key))
        for method, transform, variant in ordered:
            rows = groups[(method, transform, variant)]
            values = np.asarray([float(row["lds_percent"]) for row in rows])
            label = f"{method} {transform} / {variant}"
            if method == "DAS":
                label += f" (lambda={best_lambda[target]:g})"
                wins_text = "-"
            else:
                pint_by_query = {
                    int(row["query_id"]): float(row["lds_percent"])
                    for row in rows
                }
                das_wins = sum(
                    das_by_query[qid] > pint_by_query[qid] for qid in query_ids
                )
                wins_text = f"{das_wins}/{len(query_ids)}"
            lines.append(
                f"{label:<39} {np.mean(values):+8.3f}% "
                f"{np.median(values):+10.3f}% {np.std(values):9.3f}% "
                f"{wins_text:>10}"
            )
        lines.append("")
    report.write_text("\n".join(lines) + "\n")
    print(f"SAVED CSV: {output}")
    print(f"SAVED TXT: {report}")


if __name__ == "__main__":
    main()
