#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
for path in (SHAPES_ROOT, REFINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_config import DAS_DAMPING_SWEEP_VALUES, _prompt_tag


TARGETS = (
    "endpoint_contarfactual",
    "traj_contarfactual",
    "simple_loss",
    "noise_trajectory",
)


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.replace(",", " ").split() if value.strip()]


def lambda_tag(value: float) -> str:
    return f"{float(value):g}".replace("+", "").replace("-", "neg_").replace(".", "p")


def cache_group(eval_root: Path) -> Path:
    parent = eval_root / "lds_target_cache" / "ddim_eta0"
    groups = sorted(path for path in parent.iterdir() if path.is_dir()) if parent.is_dir() else []
    if len(groups) != 1:
        raise RuntimeError(f"Expected exactly one LDS target-cache model group under {parent}; found {groups}")
    return groups[0]


def materialize_target_csv(target_dir: Path, *, expected_models: int = 192) -> Path:
    rows = []
    for global_id in range(expected_models):
        cache_path = target_dir / f"target_{global_id:04d}.json"
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing LDS true-f cache: {cache_path}")
        payload = json.loads(cache_path.read_text())
        subset_dir = Path(payload["subset_dir"])
        match = re.search(r"_subset_seed_(\d+)", str(subset_dir))
        if match is None:
            raise ValueError(f"Cannot infer LDS subset seed from {subset_dir}")
        rows.append(
            {
                "subset_id": global_id,
                "subset_seed": int(match.group(1)),
                "subset_size": len(np.load(subset_dir / "kept_attribution_indices.npy")),
                "true_f": float(payload["true_f"]),
                "checkpoint": payload.get("checkpoint", ""),
                "subset_dir": str(subset_dir),
                "source_dir": str(target_dir),
            }
        )
    csv_path = target_dir / "true_f_results.csv"
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(csv_path)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute all 3D Shapes DAS LDS scores from cached true-f values.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument(
        "--artifact-namespace",
        default="",
        help="Optional DAS score namespace, for example aligned10x10.",
    )
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--lambdas",
        default=",".join(f"{float(value):g}" for value in DAS_DAMPING_SWEEP_VALUES),
    )
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument(
        "--prediction-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=-1.0,
    )
    args = parser.parse_args()

    namespace = args.artifact_namespace.strip().strip("_/")
    das_name = "das" if not namespace else f"das_{namespace}"
    result_algorithm_prefix = "das" if not namespace else das_name

    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    query_ids = parse_ints(args.query_ids)
    lambdas = parse_floats(args.lambdas)
    expected_results = len(query_ids) * len(lambdas) * len(TARGETS)
    result_root = SHAPES_ROOT / "result" / args.experiment
    legacy_root = REFINE_ROOT / "legacy_jax"
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))
    from LDS.DM_cifar_lds import (
        build_score_vector,
        combine_attribution_scores,
        plot_scatter,
        resolve_score_inputs,
        spearman_corr,
        sum_scores,
        write_csv,
    )

    completed = 0
    for query_id in query_ids:
        if query_id < 0 or query_id >= len(records):
            raise ValueError(f"query id {query_id} is outside [0, {len(records) - 1}]")
        record = records[query_id]
        prompt = str(record["prompt"])
        seed = int(record["initial_seed"])
        prompt_tag = _prompt_tag(prompt)
        eval_root = result_root / "eval" / "prompted_solo" / f"query_{prompt_tag}" / f"initial_seed_{seed}"
        group = cache_group(eval_root)
        score_root = (
            result_root
            / "attribution_score"
            / "prompted_solo"
            / f"train_seed_{args.train_seed}"
            / f"query_{prompt_tag}"
            / f"initial_seed_{seed}"
            / das_name
            / "score"
        )

        target_rows: dict[str, list[dict[str, str]]] = {}
        for target in TARGETS:
            target_csv = materialize_target_csv(group / target) if args.execute else group / target / "true_f_results.csv"
            if args.execute:
                with target_csv.open(newline="") as handle:
                    target_rows[target] = list(csv.DictReader(handle))

        for damping in lambdas:
            tag = lambda_tag(damping)
            score_dir = score_root / f"lambda_{tag}"
            if args.execute:
                if not score_dir.is_dir():
                    raise FileNotFoundError(f"Missing DAS score directory: {score_dir}")
                score_inputs = resolve_score_inputs(str(score_dir))
                indices, scores, sources = combine_attribution_scores(score_inputs, duplicate_policy="max")
                score_map = build_score_vector(indices, scores)
                first_rows = target_rows[TARGETS[0]]
                kept_arrays = [
                    np.load(Path(row["subset_dir"]) / "kept_attribution_indices.npy")
                    for row in first_rows
                ]
                predictions = np.asarray(
                    [sum_scores(kept, score_map, args.prediction_sign) for kept in kept_arrays],
                    dtype=np.float64,
                )

            for target in TARGETS:
                out_dir = (
                    eval_root
                    / "lds"
                    / f"{result_algorithm_prefix}_lambda_{tag}"
                    / target
                    / f"pred_kept_sign_{'p1' if args.prediction_sign > 0 else 'm1'}"
                    / group.name
                )
                if args.execute:
                    started = time.time()
                    rows = []
                    for source_row, prediction in zip(target_rows[target], predictions):
                        row = dict(source_row)
                        row.pop("source_dir", None)
                        row["prediction_subset"] = "kept"
                        row["prediction_sign"] = args.prediction_sign
                        row["pred_sum_tau"] = float(prediction)
                        rows.append(row)
                    true = np.asarray([float(row["true_f"]) for row in rows], dtype=np.float64)
                    lds = spearman_corr(predictions, true)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    write_csv(str(out_dir / "lds_results.csv"), rows)
                    summary = {
                        "algorithm": f"{result_algorithm_prefix}_lambda_{tag}",
                        "damping": damping,
                        "mode": "prompted",
                        "score_sources": sources,
                        "target_cache": str(group / target / "true_f_results.csv"),
                        "num_models": len(rows),
                        "lds_spearman": lds,
                        "lds_percent": 100.0 * lds if not math.isnan(lds) else float("nan"),
                        "target_function": target,
                        "trajectory_reduction": "snapshot_mean",
                        "prediction_subset": "kept",
                        "prediction_sign": args.prediction_sign,
                        "elapsed_sec": time.time() - started,
                    }
                    (out_dir / "lds_summary.json").write_text(json.dumps(summary, indent=2))
                    plot_scatter(
                        str(out_dir / "lds_scatter.png"),
                        predictions,
                        true,
                        f"LDS={lds:.4f} ({100.0 * lds:.2f}%)",
                    )
                    result = f"LDS={100.0 * lds:8.3f}%"
                else:
                    result = "dry-run"
                completed += 1
                print(
                    f"[{completed:03d}/{expected_results}] query={query_id} seed={seed} "
                    f"lambda={damping:g} target={target.replace('contarfactual', 'counterfactual')} {result}",
                    flush=True,
                )

    print(f"Completed {completed} cached DAS LDS evaluations.", flush=True)


if __name__ == "__main__":
    main()
