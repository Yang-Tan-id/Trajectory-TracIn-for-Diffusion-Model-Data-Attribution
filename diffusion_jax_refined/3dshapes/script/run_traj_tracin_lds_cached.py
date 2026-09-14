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
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))
if str(REFINE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFINE_ROOT))

from dataset_config import _prompt_tag


TARGETS = (
    "endpoint_contarfactual",
    "traj_contarfactual",
    "simple_loss",
    "noise_trajectory",
)
VARIANTS = (
    ("raw", "score"),
    ("query_l2", "score_query_normalized"),
    ("train_l2", "score_train_l2_normalized"),
    ("query_train_l2", "score_query_train_l2_normalized"),
)


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value.strip()]


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
        subset_size = len(np.load(subset_dir / "kept_attribution_indices.npy"))
        rows.append(
            {
                "subset_id": global_id,
                "subset_seed": int(match.group(1)),
                "subset_size": subset_size,
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
    parser = argparse.ArgumentParser(
        description="Compute all 3D Shapes Traj TracIn LDS scores from cached true-f values."
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    args = parser.parse_args()

    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    query_ids = parse_ints(args.query_ids)
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
        eval_root = (
            result_root
            / "eval"
            / "prompted_solo"
            / f"query_{prompt_tag}"
            / f"initial_seed_{seed}"
        )
        group = cache_group(eval_root)
        score_root = (
            result_root
            / "attribution_score"
            / "prompted_solo"
            / f"train_seed_{args.train_seed}"
            / f"query_{prompt_tag}"
            / f"initial_seed_{seed}"
            / "traj_tracin"
        )

        target_rows = {}
        for target in TARGETS:
            target_csv = (
                materialize_target_csv(group / target)
                if args.execute
                else group / target / "true_f_results.csv"
            )
            if args.execute:
                with target_csv.open(newline="") as handle:
                    target_rows[target] = list(csv.DictReader(handle))

        # Prediction sums depend on query/variant/subset, but not on target.
        # Load each score variant once, then reuse its 192 predictions for all
        # four target functions.
        for variant, score_name in VARIANTS:
            score_dir = score_root / score_name
            if args.execute and not score_dir.is_dir():
                raise FileNotFoundError(f"Missing Traj TracIn score directory: {score_dir}")
            if args.execute:
                score_inputs = resolve_score_inputs(str(score_dir))
                indices, scores, sources = combine_attribution_scores(
                    score_inputs, duplicate_policy="max"
                )
                score_map = build_score_vector(indices, scores)
                first_rows = target_rows[TARGETS[0]]
                kept_arrays = [
                    np.load(Path(row["subset_dir"]) / "kept_attribution_indices.npy")
                    for row in first_rows
                ]
                predictions = np.asarray(
                    [sum_scores(kept, score_map, -1.0) for kept in kept_arrays],
                    dtype=np.float64,
                )

            for target in TARGETS:
                score_dir = score_root / score_name
                out_dir = (
                    eval_root
                    / "lds"
                    / f"traj_tracin_{variant}"
                    / target
                    / "pred_kept_sign_m1"
                    / group.name
                )
                print(
                    f"[{completed + 1}/160] query={query_id} target={target} variant={variant}",
                    flush=True,
                )
                if args.execute:
                    started = time.time()
                    rows = []
                    for source_row, prediction in zip(target_rows[target], predictions):
                        row = dict(source_row)
                        row.pop("source_dir", None)
                        row["prediction_subset"] = "kept"
                        row["prediction_sign"] = -1.0
                        row["pred_sum_tau"] = float(prediction)
                        rows.append(row)
                    true = np.asarray([float(row["true_f"]) for row in rows], dtype=np.float64)
                    lds = spearman_corr(predictions, true)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    write_csv(str(out_dir / "lds_results.csv"), rows)
                    summary = {
                        "algorithm": f"traj_tracin_{variant}",
                        "mode": "prompted",
                        "score_sources": sources,
                        "target_cache": str(group / target / "true_f_results.csv"),
                        "num_models": len(rows),
                        "lds_spearman": lds,
                        "lds_percent": 100.0 * lds if not math.isnan(lds) else float("nan"),
                        "target_function": target,
                        "trajectory_reduction": "snapshot_mean",
                        "prediction_subset": "kept",
                        "prediction_sign": -1.0,
                        "elapsed_sec": time.time() - started,
                    }
                    (out_dir / "lds_summary.json").write_text(json.dumps(summary, indent=2))
                    plot_scatter(
                        str(out_dir / "lds_scatter.png"),
                        predictions,
                        true,
                        f"LDS={lds:.4f} ({100.0 * lds:.2f}%)",
                    )
                    print(f"Saved cached LDS evaluation to {out_dir}", flush=True)
                else:
                    print(f"score={score_dir} target={group / target} out={out_dir}", flush=True)
                completed += 1

    print(f"Completed {completed} cached Traj TracIn LDS evaluations.", flush=True)


if __name__ == "__main__":
    main()
