from __future__ import annotations

"""Fast LDS scoring against cached target values.

This reuses an existing lds_results.csv as the target cache. The expensive
target column true_f is independent of the attribution algorithm/variant, so
new score folders only need to recompute pred_sum_tau and Spearman.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from common.config_loader import load_config, require_attr


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _prediction_indices(subset_dir: Path, subset: str) -> np.ndarray:
    filename = (
        "kept_attribution_indices.npy"
        if subset == "kept"
        else "excluded_attribution_indices.npy"
    )
    return np.load(subset_dir / filename)


def _infer_target_function(rows: list[dict[str, str]], default: str) -> str:
    if not rows:
        return default
    source_dir = rows[0].get("source_dir") or ""
    parts = Path(source_dir).parts
    for target in ("noise_trajectory", "projected_trajectory", "simple_loss"):
        if target in parts:
            return target
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast LDS eval from cached lds_results.csv targets.")
    parser.add_argument("config", help="Dataset dataset_config.py")
    parser.add_argument("--target-results", required=True, help="Existing lds_results.csv with true_f and subset_dir.")
    parser.add_argument("--score-file", required=True, help="Attribution result folder(s), comma-separated.")
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prediction-subset", choices=["kept", "removed"], default="kept")
    parser.add_argument("--prediction-sign", type=float, default=-1.0)
    parser.add_argument("--duplicate-policy", choices=["max", "sum", "mean"], default="max")
    parser.add_argument("--target-function", default=None)
    parser.add_argument("--trajectory-reduction", default="snapshot_mean")
    parser.add_argument("--mode", choices=["prompted", "unprompted"], default="prompted")
    args = parser.parse_args()

    dataset_cfg = load_config(args.config)
    legacy_root = Path(require_attr(dataset_cfg, "LEGACY_JAX_ROOT"))
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

    target_path = Path(args.target_results).expanduser().resolve()
    target_rows = _read_rows(target_path)
    if not target_rows:
        raise ValueError(f"No rows in target results: {target_path}")

    score_inputs = resolve_score_inputs(args.score_file)
    indices, scores, sources = combine_attribution_scores(
        score_inputs,
        duplicate_policy=args.duplicate_policy,
    )
    score_map = build_score_vector(indices, scores)

    rows = []
    started = time.time()
    for source_row in target_rows:
        subset_dir = Path(source_row["subset_dir"])
        prediction_indices = _prediction_indices(subset_dir, args.prediction_subset)
        row = dict(source_row)
        row["prediction_subset"] = args.prediction_subset
        row["prediction_sign"] = args.prediction_sign
        row["pred_sum_tau"] = sum_scores(prediction_indices, score_map, args.prediction_sign)
        rows.append(row)

    pred = np.asarray([float(row["pred_sum_tau"]) for row in rows], dtype=np.float64)
    true = np.asarray([float(row["true_f"]) for row in rows], dtype=np.float64)
    lds = spearman_corr(pred, true)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(str(out_dir / "lds_results.csv"), rows)

    target_function = args.target_function or _infer_target_function(target_rows, "cached_target")
    summary = {
        "algorithm": args.algorithm,
        "mode": args.mode,
        "score_sources": sources,
        "target_cache": str(target_path),
        "num_models": len(rows),
        "lds_spearman": lds,
        "lds_percent": 100.0 * lds if not math.isnan(lds) else float("nan"),
        "target_function": target_function,
        "trajectory_reduction": args.trajectory_reduction,
        "prediction_subset": args.prediction_subset,
        "prediction_sign": args.prediction_sign,
        "elapsed_sec": time.time() - started,
    }
    (out_dir / "lds_summary.json").write_text(json.dumps(summary, indent=2))
    plot_scatter(str(out_dir / "lds_scatter.png"), pred, true, f"LDS={lds:.4f} ({100.0 * lds:.2f}%)")
    print(f"Saved fast LDS evaluation to {out_dir}")


if __name__ == "__main__":
    main()
