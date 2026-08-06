from __future__ import annotations

"""Fast LDS eval directly from query-cached stream score artifacts.

This is the no-score-folder companion to fast_lds_score_eval.py. It reuses an
existing lds_results.csv target cache and reads one query slice from a merged
stream_scores_merged.npz artifact in memory.
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


SCORE_KEYS = {
    "raw": "scores_raw",
    "query_l2_normalized": "scores_query_l2_normalized",
    "train_l2_normalized": "scores_train_l2_normalized",
    "query_train_l2_normalized": "scores_query_train_l2_normalized",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_rewrites(items: list[str]) -> list[tuple[str, str]]:
    rewrites = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--subset-dir-rewrite must look like OLD=NEW, got {item!r}")
        old, new = item.split("=", 1)
        if not old:
            raise ValueError(f"Empty OLD prefix in rewrite {item!r}")
        rewrites.append((old.rstrip("/"), new.rstrip("/")))
    return rewrites


def rewrite_path(path_text: str, rewrites: list[tuple[str, str]]) -> str:
    for old, new in rewrites:
        if path_text == old or path_text.startswith(old + "/"):
            return new + path_text[len(old) :]
    return path_text


def split_csv(text: str | None) -> list[str]:
    if text is None or not text.strip():
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def select_query_index(query_artifacts: np.ndarray, query_filter: str) -> int:
    paths = [str(x) for x in query_artifacts.tolist()]
    matches = [i for i, path in enumerate(paths) if query_filter in path]
    if not matches:
        raise ValueError(f"No query artifact matched --query-filter {query_filter!r}")
    if len(matches) > 1:
        sample = "\n".join(f"[{i}] {paths[i]}" for i in matches[:10])
        raise ValueError(
            f"--query-filter {query_filter!r} matched {len(matches)} query artifacts; "
            f"make it more specific.\n{sample}"
        )
    return matches[0]


def score_map_from_stream(
    payload: np.lib.npyio.NpzFile,
    *,
    query_index: int,
    proj_dim: int,
    variant: str,
) -> dict[int, float]:
    proj_dims = np.asarray(payload["proj_dims"], dtype=np.int64).reshape(-1)
    dim_matches = np.where(proj_dims == int(proj_dim))[0]
    if len(dim_matches) != 1:
        raise ValueError(f"Projection dim {proj_dim} not found in proj_dims={proj_dims.tolist()}")
    if variant not in SCORE_KEYS:
        raise ValueError(f"Unknown variant {variant!r}; choices are {sorted(SCORE_KEYS)}")

    scores = np.asarray(payload[SCORE_KEYS[variant]][int(dim_matches[0]), int(query_index), :], dtype=np.float64)
    indices = np.asarray(payload["score_indices"], dtype=np.int64).reshape(-1)
    if len(scores) != len(indices):
        raise ValueError(f"scores length {len(scores)} does not match score_indices length {len(indices)}")
    return {int(idx): float(score) for idx, score in zip(indices, scores)}


def infer_target_function(rows: list[dict[str, str]], default: str) -> str:
    if not rows:
        return default
    source_dir = rows[0].get("source_dir") or ""
    parts = Path(source_dir).parts
    for target in ("noise_trajectory", "projected_trajectory", "simple_loss"):
        if target in parts:
            return target
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast LDS eval directly from stream_scores_merged.npz.")
    parser.add_argument("config", help="Dataset dataset_config.py")
    parser.add_argument("--stream-score-npz", required=True, help="Merged stream_scores_merged.npz artifact.")
    parser.add_argument("--target-results", required=True, help="Existing lds_results.csv with true_f and subset_dir.")
    parser.add_argument("--query-filter", required=True, help="Unique substring selecting one query artifact path.")
    parser.add_argument("--projection-dims", default=None, help="Comma-separated dims. Defaults to all dims in the artifact.")
    parser.add_argument("--variants", default="query_train_l2_normalized", help="Comma-separated score variants.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prediction-subset", choices=["kept", "removed"], default="kept")
    parser.add_argument("--prediction-sign", type=float, default=-1.0)
    parser.add_argument("--target-function", default=None)
    parser.add_argument("--trajectory-reduction", default="snapshot_mean")
    parser.add_argument("--mode", choices=["prompted", "unprompted"], default="prompted")
    parser.add_argument(
        "--subset-dir-rewrite",
        action="append",
        default=[],
        help="Rewrite subset_dir prefixes in target CSV, e.g. /work2/.../repo=/local/repo. May repeat.",
    )
    args = parser.parse_args()

    dataset_cfg = load_config(args.config)
    legacy_root = Path(require_attr(dataset_cfg, "LEGACY_JAX_ROOT"))
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))

    from LDS.DM_cifar_lds import plot_scatter, spearman_corr, sum_scores

    target_path = Path(args.target_results).expanduser().resolve()
    target_rows = read_rows(target_path)
    if not target_rows:
        raise ValueError(f"No rows in target results: {target_path}")

    rewrites = parse_rewrites(args.subset_dir_rewrite)
    variants = split_csv(args.variants)
    if not variants:
        raise ValueError("--variants is empty")

    stream_path = Path(args.stream_score_npz).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()

    with np.load(stream_path, allow_pickle=True) as payload:
        query_index = select_query_index(payload["query_artifacts"], args.query_filter)
        query_artifact = str(payload["query_artifacts"][query_index])
        artifact_dims = [int(x) for x in np.asarray(payload["proj_dims"], dtype=np.int64).reshape(-1)]
        proj_dims = [int(x) for x in split_csv(args.projection_dims)] if args.projection_dims else artifact_dims

        for proj_dim in proj_dims:
            for variant in variants:
                score_map = score_map_from_stream(
                    payload,
                    query_index=query_index,
                    proj_dim=proj_dim,
                    variant=variant,
                )
                rows: list[dict[str, object]] = []
                for source_row in target_rows:
                    row = dict(source_row)
                    subset_dir = Path(rewrite_path(row["subset_dir"], rewrites))
                    filename = (
                        "kept_attribution_indices.npy"
                        if args.prediction_subset == "kept"
                        else "excluded_attribution_indices.npy"
                    )
                    prediction_indices = np.load(subset_dir / filename)
                    row["subset_dir"] = str(subset_dir)
                    row["prediction_subset"] = args.prediction_subset
                    row["prediction_sign"] = args.prediction_sign
                    row["pred_sum_tau"] = sum_scores(prediction_indices, score_map, args.prediction_sign)
                    rows.append(row)

                pred = np.asarray([float(row["pred_sum_tau"]) for row in rows], dtype=np.float64)
                true = np.asarray([float(row["true_f"]) for row in rows], dtype=np.float64)
                lds = spearman_corr(pred, true)

                out_dir = out_root / f"proj_{proj_dim}" / variant
                out_dir.mkdir(parents=True, exist_ok=True)
                write_csv(out_dir / "lds_results.csv", rows)
                summary = {
                    "mode": args.mode,
                    "query_filter": args.query_filter,
                    "query_index": int(query_index),
                    "query_artifact": query_artifact,
                    "stream_score_npz": str(stream_path),
                    "target_cache": str(target_path),
                    "num_models": len(rows),
                    "num_scores": len(score_map),
                    "proj_dim": int(proj_dim),
                    "score_variant": variant,
                    "lds_spearman": lds,
                    "lds_percent": 100.0 * lds if not math.isnan(lds) else float("nan"),
                    "target_function": args.target_function or infer_target_function(target_rows, "cached_target"),
                    "trajectory_reduction": args.trajectory_reduction,
                    "prediction_subset": args.prediction_subset,
                    "prediction_sign": args.prediction_sign,
                    "subset_dir_rewrite": args.subset_dir_rewrite,
                    "elapsed_sec": time.time() - started,
                }
                (out_dir / "lds_summary.json").write_text(json.dumps(summary, indent=2))
                plot_scatter(str(out_dir / "lds_scatter.png"), pred, true, f"LDS={lds:.4f} ({100.0 * lds:.2f}%)")
                print(f"[saved] {out_dir} | LDS={lds:.6f}")


if __name__ == "__main__":
    main()
