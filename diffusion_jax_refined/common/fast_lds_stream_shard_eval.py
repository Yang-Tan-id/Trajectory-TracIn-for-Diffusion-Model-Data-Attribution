from __future__ import annotations

"""Fast LDS eval directly from query-cached stream score shards.

This avoids merging large stream score shards. It reads only the aggregate
scores_* arrays from each shard, selects one query slice, and evaluates LDS
against an existing lds_results.csv target cache.
"""

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from common.config_loader import load_config, require_attr
from common.fast_lds_stream_score_eval import (
    SCORE_KEYS,
    infer_target_function,
    parse_rewrites,
    plot_scatter,
    read_rows,
    rewrite_path,
    select_query_index,
    spearman_corr,
    split_csv,
    sum_scores,
    write_csv,
)


def sorted_shard_paths(pattern: str) -> list[Path]:
    paths = [Path(path) for path in glob.glob(pattern)]
    if not paths:
        raise FileNotFoundError(f"No stream shards matched: {pattern}")
    return sorted(paths)


def load_score_maps_from_shards(
    shard_paths: list[Path],
    *,
    query_filter: str,
    proj_dims_requested: list[int] | None,
    variants: list[str],
) -> tuple[dict[tuple[int, str], dict[int, float]], dict[str, object]]:
    score_maps: dict[tuple[int, str], dict[int, float]] = {}
    query_index: int | None = None
    query_artifact: str | None = None
    proj_dims: list[int] | None = None
    total_scores = 0
    started = time.time()

    for shard_no, shard_path in enumerate(shard_paths, 1):
        print(f"[load] shard {shard_no}/{len(shard_paths)} {shard_path}", flush=True)
        with np.load(shard_path, allow_pickle=True) as payload:
            shard_query_artifacts = payload["query_artifacts"]
            if query_index is None:
                query_index = select_query_index(shard_query_artifacts, query_filter)
                query_artifact = str(shard_query_artifacts[query_index])
                proj_dims = [int(x) for x in np.asarray(payload["proj_dims"], dtype=np.int64).reshape(-1)]
                for variant in variants:
                    if variant not in SCORE_KEYS:
                        raise ValueError(f"Unknown variant {variant!r}; choices are {sorted(SCORE_KEYS)}")
                if proj_dims_requested is None:
                    selected_dims = proj_dims
                else:
                    missing = [dim for dim in proj_dims_requested if dim not in proj_dims]
                    if missing:
                        raise ValueError(f"Requested projection dims missing from shard: {missing}; available={proj_dims}")
                    selected_dims = proj_dims_requested
                for dim in selected_dims:
                    for variant in variants:
                        score_maps[(dim, variant)] = {}
            else:
                this_query_index = select_query_index(shard_query_artifacts, query_filter)
                if this_query_index != query_index:
                    raise ValueError(f"Query index mismatch in shard {shard_path}: {this_query_index} != {query_index}")
                this_dims = [int(x) for x in np.asarray(payload["proj_dims"], dtype=np.int64).reshape(-1)]
                if this_dims != proj_dims:
                    raise ValueError(f"Projection dims mismatch in shard {shard_path}: {this_dims} != {proj_dims}")

            assert query_index is not None and proj_dims is not None
            score_indices = np.asarray(payload["score_indices"], dtype=np.int64).reshape(-1)
            total_scores += len(score_indices)
            for (dim, variant), score_map in score_maps.items():
                dim_i = proj_dims.index(dim)
                values = np.asarray(payload[SCORE_KEYS[variant]][dim_i, query_index, :], dtype=np.float64).reshape(-1)
                if len(values) != len(score_indices):
                    raise ValueError(
                        f"{shard_path} {variant} dim={dim} length {len(values)} "
                        f"does not match score_indices length {len(score_indices)}"
                    )
                for idx, value in zip(score_indices, values):
                    idx_int = int(idx)
                    if idx_int in score_map:
                        raise ValueError(f"Duplicate score index {idx_int} while reading {shard_path}")
                    score_map[idx_int] = float(value)
        print(f"[load] shard {shard_no}/{len(shard_paths)} done | total_scores={total_scores}", flush=True)

    assert query_index is not None and query_artifact is not None and proj_dims is not None
    return score_maps, {
        "query_index": int(query_index),
        "query_artifact": query_artifact,
        "proj_dims": proj_dims,
        "num_scores": int(total_scores),
        "elapsed_load_sec": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Dataset dataset_config.py")
    parser.add_argument("--stream-shard-glob", required=True, help="Quoted glob for shard stream_scores.npz files.")
    parser.add_argument("--target-results", required=True, help="Existing lds_results.csv with true_f and subset_dir.")
    parser.add_argument("--query-filter", required=True, help="Unique substring selecting one query artifact path.")
    parser.add_argument("--projection-dims", default=None, help="Comma-separated dims. Defaults to all dims in the shards.")
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

    target_path = Path(args.target_results).expanduser().resolve()
    target_rows = read_rows(target_path)
    if not target_rows:
        raise ValueError(f"No rows in target results: {target_path}")

    variants = split_csv(args.variants)
    if not variants:
        raise ValueError("--variants is empty")
    proj_dims = [int(x) for x in split_csv(args.projection_dims)] if args.projection_dims else None
    rewrites = parse_rewrites(args.subset_dir_rewrite)
    shard_paths = sorted_shard_paths(args.stream_shard_glob)

    score_maps, score_meta = load_score_maps_from_shards(
        shard_paths,
        query_filter=args.query_filter,
        proj_dims_requested=proj_dims,
        variants=variants,
    )

    subset_filename = (
        "kept_attribution_indices.npy" if args.prediction_subset == "kept" else "excluded_attribution_indices.npy"
    )
    subset_indices_by_row = []
    for source_row in target_rows:
        subset_dir = Path(rewrite_path(source_row["subset_dir"], rewrites))
        subset_indices_by_row.append((subset_dir, np.load(subset_dir / subset_filename)))

    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for (proj_dim, variant), score_map in sorted(score_maps.items()):
        rows: list[dict[str, object]] = []
        for source_row, (subset_dir, prediction_indices) in zip(target_rows, subset_indices_by_row):
            row = dict(source_row)
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
            "algorithm": "traj_tracin_projected_stream_sharded",
            "mode": args.mode,
            "query_filter": args.query_filter,
            "query_index": score_meta["query_index"],
            "query_artifact": score_meta["query_artifact"],
            "stream_shard_glob": args.stream_shard_glob,
            "num_shards": len(shard_paths),
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
            "load_meta": score_meta,
        }
        (out_dir / "lds_summary.json").write_text(json.dumps(summary, indent=2))
        plot_scatter(str(out_dir / "lds_scatter.png"), pred, true, f"LDS={lds:.4f} ({100.0 * lds:.2f}%)")
        print(f"[saved] {out_dir} | LDS={lds:.6f}", flush=True)


if __name__ == "__main__":
    main()
