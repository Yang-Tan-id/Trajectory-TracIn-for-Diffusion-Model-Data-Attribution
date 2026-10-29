#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


CELEBA_ROOT = Path(__file__).resolve().parents[1]
if str(CELEBA_ROOT) not in sys.path:
    sys.path.insert(0, str(CELEBA_ROOT))

from dataset_config_attr40 import _prompt_path_tag


TARGETS = (
    "endpoint_contarfactual",
    "traj_contarfactual",
    "simple_loss",
    "noise_trajectory",
    "predicted_noise_kl",
    "simple_loss_50x1",
    "vlb_nll",
    "ddim_aux_path_log_ratio",
)
VARIANTS = (
    ("score", "raw"),
    ("score_query_normalized", "query_l2"),
    ("score_train_l2_normalized", "train_l2"),
    ("score_query_train_l2_normalized", "query_train_l2"),
)
MODEL_GROUP = (
    "m_64_k_2500_subset_seed_0__"
    "m_64_k_2500_subset_seed_1__"
    "m_64_k_2500_subset_seed_2"
)


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fields = list(rows[0]) if rows else ["query_id"]
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(prediction: np.ndarray, target: np.ndarray) -> float:
    if len(prediction) != len(target) or len(prediction) < 2:
        return float("nan")
    pred_rank = rank_average(np.asarray(prediction, dtype=np.float64))
    true_rank = rank_average(np.asarray(target, dtype=np.float64))
    pred_rank -= pred_rank.mean()
    true_rank -= true_rank.mean()
    denominator = float(np.linalg.norm(pred_rank) * np.linalg.norm(true_rank))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(pred_rank, true_rank) / denominator)


def namespaces() -> list[str]:
    values = []
    for transform in ("raw_gradient", "adamw_full"):
        values.append(
            f"celeba_endpoint_tracin_{transform}_"
            "train_mean50_query_mean50_linear_shape3d_semantics"
        )
        for reduction in ("linear", "termwise_squared", "timestamp_sum_squared"):
            values.append(
                f"celeba_endpoint_pollute_{transform}_train1x50_query1x50_"
                f"{reduction}_shape3d_semantics"
            )
    return values


def query_id(record: dict[str, object], fallback: int) -> int:
    return int(record.get("query_id", record.get("id", fallback)))


def query_seed(record: dict[str, object]) -> int:
    return int(record.get("initial_seed", record.get("seed")))


def relocate_subset_dir(path_text: str, lds_model_root: Path) -> Path:
    path = Path(path_text)
    if path.is_dir():
        return path
    parts = path.parts
    if "lds_model" in parts:
        relocated = lds_model_root.joinpath(*parts[parts.index("lds_model") + 1 :])
        if relocated.is_dir():
            return relocated
    raise FileNotFoundError(f"cannot locate LDS subset directory: {path}")


def load_targets(path: Path) -> tuple[
    dict[tuple[int, str], list[dict[str, str]]], dict[int, str]
]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = {}
    subset_dirs: dict[int, str] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["query_id"]), row["target_function"])
            grouped.setdefault(key, []).append(row)
            global_id = int(row["global_subset_id"])
            previous = subset_dirs.setdefault(global_id, row["subset_dir"])
            if previous != row["subset_dir"]:
                raise ValueError(
                    f"subset_dir disagreement for global_subset_id={global_id}: "
                    f"{previous} != {row['subset_dir']}"
                )
    for key, rows in grouped.items():
        rows.sort(key=lambda row: int(row["global_subset_id"]))
        ids = [int(row["global_subset_id"]) for row in rows]
        if ids != list(range(192)):
            raise ValueError(f"{key} has invalid subset ids: {len(ids)}/192")
    return grouped, subset_dirs


def build_membership(
    score_indices: np.ndarray,
    subset_dirs: dict[int, str],
    lds_model_root: Path,
) -> np.ndarray:
    if sorted(subset_dirs) != list(range(192)):
        raise ValueError(f"true-f CSV contains {len(subset_dirs)}/192 subset directories")
    score_indices = np.asarray(score_indices, dtype=np.int64).reshape(-1)
    if len(score_indices) != len(np.unique(score_indices)):
        raise ValueError("score_indices.npy contains duplicates")
    order = np.argsort(score_indices)
    sorted_indices = score_indices[order]
    membership = np.zeros((192, len(score_indices)), dtype=np.float64)
    for global_id in range(192):
        subset_dir = relocate_subset_dir(subset_dirs[global_id], lds_model_root)
        kept = np.asarray(
            np.load(subset_dir / "kept_attribution_indices.npy"), dtype=np.int64
        ).reshape(-1)
        positions = np.searchsorted(sorted_indices, kept)
        valid = positions < len(sorted_indices)
        positions = positions[valid]
        kept_valid = kept[valid]
        matched = sorted_indices[positions] == kept_valid
        membership[global_id, order[positions[matched]]] = 1.0
    return membership


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute CelebA endpoint LDS from cached true-f values and saved scores; "
            "no model, gradient, sampler, or GPU work is performed."
        )
    )
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", default=",".join(str(i) for i in range(100)))
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument(
        "--true-f-csv",
        type=Path,
        default=None,
        help="Override the default query100 cached true-f CSV.",
    )
    parser.add_argument(
        "--namespace-filter",
        default="",
        help="Only evaluate namespaces containing this text.",
    )
    parser.add_argument(
        "--namespaces",
        default="",
        help="Comma-separated explicit score namespaces; overrides the built-in list.",
    )
    parser.add_argument(
        "--score-transform",
        choices=("identity", "square"),
        default="identity",
        help="Optional elementwise transform applied to each datapoint score.",
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="Suffix for aggregate CSV names so separate experiments do not overwrite them.",
    )
    parser.add_argument("--prediction-sign", type=float, default=-1.0)
    parser.add_argument(
        "--target-functions",
        default="endpoint_contarfactual,traj_contarfactual,simple_loss,noise_trajectory",
        help="Comma/space-separated cached LDS targets to evaluate.",
    )
    parser.add_argument(
        "--write-result-csv",
        action="store_true",
        help="Also write the 192-row lds_results.csv beside every summary (many files).",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    targets = tuple(
        value
        for value in args.target_functions.replace(",", " ").split()
        if value
    )
    if not targets:
        parser.error("--target-functions is empty")
    invalid_targets = sorted(set(targets) - set(TARGETS))
    if invalid_targets:
        parser.error(
            "unsupported --target-functions: " + ", ".join(invalid_targets)
        )

    requested = set(parse_ints(args.query_ids))
    records = json.loads(args.query_file.read_text())["queries"]
    selected = [
        record for fallback, record in enumerate(records)
        if query_id(record, fallback) in requested
    ]
    if len(selected) != len(requested):
        found = {query_id(record, index) for index, record in enumerate(records)}
        raise ValueError(f"query ids absent from manifest: {sorted(requested - found)}")

    result_root = CELEBA_ROOT / "result" / args.experiment
    true_f_csv = args.true_f_csv or (
        result_root / "query100_lds_csv_ddim50"
        / "celeba_attr40_query100_lds_true_f_long.csv"
    )
    true_f_csv = true_f_csv.expanduser().resolve()
    if not true_f_csv.is_file():
        raise FileNotFoundError(true_f_csv)
    targets_by_key, subset_dirs = load_targets(true_f_csv)
    # CSV paths may have been produced on another machine.  Relocation starts
    # immediately after the stable ``lds_model`` path component.
    lds_model_root = result_root / "lds_model"
    attribution_root = (
        result_root / "attribution_score" / "prompted_solo"
        / f"train_seed_{args.train_seed}"
    )
    eval_root = result_root / "eval" / "prompted_solo"
    output_root = result_root / "endpoint_cached_lds"
    sign_tag = "m1" if args.prediction_sign == -1.0 else (
        "p1" if args.prediction_sign == 1.0 else str(args.prediction_sign).replace(".", "p")
    )

    candidate_namespaces = (
        [value for value in args.namespaces.replace(",", " ").split() if value]
        if args.namespaces
        else namespaces()
    )
    selected_namespaces = [
        namespace for namespace in candidate_namespaces
        if args.namespace_filter in namespace
    ]
    if not selected_namespaces:
        raise ValueError(f"no namespace matches filter: {args.namespace_filter!r}")

    membership_cache: dict[str, np.ndarray] = {}
    all_rows: list[dict[str, object]] = []
    missing: list[str] = []
    expected_per_query = len(selected_namespaces) * len(VARIANTS)
    started = time.time()

    for query_number, record in enumerate(selected, start=1):
        qid = query_id(record, query_number - 1)
        seed = query_seed(record)
        prompt_tag = _prompt_path_tag(str(record["prompt"]))
        query_score_root = (
            attribution_root / f"query_{prompt_tag}" / f"initial_seed_{seed}"
        )
        completed = 0
        for namespace in selected_namespaces:
            scheme_root = query_score_root / f"traj_tracin_{namespace}"
            for score_directory, variant in VARIANTS:
                score_dir = scheme_root / score_directory
                score_path = score_dir / "scores.npy"
                index_path = score_dir / "score_indices.npy"
                if not score_path.is_file() or not index_path.is_file():
                    missing.append(f"Q{qid} {namespace}/{variant}")
                    continue
                scores = np.asarray(np.load(score_path), dtype=np.float64).reshape(-1)
                if args.score_transform == "square":
                    scores = np.square(scores)
                score_indices = np.asarray(np.load(index_path), dtype=np.int64).reshape(-1)
                if scores.shape != score_indices.shape:
                    raise ValueError(f"score/index shape mismatch: {score_dir}")
                digest = hashlib.sha256(score_indices.tobytes()).hexdigest()
                if digest not in membership_cache:
                    membership_cache[digest] = build_membership(
                        score_indices, subset_dirs, lds_model_root
                    )
                prediction = args.prediction_sign * (membership_cache[digest] @ scores)
                transform_suffix = (
                    "" if args.score_transform == "identity"
                    else "_final_score_squared"
                )
                algorithm = f"traj_tracin_{namespace}{transform_suffix}_{variant}"
                for target in targets:
                    target_rows = targets_by_key.get((qid, target))
                    if target_rows is None:
                        raise ValueError(f"true-f rows missing for Q{qid} {target}")
                    true_f = np.asarray(
                        [float(row["true_f"]) for row in target_rows], dtype=np.float64
                    )
                    correlation = spearman(prediction, true_f)
                    out_dir = (
                        eval_root / f"query_{prompt_tag}" / f"initial_seed_{seed}"
                        / "lds" / algorithm / target / f"pred_kept_sign_{sign_tag}"
                        / MODEL_GROUP
                    )
                    if args.write_result_csv:
                        result_rows = []
                        for source, predicted in zip(target_rows, prediction, strict=True):
                            result_rows.append(
                                dict(source)
                                | {
                                    "prediction_subset": "kept",
                                    "prediction_sign": args.prediction_sign,
                                    "pred_sum_tau": float(predicted),
                                }
                            )
                        atomic_csv(out_dir / "lds_results.csv", result_rows)
                    summary = {
                        "algorithm": algorithm,
                        "score_source": str(score_dir.resolve()),
                        "target_cache": str(true_f_csv.resolve()),
                        "query_id": qid,
                        "initial_seed": seed,
                        "target_function": target,
                        "score_variant": variant,
                        "score_transform": args.score_transform,
                        "prediction_subset": "kept",
                        "prediction_sign": args.prediction_sign,
                        "num_models": len(target_rows),
                        "lds_spearman": correlation,
                        "lds_percent": 100.0 * correlation,
                    }
                    atomic_json(out_dir / "lds_summary.json", summary)
                    all_rows.append(
                        {
                            "query_id": qid,
                            "initial_seed": seed,
                            "namespace": namespace,
                            "variant": variant,
                            "target_function": target,
                            "prediction_sign": args.prediction_sign,
                            "lds_spearman": correlation,
                            "lds_percent": 100.0 * correlation,
                            "summary_path": str(out_dir / "lds_summary.json"),
                        }
                    )
                completed += 1
        elapsed = time.time() - started
        print(
            f"[query {query_number}/{len(selected)}] Q{qid} "
            f"score variants={completed}/{expected_per_query} elapsed={elapsed:.1f}s",
            flush=True,
        )

    if not all_rows:
        preview = "\n".join(missing[:20])
        raise RuntimeError(f"no completed endpoint score folders found; first missing:\n{preview}")
    all_rows.sort(
        key=lambda row: (
            str(row["namespace"]), str(row["variant"]),
            str(row["target_function"]), int(row["query_id"]),
        )
    )
    output_parts = []
    if args.score_transform != "identity":
        output_parts.append(args.score_transform)
    if args.output_tag:
        output_parts.append(
            "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in args.output_tag
            ).strip("_")
        )
    output_infix = "_" + "_".join(output_parts) if output_parts else ""
    all_results_path = output_root / f"all_results{output_infix}_sign_{sign_tag}.csv"
    mean_results_path = output_root / f"mean_results{output_infix}_sign_{sign_tag}.csv"
    atomic_csv(all_results_path, all_rows)

    mean_rows: list[dict[str, object]] = []
    for namespace in selected_namespaces:
        for _, variant in VARIANTS:
            for target in targets:
                values = [
                    float(row["lds_percent"])
                    for row in all_rows
                    if row["namespace"] == namespace
                    and row["variant"] == variant
                    and row["target_function"] == target
                ]
                if values:
                    mean_rows.append(
                        {
                            "namespace": namespace,
                            "variant": variant,
                            "target_function": target,
                            "queries": len(values),
                            "mean_lds_percent": float(np.mean(values)),
                            "std_lds_percent": float(np.std(values)),
                        }
                    )
    atomic_csv(mean_results_path, mean_rows)

    print("\nNAMESPACE / VARIANT / TARGET                         N       MEAN        STD")
    print("-" * 100)
    for row in mean_rows:
        label = f"{row['namespace']} / {row['variant']} / {row['target_function']}"
        print(
            f"{label:<72} {int(row['queries']):3d} "
            f"{float(row['mean_lds_percent']):+9.3f}% "
            f"{float(row['std_lds_percent']):8.3f}%"
        )
    print(f"\n[done] LDS summaries={len(all_rows)} missing score variants={len(missing)}")
    print(f"[saved] {all_results_path}")
    print(f"[saved] {mean_results_path}")
    if missing:
        print("[missing preview]")
        for item in missing[:20]:
            print(f"  {item}")
    if args.strict and missing:
        raise SystemExit(
            f"strict mode: {len(missing)} score variants are missing; rerun after score merge"
        )


if __name__ == "__main__":
    main()
