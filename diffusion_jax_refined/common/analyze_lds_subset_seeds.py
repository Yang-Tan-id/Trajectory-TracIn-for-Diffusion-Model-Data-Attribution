from __future__ import annotations

"""Compare LDS-good and LDS-bad seeds by their subset composition.

This script is intentionally lightweight: it reads LDS eval CSVs, LDS subset
index files, CIFAR labels, and attribution scores. It does not need GPU.
"""

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(x: object) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _mean(xs: Iterable[float]) -> float:
    arr = np.asarray([x for x in xs if not math.isnan(float(x))], dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def _std(xs: Iterable[float]) -> float:
    arr = np.asarray([x for x in xs if not math.isnan(float(x))], dtype=np.float64)
    return float(arr.std()) if arr.size else float("nan")


def _rank_average_ties(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def _corr(a: list[float], b: list[float], *, spearman: bool) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(a, b) if not (math.isnan(float(x)) or math.isnan(float(y)))]
    if len(pairs) < 3:
        return float("nan")
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if spearman:
        x = _rank_average_ties(x.tolist())
        y = _rank_average_ties(y.tolist())
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _decode_if_bytes(x):
    return x.decode("utf-8") if isinstance(x, bytes) else x


def _load_cifar_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        d = pickle.load(f, encoding="bytes")
    return np.asarray(d[b"data"], dtype=np.uint8), np.asarray(d[b"labels"], dtype=np.int64)


def _load_cifar_label_names(data_root: Path) -> list[str]:
    with (data_root / "batches.meta").open("rb") as f:
        meta = pickle.load(f, encoding="bytes")
    return [_decode_if_bytes(x) for x in meta[b"label_names"]]


def _filtered_labels(data_root: Path, class_names: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    label_names = _load_cifar_label_names(data_root)
    name_to_id = {name: i for i, name in enumerate(label_names)}
    keep_ids = {name_to_id[name] for name in class_names}
    labels: list[int] = []
    for batch_id in range(1, 6):
        _, batch_labels = _load_cifar_batch(data_root / f"data_batch_{batch_id}")
        for label in batch_labels.tolist():
            if int(label) in keep_ids:
                labels.append(int(label))
    return np.asarray(labels, dtype=np.int64), label_names


def _path_tag(query: str) -> str:
    return query.replace(",", "_").replace(" ", "_").replace("+", "_").strip("_")


def _load_score_indices(score_dir: Path) -> np.ndarray:
    npy = score_dir / "score_indices.npy"
    js = score_dir / "score_indices.json"
    if npy.is_file():
        return np.asarray(np.load(npy), dtype=np.int64).reshape(-1)
    if js.is_file():
        payload = json.loads(js.read_text())
        key = "score_indices" if "score_indices" in payload else "picked_indices"
        return np.asarray(payload[key], dtype=np.int64).reshape(-1)
    raise FileNotFoundError(f"Missing score_indices.npy/json in {score_dir}")


def _load_scores_for_algorithm(base: Path, query: str, initial_seed: int, algorithm: str) -> tuple[np.ndarray, np.ndarray]:
    model_mode = os.environ.get("ATTRIBUTION_SCORE_MODEL_MODE", os.environ.get("SAMPLE_MODEL_MODE", "prompted_solo"))
    train_seed = int(os.environ.get("TRAIN_SEED", "42"))
    root = (
        base
        / "attribution_score"
        / model_mode
        / f"train_seed_{train_seed}"
        / f"query_{_path_tag(query)}"
        / f"initial_seed_{initial_seed}"
    )
    if algorithm.startswith("traj_tracin"):
        names = [
            f"{algorithm}_range_1_2000",
            f"{algorithm}_range_2001_4000",
            f"{algorithm}_range_4001_6000",
            f"{algorithm}_range_6001_8000",
            f"{algorithm}_range_8001_10000",
        ]
    else:
        names = [f"{algorithm}_range_1_10000"]
    idx_parts = []
    score_parts = []
    for name in names:
        d = root / name
        idx = _load_score_indices(d)
        scores = np.asarray(np.load(d / "scores.npy"), dtype=np.float64).reshape(-1)
        if len(idx) != len(scores):
            raise ValueError(f"Length mismatch in {d}: {len(idx)} indices vs {len(scores)} scores")
        idx_parts.append(idx)
        score_parts.append(scores)
    indices = np.concatenate(idx_parts)
    scores = np.concatenate(score_parts)
    order = np.argsort(indices, kind="mergesort")
    return indices[order], scores[order]


def _load_lds_by_seed(eval_base: Path, query: str, algorithm: str, target_function: str, prediction_dir: str, model_glob: str) -> dict[int, float]:
    group = eval_base / f"query_{_path_tag(query)}" / "initial_seed_42" / "lds" / algorithm / target_function / prediction_dir
    aggregates = sorted(group.glob("aggregate_*/per_seed_summary.csv"))
    rows: list[dict[str, str]] = []
    if aggregates:
        rows = _read_csv(aggregates[-1])
    else:
        for seed_dir in sorted(group.glob(model_glob)):
            csv_path = seed_dir / "lds_results.csv"
            result_rows = _read_csv(csv_path)
            if not result_rows:
                continue
            pred = [_safe_float(r.get("pred_sum_tau")) for r in result_rows]
            true = [_safe_float(r.get("true_f")) for r in result_rows]
            seed = int(seed_dir.name.rsplit("_seed_", 1)[1])
            rows.append({"seed": str(seed), "lds_percent": str(100.0 * _corr(pred, true, spearman=True))})
    return {int(r["seed"]): _safe_float(r.get("lds_percent")) for r in rows if r.get("seed")}


def _subset_overlap_stats(subsets: list[np.ndarray]) -> tuple[float, float]:
    if len(subsets) < 2:
        return float("nan"), float("nan")
    sets = [set(s.tolist()) for s in subsets]
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            vals.append(inter / union if union else float("nan"))
    return _mean(vals), _std(vals)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze LDS seed performance against subset label/score composition.")
    parser.add_argument("--experiment-root", type=Path, default=Path("diffusion_jax_refined/cifar2/result/experiment1_42"))
    parser.add_argument("--data-root", type=Path, default=Path("diffusion_jax_refined/dataset/cifar2/cifar-10-batches-py"))
    parser.add_argument("--initial-seed", type=int, default=42)
    parser.add_argument("--lds-m", type=int, default=50)
    parser.add_argument("--lds-k", type=int, default=8000)
    parser.add_argument("--lds-seeds", nargs="+", type=int, default=list(range(1, 17)))
    parser.add_argument("--queries", nargs="+", default=["horse", "automobile", "horse,automobile"])
    parser.add_argument("--algorithms", nargs="+", default=["das", "traj_tracin"])
    parser.add_argument("--target-function", default="noise_trajectory")
    parser.add_argument("--prediction-dir", default="pred_kept_sign_m1")
    parser.add_argument("--class-names", nargs="+", default=["horse", "automobile"])
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    exp = args.experiment_root
    out_dir = args.out_dir or (exp / "analysis" / f"lds_seed_subset_m_{args.lds_m}_k_{args.lds_k}")
    out_dir.mkdir(parents=True, exist_ok=True)

    filtered_labels, label_names = _filtered_labels(args.data_root, tuple(args.class_names))
    class_to_id = {name: label_names.index(name) for name in args.class_names}
    label0, label1 = args.class_names[0], args.class_names[1]
    id0, id1 = class_to_id[label0], class_to_id[label1]

    model_glob = f"m_{args.lds_m}_k_{args.lds_k}_seed_*"
    subset_rows: list[dict] = []
    seed_base_rows: list[dict] = []
    subsets_by_seed: dict[int, list[np.ndarray]] = {}

    for seed in args.lds_seeds:
        model_dir = exp / "lds_model" / f"m_{args.lds_m}_k_{args.lds_k}_seed_{seed}" / "models"
        subsets: list[np.ndarray] = []
        per_subset_label0_frac = []
        per_subset_index_mean = []
        per_subset_bin_l1 = []
        if not model_dir.is_dir():
            print(f"[warn] missing {model_dir}")
            continue
        for subset_dir in sorted(model_dir.glob("subset_*")):
            kept_path = subset_dir / "kept_attribution_indices.npy"
            if not kept_path.is_file():
                continue
            kept = np.asarray(np.load(kept_path), dtype=np.int64).reshape(-1)
            subsets.append(kept)
            labels = filtered_labels[kept]
            label0_count = int(np.sum(labels == id0))
            label1_count = int(np.sum(labels == id1))
            frac0 = label0_count / len(kept)
            bins = np.bincount(np.minimum((kept * 10) // len(filtered_labels), 9), minlength=10) / len(kept)
            bin_l1 = float(np.abs(bins - 0.1).sum())
            per_subset_label0_frac.append(frac0)
            per_subset_index_mean.append(float(kept.mean()))
            per_subset_bin_l1.append(bin_l1)
            subset_rows.append(
                {
                    "lds_seed": seed,
                    "subset": subset_dir.name,
                    "k": len(kept),
                    f"{label0}_count": label0_count,
                    f"{label1}_count": label1_count,
                    f"{label0}_frac": frac0,
                    "index_mean": float(kept.mean()),
                    "index_std": float(kept.std()),
                    "index_bin_l1_from_uniform": bin_l1,
                }
            )
        subsets_by_seed[seed] = subsets
        overlap_mean, overlap_std = _subset_overlap_stats(subsets)
        seed_base_rows.append(
            {
                "lds_seed": seed,
                "num_subsets": len(subsets),
                "k": args.lds_k,
                f"{label0}_frac_mean": _mean(per_subset_label0_frac),
                f"{label0}_frac_std": _std(per_subset_label0_frac),
                f"{label0}_frac_min": min(per_subset_label0_frac) if per_subset_label0_frac else float("nan"),
                f"{label0}_frac_max": max(per_subset_label0_frac) if per_subset_label0_frac else float("nan"),
                "index_mean_mean": _mean(per_subset_index_mean),
                "index_mean_std": _std(per_subset_index_mean),
                "index_bin_l1_mean": _mean(per_subset_bin_l1),
                "index_bin_l1_std": _std(per_subset_bin_l1),
                "pairwise_jaccard_mean": overlap_mean,
                "pairwise_jaccard_std": overlap_std,
            }
        )

    _write_csv(
        out_dir / "subset_label_index_features.csv",
        subset_rows,
        [
            "lds_seed",
            "subset",
            "k",
            f"{label0}_count",
            f"{label1}_count",
            f"{label0}_frac",
            "index_mean",
            "index_std",
            "index_bin_l1_from_uniform",
        ],
    )

    all_seed_rows: list[dict] = []
    corr_rows: list[dict] = []
    for query in args.queries:
        for algorithm in args.algorithms:
            try:
                score_indices, scores = _load_scores_for_algorithm(exp, query, args.initial_seed, algorithm)
            except Exception as e:
                print(f"[warn] skipping score features for query={query} algorithm={algorithm}: {e}")
                score_indices = np.arange(len(filtered_labels), dtype=np.int64)
                scores = np.full(len(filtered_labels), np.nan, dtype=np.float64)
            score_by_idx = np.full(len(filtered_labels), np.nan, dtype=np.float64)
            score_by_idx[score_indices] = scores
            finite_scores = scores[np.isfinite(scores)]
            top10_threshold = float(np.quantile(finite_scores, 0.90)) if finite_scores.size else float("nan")
            top20_threshold = float(np.quantile(finite_scores, 0.80)) if finite_scores.size else float("nan")
            lds_by_seed = _load_lds_by_seed(
                exp / "eval",
                query,
                algorithm,
                args.target_function,
                args.prediction_dir,
                model_glob,
            )
            rows_for_group: list[dict] = []
            for base_row in seed_base_rows:
                seed = int(base_row["lds_seed"])
                subsets = subsets_by_seed.get(seed, [])
                subset_score_means = []
                subset_score_sums = []
                subset_top10_fracs = []
                subset_top20_fracs = []
                for kept in subsets:
                    kept_scores = score_by_idx[kept]
                    subset_score_means.append(float(np.nanmean(kept_scores)))
                    subset_score_sums.append(float(np.nansum(kept_scores)))
                    subset_top10_fracs.append(float(np.nanmean(kept_scores >= top10_threshold)))
                    subset_top20_fracs.append(float(np.nanmean(kept_scores >= top20_threshold)))
                row = {
                    "query": _path_tag(query),
                    "algorithm": algorithm,
                    **base_row,
                    "lds_percent": lds_by_seed.get(seed, float("nan")),
                    "kept_score_mean_mean": _mean(subset_score_means),
                    "kept_score_mean_std": _std(subset_score_means),
                    "kept_score_sum_mean": _mean(subset_score_sums),
                    "kept_score_sum_std": _std(subset_score_sums),
                    "kept_top10_score_frac_mean": _mean(subset_top10_fracs),
                    "kept_top10_score_frac_std": _std(subset_top10_fracs),
                    "kept_top20_score_frac_mean": _mean(subset_top20_fracs),
                    "kept_top20_score_frac_std": _std(subset_top20_fracs),
                }
                rows_for_group.append(row)
                all_seed_rows.append(row)
            feature_names = [
                k
                for k in rows_for_group[0].keys()
                if k not in ("query", "algorithm", "lds_seed", "lds_percent")
                and isinstance(rows_for_group[0].get(k), (int, float))
            ] if rows_for_group else []
            for feature in feature_names:
                xs = [_safe_float(r.get(feature)) for r in rows_for_group]
                ys = [_safe_float(r.get("lds_percent")) for r in rows_for_group]
                corr_rows.append(
                    {
                        "query": _path_tag(query),
                        "algorithm": algorithm,
                        "feature": feature,
                        "pearson_with_lds_percent": _corr(xs, ys, spearman=False),
                        "spearman_with_lds_percent": _corr(xs, ys, spearman=True),
                    }
                )

    seed_fieldnames = [
        "query",
        "algorithm",
        "lds_seed",
        "lds_percent",
        "num_subsets",
        "k",
        f"{label0}_frac_mean",
        f"{label0}_frac_std",
        f"{label0}_frac_min",
        f"{label0}_frac_max",
        "index_mean_mean",
        "index_mean_std",
        "index_bin_l1_mean",
        "index_bin_l1_std",
        "pairwise_jaccard_mean",
        "pairwise_jaccard_std",
        "kept_score_mean_mean",
        "kept_score_mean_std",
        "kept_score_sum_mean",
        "kept_score_sum_std",
        "kept_top10_score_frac_mean",
        "kept_top10_score_frac_std",
        "kept_top20_score_frac_mean",
        "kept_top20_score_frac_std",
    ]
    _write_csv(out_dir / "seed_subset_score_features_by_query_algo.csv", all_seed_rows, seed_fieldnames)
    _write_csv(
        out_dir / "feature_correlations_with_lds.csv",
        corr_rows,
        ["query", "algorithm", "feature", "pearson_with_lds_percent", "spearman_with_lds_percent"],
    )

    lines = []
    lines.append(f"Output directory: {out_dir}")
    lines.append(f"Subset rows: {len(subset_rows)}")
    lines.append(f"Seed/query/algorithm rows: {len(all_seed_rows)}")
    lines.append("")
    for query in args.queries:
        for algorithm in args.algorithms:
            group_rows = [r for r in all_seed_rows if r["query"] == _path_tag(query) and r["algorithm"] == algorithm]
            group_rows = [r for r in group_rows if not math.isnan(_safe_float(r["lds_percent"]))]
            if not group_rows:
                continue
            group_rows.sort(key=lambda r: _safe_float(r["lds_percent"]), reverse=True)
            lines.append(f"{_path_tag(query)} {algorithm}")
            lines.append("  best seeds: " + ", ".join(f"{r['lds_seed']} ({_safe_float(r['lds_percent']):.2f}%)" for r in group_rows[:3]))
            lines.append("  worst seeds: " + ", ".join(f"{r['lds_seed']} ({_safe_float(r['lds_percent']):.2f}%)" for r in group_rows[-3:]))
            corrs = [r for r in corr_rows if r["query"] == _path_tag(query) and r["algorithm"] == algorithm]
            corrs.sort(key=lambda r: abs(_safe_float(r["spearman_with_lds_percent"])), reverse=True)
            lines.append("  top feature correlations:")
            for r in corrs[:6]:
                lines.append(
                    f"    {r['feature']}: Spearman={_safe_float(r['spearman_with_lds_percent']):+.3f}, "
                    f"Pearson={_safe_float(r['pearson_with_lds_percent']):+.3f}"
                )
            lines.append("")
    (out_dir / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
