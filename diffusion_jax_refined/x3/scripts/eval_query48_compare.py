from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


LAMBDA_DEFAULT = (
    "0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10 20 50 100 200 "
    "500 1000 2000 5000 10000 20000 50000"
)


def path_tag(value: str) -> str:
    text = str(value).strip().replace(",", "__").replace("+", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unprompted"


def damping_tag(value: str) -> str:
    return path_tag(str(value).replace("+", "_").replace("-", "neg_").replace(".", "p"))


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if mask.sum() < 2:
        return float("nan")
    rx = rankdata_average(xa[mask])
    ry = rankdata_average(ya[mask])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_query_plan(path: Path) -> list[dict[str, str]]:
    rows = read_csv_rows_tab(path)
    if not rows:
        raise ValueError(f"empty query plan: {path}")
    return rows


def read_csv_rows_tab(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def stream_root(result_root: Path, method: str, score_mode: str) -> Path:
    if method == "OldNext":
        base = result_root / "projected_traj_tracin_artifacts_next_ckpt"
    elif method == "OldRef":
        base = result_root / "projected_traj_tracin_artifacts_refproj"
    else:
        raise ValueError(method)
    return (
        base
        / "stream_term_scores"
        / f"score_mode_{score_mode}"
        / "cache_4096"
        / "proj_4096"
        / "variants_train_l2_normalized"
    )


def query_filter(row: dict[str, str]) -> str:
    if row["unprompted"] == "1":
        return f"unprompted/initial_seed_{row['initial_seed']}/shared_query/proj_4096/query_gradient_artifact.npz"
    return (
        f"query_{path_tag(row['query'])}/initial_seed_{row['initial_seed']}/"
        "shared_query/proj_4096/query_gradient_artifact.npz"
    )


def target_csv(result_root: Path, row: dict[str, str], target: str, lds_seed: int, das_lambda: str) -> Path:
    score_mode = row["score_mode"]
    init = row["initial_seed"]
    if row["unprompted"] == "1":
        return (
            result_root
            / "eval"
            / score_mode
            / "unprompted"
            / f"initial_seed_{init}"
            / "lds_unprompted"
            / f"das_lambda_{damping_tag(das_lambda)}"
            / target
            / "pred_kept_sign_m1"
            / f"m_64_k_2500_pct_25_subset_seed_{lds_seed}"
            / "lds_results.csv"
        )
    return (
        result_root
        / "eval"
        / score_mode
        / f"query_{path_tag(row['query'])}"
        / f"initial_seed_{init}"
        / "lds"
        / f"das_lambda_{damping_tag(das_lambda)}"
        / target
        / "pred_kept_sign_m1"
        / f"m_64_k_2500_pct_25_subset_seed_{lds_seed}"
        / "lds_results.csv"
    )


def das_score_file(result_root: Path, row: dict[str, str], lam: str, train_seed: int) -> Path:
    score_mode = row["score_mode"]
    init = row["initial_seed"]
    if row["unprompted"] == "1":
        return (
            result_root
            / "attribution_score"
            / score_mode
            / f"train_seed_{train_seed}"
            / "unprompted"
            / f"initial_seed_{init}"
            / "das_unprompted"
            / f"lambda_{damping_tag(lam)}"
            / "scores.npy"
        )
    return (
        result_root
        / "attribution_score"
        / score_mode
        / f"train_seed_{train_seed}"
        / f"query_{path_tag(row['query'])}"
        / f"initial_seed_{init}"
        / "das"
        / f"lambda_{damping_tag(lam)}"
        / "scores.npy"
    )


def load_stream_scores(root: Path) -> dict[str, np.ndarray]:
    paths = sorted((root / "shards").glob("range_*/stream_scores.npz"))
    if not paths:
        raise FileNotFoundError(f"no stream shards under {root}")
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for i, path in enumerate(paths, 1):
        print(f"[load-stream] {root.name} shard {i}/{len(paths)} {path}", flush=True)
        with np.load(path, allow_pickle=True) as z:
            qarts = [str(x) for x in z["query_artifacts"]]
            dims = [int(x) for x in np.asarray(z["proj_dims"]).reshape(-1)]
            dim_i = dims.index(4096)
            values = np.asarray(z["scores_train_l2_normalized"][dim_i], dtype=np.float64)
            indices = np.asarray(z["score_indices"], dtype=np.int64)
            for qi, qart in enumerate(qarts):
                out[qart].append((indices, values[qi]))
    merged = {}
    for qart, chunks in out.items():
        arr = np.full(10000, np.nan, dtype=np.float64)
        for idx, val in chunks:
            arr[idx] = val
        if np.isnan(arr).any():
            raise ValueError(f"incomplete stream score for query artifact {qart}")
        merged[qart] = arr
    return merged


def reconstruct_kept_indices(target_row: dict[str, str], n_scores: int) -> np.ndarray:
    subset_seed = int(target_row["subset_seed"])
    subset_size = int(target_row["subset_size"])
    rng = np.random.default_rng(subset_seed)
    return np.sort(rng.choice(n_scores, size=subset_size, replace=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare x3 query-plan LDS for DAS, OldNext, and OldRef.")
    parser.add_argument("--result-root", default="diffusion_jax_refined/x3/result/experiment1_67")
    parser.add_argument("--query-plan", default=None)
    parser.add_argument("--train-seed", type=int, default=67)
    parser.add_argument("--lambdas", default=LAMBDA_DEFAULT)
    parser.add_argument("--targets", default="simple_loss trajectory_state_mse")
    parser.add_argument("--lds-seeds", default="0 1 2")
    parser.add_argument("--das-reference-lambda", default="200")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    query_plan = Path(args.query_plan) if args.query_plan else result_root / "query_plan" / "query48.tsv"
    lambdas = [x for x in args.lambdas.replace(",", " ").split() if x]
    targets = [x for x in args.targets.replace(",", " ").split() if x]
    lds_seeds = [int(x) for x in args.lds_seeds.replace(",", " ").split() if x]
    plan_rows = load_query_plan(query_plan)

    print(f"[plan] rows={len(plan_rows)} query_plan={query_plan}")
    stream_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    rows_out = []
    grouped = defaultdict(list)

    for target in targets:
        for plan_i, qrow in enumerate(plan_rows, 1):
            key = (target, qrow["score_mode"], path_tag(qrow["query"]), int(qrow["initial_seed"]))
            true_by_subset: list[float] = []
            subset_indices: list[np.ndarray] = []
            for lds_seed in lds_seeds:
                csv_path = target_csv(result_root, qrow, target, lds_seed, args.das_reference_lambda)
                if not csv_path.exists():
                    print(f"[missing-target] {csv_path}", flush=True)
                    continue
                for tr in read_csv_rows(csv_path):
                    true_by_subset.append(float(tr["true_f"]))
                    subset_indices.append(reconstruct_kept_indices(tr, 10000))
            if not true_by_subset:
                continue

            method_scores: dict[str, list[float]] = {}
            for method in ("OldNext", "OldRef"):
                sc_key = (method, qrow["score_mode"])
                if sc_key not in stream_cache:
                    stream_cache[sc_key] = load_stream_scores(stream_root(result_root, method, qrow["score_mode"]))
                qf = query_filter(qrow)
                matches = [qart for qart in stream_cache[sc_key] if qf in qart]
                if len(matches) != 1:
                    print(f"[missing-query] method={method} filter={qf} matches={len(matches)}", flush=True)
                    continue
                scores = stream_cache[sc_key][matches[0]]
                method_scores[method] = [-float(scores[idx].sum()) for idx in subset_indices]

            best_lam = None
            best_lds = float("-inf")
            for lam in lambdas:
                sp = das_score_file(result_root, qrow, lam, args.train_seed)
                if not sp.exists():
                    continue
                scores = np.asarray(np.load(sp), dtype=np.float64).reshape(-1)
                pred = [-float(scores[idx].sum()) for idx in subset_indices]
                lds = 100.0 * spearman(pred, true_by_subset)
                if math.isfinite(lds) and lds > best_lds:
                    best_lds = lds
                    best_lam = lam

            row = {
                "target": target,
                "score_mode": qrow["score_mode"],
                "query": path_tag(qrow["query"]),
                "initial_seed": int(qrow["initial_seed"]),
                "points": len(true_by_subset),
                "DAS_best_lambda": best_lam,
                "DAS": best_lds if best_lam is not None else float("nan"),
                "OldNext": 100.0 * spearman(method_scores.get("OldNext", []), true_by_subset)
                if "OldNext" in method_scores
                else float("nan"),
                "OldRef": 100.0 * spearman(method_scores.get("OldRef", []), true_by_subset)
                if "OldRef" in method_scores
                else float("nan"),
            }
            rows_out.append(row)
            print(
                f"[{len(rows_out):03d}] {target:22s} {row['score_mode']:16s} {row['query']:30s} "
                f"init={row['initial_seed']:2d} n={row['points']:3d} "
                f"DAS={row['DAS']:8.3f}@{row['DAS_best_lambda']} "
                f"OldNext={row['OldNext']:8.3f} OldRef={row['OldRef']:8.3f}",
                flush=True,
            )
            for method in ("DAS", "OldNext", "OldRef"):
                if math.isfinite(row[method]):
                    grouped[(target, method)].append(row[method])

    print("\n== grouped means ==")
    for (target, method), vals in sorted(grouped.items()):
        print(f"{target:22s} {method:8s} n={len(vals):3d} mean={float(np.mean(vals)):8.3f}")

    out_path = Path(args.out) if args.out else result_root / "eval" / "query48_compare" / "pooled_lds_compare.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": rows_out}, indent=2))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
