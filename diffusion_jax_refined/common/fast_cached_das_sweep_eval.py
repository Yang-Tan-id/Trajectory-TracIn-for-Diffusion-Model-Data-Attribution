from __future__ import annotations

"""Fast DAS LDS lambda sweep from cached target rows.

The expensive LDS target value ``true_f`` is independent of DAS damping lambda.
This script reuses any existing ``lds_results.csv`` for a
query/initial-seed/target/LDS-seed group, then recomputes only ``pred_sum_tau``
and Spearman for each DAS lambda score folder.
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


DEFAULT_LAMBDAS = (
    "0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10 20 50 100 200 500 "
    "1000 2000 5000 10000 20000 50000"
)


def tag_value(value: str) -> str:
    text = value.replace(",", "__").replace("+", "_")
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    while "__" in text:
        text = text.replace("__", "_")
    text = text.strip("_")
    return text or "unprompted"


def damping_tag(value: str) -> str:
    return tag_value(value.replace("+", "_").replace("-", "neg_").replace(".", "p"))


def int_items(text: str) -> list[int]:
    return [int(part) for part in text.replace(",", " ").split() if part.strip()]


def str_items(text: str) -> list[str]:
    return [part for part in text.replace(",", " ").split() if part.strip()]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def prediction_indices(subset_dir: Path, subset: str) -> np.ndarray:
    name = "kept_attribution_indices.npy" if subset == "kept" else "excluded_attribution_indices.npy"
    return np.load(subset_dir / name)


def find_one(pattern: str) -> Path | None:
    matches = sorted(Path().glob(pattern) if not pattern.startswith("/") else Path("/").glob(pattern[1:]))
    if len(matches) == 1:
        return matches[0]
    return None


def score_dir_for_lambda(
    result_root: Path,
    *,
    score_mode: str,
    train_seed: int,
    query: str,
    initial_seed: int,
    unprompted: bool,
    lambda_tag: str,
) -> Path | None:
    if unprompted:
        pattern = (
            result_root
            / "attribution_score"
            / score_mode
            / f"train_seed_{train_seed}"
            / "unprompted"
            / f"initial_seed_{initial_seed}"
            / "das_unprompted*"
            / f"lambda_{lambda_tag}"
        )
    else:
        pattern = (
            result_root
            / "attribution_score"
            / score_mode
            / f"train_seed_{train_seed}"
            / f"query_{tag_value(query)}"
            / f"initial_seed_{initial_seed}"
            / "das*"
            / f"lambda_{lambda_tag}"
        )
    matches = sorted(pattern.parent.glob(pattern.name))
    return matches[0] if len(matches) == 1 else None


def target_cache_for_group(
    result_root: Path,
    *,
    score_mode: str,
    query: str,
    initial_seed: int,
    unprompted: bool,
    target: str,
    pred_tag: str,
    lds_m: int,
    lds_pct: int,
    lds_seed: int,
) -> Path | None:
    if unprompted:
        root = (
            result_root
            / "eval"
            / score_mode
            / "unprompted"
            / f"initial_seed_{initial_seed}"
            / "lds_unprompted"
        )
    else:
        root = (
            result_root
            / "eval"
            / score_mode
            / f"query_{tag_value(query)}"
            / f"initial_seed_{initial_seed}"
            / "lds"
        )
    pattern = (
        f"das_lambda_*/{target}/{pred_tag}/"
        f"m_{lds_m}_k_*_pct_{lds_pct}_subset_seed_{lds_seed}/lds_results.csv"
    )
    matches = sorted(root.glob(pattern))
    if not matches:
        return None
    # Prefer an already complete 64-row cache.
    complete = [path for path in matches if len(read_rows(path)) >= lds_m]
    return complete[0] if complete else matches[0]


def output_dir_for_group(
    result_root: Path,
    *,
    score_mode: str,
    query: str,
    initial_seed: int,
    unprompted: bool,
    lambda_tag: str,
    target: str,
    pred_tag: str,
    cache_csv: Path,
) -> Path:
    model_name = cache_csv.parent.name
    if unprompted:
        return (
            result_root
            / "eval"
            / score_mode
            / "unprompted"
            / f"initial_seed_{initial_seed}"
            / "lds_unprompted"
            / f"das_lambda_{lambda_tag}"
            / target
            / pred_tag
            / model_name
        )
    return (
        result_root
        / "eval"
        / score_mode
        / f"query_{tag_value(query)}"
        / f"initial_seed_{initial_seed}"
        / "lds"
        / f"das_lambda_{lambda_tag}"
        / target
        / pred_tag
        / model_name
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Dataset dataset_config.py")
    parser.add_argument("--result-root", required=True, help="Dataset result/<experiment> directory.")
    parser.add_argument("--train-seed", type=int, default=67)
    parser.add_argument("--lds-m", type=int, default=64)
    parser.add_argument("--lds-pct", type=int, default=25)
    parser.add_argument("--lds-seeds", default="0 1 2 3 4 5 6 7")
    parser.add_argument("--targets", default="trajectory_state_mse simple_loss")
    parser.add_argument("--prompted-initial-seeds", default="0 1 2 3 4 5 6 7")
    parser.add_argument("--unprompted-initial-seeds", default="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23")
    parser.add_argument("--lambdas", default=DEFAULT_LAMBDAS)
    parser.add_argument("--pred-tag", default="pred_kept_sign_m1")
    parser.add_argument("--prediction-subset", choices=["kept", "removed"], default="kept")
    parser.add_argument("--prediction-sign", type=float, default=-1.0)
    parser.add_argument("--duplicate-policy", choices=["max", "sum", "mean"], default="max")
    parser.add_argument("--overwrite", action="store_true")
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

    result_root = Path(args.result_root).expanduser().resolve()
    lds_seeds = int_items(args.lds_seeds)
    targets = str_items(args.targets)
    lambdas = str_items(args.lambdas)
    prompted_initial_seeds = int_items(args.prompted_initial_seeds)
    unprompted_initial_seeds = int_items(args.unprompted_initial_seeds)

    specs: list[tuple[str, str, str, int, bool]] = []
    for seed in unprompted_initial_seeds:
        specs.append(("unprompted_solo", "unprompted", "unconditional", seed, True))
    for seed in prompted_initial_seeds:
        specs.extend(
            [
                ("prompted_solo", "horse", "horse", seed, False),
                ("prompted_solo", "automobile", "automobile", seed, False),
                ("prompted_solo", "horse_automobile", "horse,automobile", seed, False),
            ]
        )

    written = skipped = missing_cache = missing_score = 0
    started = time.time()
    for score_mode, query_path, query_env, initial_seed, unprompted in specs:
        for target in targets:
            for lds_seed in lds_seeds:
                cache_csv = target_cache_for_group(
                    result_root,
                    score_mode=score_mode,
                    query=query_env,
                    initial_seed=initial_seed,
                    unprompted=unprompted,
                    target=target,
                    pred_tag=args.pred_tag,
                    lds_m=args.lds_m,
                    lds_pct=args.lds_pct,
                    lds_seed=lds_seed,
                )
                if cache_csv is None:
                    missing_cache += len(lambdas)
                    print(
                        f"[missing-cache] mode={score_mode} query={query_path} "
                        f"initial_seed={initial_seed} target={target} lds_seed={lds_seed}",
                        flush=True,
                    )
                    continue
                target_rows = read_rows(cache_csv)
                if len(target_rows) < args.lds_m:
                    missing_cache += len(lambdas)
                    print(f"[partial-cache] {cache_csv} rows={len(target_rows)}", flush=True)
                    continue

                for damping in lambdas:
                    lambda_tag = damping_tag(damping)
                    out_dir = output_dir_for_group(
                        result_root,
                        score_mode=score_mode,
                        query=query_env,
                        initial_seed=initial_seed,
                        unprompted=unprompted,
                        lambda_tag=lambda_tag,
                        target=target,
                        pred_tag=args.pred_tag,
                        cache_csv=cache_csv,
                    )
                    if (out_dir / "lds_summary.json").exists() and not args.overwrite:
                        skipped += 1
                        continue
                    score_dir = score_dir_for_lambda(
                        result_root,
                        score_mode=score_mode,
                        train_seed=args.train_seed,
                        query=query_env,
                        initial_seed=initial_seed,
                        unprompted=unprompted,
                        lambda_tag=lambda_tag,
                    )
                    if score_dir is None:
                        missing_score += 1
                        print(
                            f"[missing-score] mode={score_mode} query={query_path} "
                            f"initial_seed={initial_seed} lambda={damping}",
                            flush=True,
                        )
                        continue

                    score_inputs = resolve_score_inputs(str(score_dir))
                    indices, scores, sources = combine_attribution_scores(
                        score_inputs,
                        duplicate_policy=args.duplicate_policy,
                    )
                    score_map = build_score_vector(indices, scores)
                    rows = []
                    for row in target_rows:
                        subset_dir = Path(row["subset_dir"])
                        pred_indices = prediction_indices(subset_dir, args.prediction_subset)
                        out_row = dict(row)
                        out_row["prediction_subset"] = args.prediction_subset
                        out_row["prediction_sign"] = args.prediction_sign
                        out_row["pred_sum_tau"] = sum_scores(
                            pred_indices,
                            score_map,
                            args.prediction_sign,
                        )
                        rows.append(out_row)
                    pred = np.asarray([float(row["pred_sum_tau"]) for row in rows], dtype=np.float64)
                    true = np.asarray([float(row["true_f"]) for row in rows], dtype=np.float64)
                    lds = spearman_corr(pred, true)

                    out_dir.mkdir(parents=True, exist_ok=True)
                    write_csv(str(out_dir / "lds_results.csv"), rows)
                    summary = {
                        "algorithm": f"das_lambda_{lambda_tag}",
                        "mode": "unprompted" if unprompted else "prompted",
                        "score_sources": sources,
                        "target_cache": str(cache_csv),
                        "num_models": len(rows),
                        "lds_spearman": float(lds),
                        "lds_percent": float(100.0 * lds) if not math.isnan(lds) else float("nan"),
                        "target_function": target,
                        "prediction_subset": args.prediction_subset,
                        "prediction_sign": args.prediction_sign,
                        "elapsed_sec": time.time() - started,
                    }
                    (out_dir / "lds_summary.json").write_text(json.dumps(summary, indent=2))
                    plot_scatter(
                        str(out_dir / "lds_scatter.png"),
                        pred,
                        true,
                        f"LDS={lds:.4f} ({100.0 * lds:.2f}%)",
                    )
                    written += 1
                    print(f"[saved] {out_dir} | LDS={lds:.6f}", flush=True)

    print(
        "done | "
        f"written={written} skipped={skipped} missing_cache={missing_cache} "
        f"missing_score={missing_score} elapsed={time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
