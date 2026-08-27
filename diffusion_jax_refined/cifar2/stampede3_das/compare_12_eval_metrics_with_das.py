from __future__ import annotations

"""Compare LDS eval metrics from experiment_67 and external DAS scores.

This script is meant for the Stampede3/school-server workflow where:

* target/eval results are under result/experiment_67/eval
* DAS score folders were uploaded under result/experiment1_67/attribution_score

It reuses existing lds_results.csv target caches from experiment_67, computes
one selected DAS lambda against those same cached targets when needed, and
writes a compact text report grouped by prompt/query plus overall averages.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


TARGET_ALIASES = {
    "endpoint_counterfactual": "endpoint_contarfactual",
    "traj_counterfactual": "traj_contarfactual",
}


def tag_value(value: str) -> str:
    text = value.replace(",", "__").replace("+", "_")
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    while "__" in text:
        text = text.replace("__", "_")
    text = text.strip("_")
    return text or "unprompted"


def damping_tag(value: str) -> str:
    return tag_value(value.replace("+", "_").replace("-", "neg_").replace(".", "p"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def int_items(text: str) -> list[int]:
    return [int(part) for part in text.replace(",", " ").split() if part.strip()]


def str_items(text: str) -> list[str]:
    return [TARGET_ALIASES.get(part, part) for part in text.replace(",", " ").split() if part.strip()]


def fmt(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:+.4f}"


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def sem(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    m = mean(values)
    variance = sum((value - m) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def parse_eval_summary(path: Path, eval_root: Path) -> dict[str, object] | None:
    rel = path.relative_to(eval_root)
    parts = rel.parts
    if len(parts) < 9:
        return None

    mode = parts[0]
    if mode == "unprompted_solo":
        if len(parts) < 8 or parts[1] != "unprompted":
            return None
        query = "unprompted"
        initial_seed = int(parts[2].replace("initial_seed_", ""))
        algorithm = parts[4]
        target = parts[5]
        lds_group = parts[7]
    elif mode == "prompted_solo":
        if len(parts) < 9 or not parts[1].startswith("query_"):
            return None
        query = parts[1].replace("query_", "")
        initial_seed = int(parts[2].replace("initial_seed_", ""))
        algorithm = parts[4]
        target = parts[5]
        lds_group = parts[7]
    else:
        return None

    if not lds_group.startswith("m_"):
        return None
    try:
        lds_seed = int(lds_group.rsplit("_subset_seed_", 1)[1])
    except (IndexError, ValueError):
        return None
    data = json.loads(path.read_text())
    score = float(data["lds_spearman"])
    return {
        "mode": mode,
        "query": query,
        "initial_seed": initial_seed,
        "algorithm": algorithm,
        "target": TARGET_ALIASES.get(target, target),
        "lds_seed": lds_seed,
        "score": score,
        "path": path,
    }


def target_cache_path(
    eval_root: Path,
    *,
    mode: str,
    query: str,
    initial_seed: int,
    target: str,
    lds_m: int,
    lds_pct: int,
    lds_seed: int,
    pred_tag: str,
) -> Path | None:
    if mode == "unprompted_solo":
        root = eval_root / mode / "unprompted" / f"initial_seed_{initial_seed}" / "lds_unprompted"
    else:
        root = eval_root / mode / f"query_{query}" / f"initial_seed_{initial_seed}" / "lds"
    pattern = (
        f"*/{target}/{pred_tag}/"
        f"m_{lds_m}_k_*_pct_{lds_pct}_subset_seed_{lds_seed}/lds_results.csv"
    )
    matches = sorted(root.glob(pattern))
    complete = [path for path in matches if path.is_file() and len(read_csv_rows(path)) >= lds_m]
    return complete[0] if complete else (matches[0] if matches else None)


def das_score_dir(
    das_result_root: Path,
    *,
    mode: str,
    query: str,
    initial_seed: int,
    train_seed: int,
    lambda_tag: str,
) -> Path | None:
    if mode == "unprompted_solo":
        root = (
            das_result_root
            / "attribution_score"
            / mode
            / f"train_seed_{train_seed}"
            / "unprompted"
            / f"initial_seed_{initial_seed}"
        )
        matches = sorted(root.glob(f"das_unprompted*/lambda_{lambda_tag}"))
    else:
        root = (
            das_result_root
            / "attribution_score"
            / mode
            / f"train_seed_{train_seed}"
            / f"query_{query}"
            / f"initial_seed_{initial_seed}"
        )
        matches = sorted(root.glob(f"das*/lambda_{lambda_tag}"))
    return matches[0] if len(matches) == 1 else None


def compute_fast_das_summary(
    *,
    legacy_root: Path,
    cache_csv: Path,
    score_dir: Path,
    out_dir: Path,
    algorithm: str,
    target: str,
    mode: str,
    prediction_subset: str,
    prediction_sign: float,
    duplicate_policy: str,
    overwrite: bool,
) -> float | None:
    summary_path = out_dir / "lds_summary.json"
    if summary_path.exists() and not overwrite:
        data = json.loads(summary_path.read_text())
        return float(data["lds_spearman"])

    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))
    from LDS.DM_cifar_lds import (  # noqa: PLC0415
        build_score_vector,
        combine_attribution_scores,
        plot_scatter,
        resolve_score_inputs,
        spearman_corr,
        sum_scores,
        write_csv,
    )

    import numpy as np  # noqa: PLC0415

    def prediction_indices(subset_dir: Path):
        filename = (
            "kept_attribution_indices.npy"
            if prediction_subset == "kept"
            else "excluded_attribution_indices.npy"
        )
        return np.load(subset_dir / filename)

    target_rows = read_csv_rows(cache_csv)
    score_inputs = resolve_score_inputs(str(score_dir))
    indices, scores, sources = combine_attribution_scores(score_inputs, duplicate_policy=duplicate_policy)
    score_map = build_score_vector(indices, scores)

    rows = []
    for source_row in target_rows:
        subset_dir = Path(source_row["subset_dir"])
        row = dict(source_row)
        row["prediction_subset"] = prediction_subset
        row["prediction_sign"] = prediction_sign
        row["pred_sum_tau"] = sum_scores(prediction_indices(subset_dir), score_map, prediction_sign)
        rows.append(row)

    pred = np.asarray([float(row["pred_sum_tau"]) for row in rows], dtype=np.float64)
    true = np.asarray([float(row["true_f"]) for row in rows], dtype=np.float64)
    lds = float(spearman_corr(pred, true))

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(str(out_dir / "lds_results.csv"), rows)
    (out_dir / "lds_summary.json").write_text(
        json.dumps(
            {
                "algorithm": algorithm,
                "mode": "unprompted" if mode == "unprompted_solo" else "prompted",
                "score_sources": sources,
                "target_cache": str(cache_csv),
                "num_models": len(rows),
                "lds_spearman": lds,
                "lds_percent": 100.0 * lds if not math.isnan(lds) else float("nan"),
                "target_function": target,
                "prediction_subset": prediction_subset,
                "prediction_sign": prediction_sign,
            },
            indent=2,
        )
    )
    plot_scatter(str(out_dir / "lds_scatter.png"), pred, true, f"LDS={lds:.4f} ({100.0 * lds:.2f}%)")
    return lds


def collect_existing(eval_root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(eval_root.glob("**/lds_summary.json")):
        if "/target_cache/" in str(path):
            continue
        parsed = parse_eval_summary(path, eval_root)
        if parsed is not None:
            rows.append(parsed)
    return rows


def add_das_rows(args: argparse.Namespace, rows: list[dict[str, object]], repo_root: Path) -> list[str]:
    warnings: list[str] = []
    eval_result_root = repo_root / "diffusion_jax_refined" / "cifar2" / "result" / args.eval_experiment
    das_result_root = repo_root / "diffusion_jax_refined" / "cifar2" / "result" / args.das_score_experiment
    eval_root = eval_result_root / "eval"
    legacy_root = repo_root / "diffusion_jax_refined" / "legacy_jax"
    lambda_tag = damping_tag(args.das_lambda)
    das_algorithm = f"das_lambda_{lambda_tag}"
    present_das = {
        (
            str(row["mode"]),
            str(row["query"]),
            int(row["initial_seed"]),
            str(row["target"]),
            int(row["lds_seed"]),
            str(row["algorithm"]),
        )
        for row in rows
        if str(row["algorithm"]) == das_algorithm
    }

    existing_specs = {
        (
            str(row["mode"]),
            str(row["query"]),
            int(row["initial_seed"]),
            str(row["target"]),
            int(row["lds_seed"]),
        )
        for row in rows
        if str(row["target"]) in args.targets
    }
    computed = 0
    for mode, query, initial_seed, target, lds_seed in sorted(existing_specs):
        if mode == "unprompted_solo" and query != "unprompted":
            continue
        score_dir = das_score_dir(
            das_result_root,
            mode=mode,
            query=query,
            initial_seed=initial_seed,
            train_seed=args.train_seed,
            lambda_tag=lambda_tag,
        )
        if score_dir is None:
            warnings.append(
                f"missing DAS score: mode={mode} query={query} initial_seed={initial_seed} lambda={args.das_lambda}"
            )
            continue
        cache_csv = target_cache_path(
            eval_root,
            mode=mode,
            query=query,
            initial_seed=initial_seed,
            target=target,
            lds_m=args.lds_m,
            lds_pct=args.lds_pct,
            lds_seed=lds_seed,
            pred_tag=args.pred_tag,
        )
        if cache_csv is None:
            warnings.append(
                f"missing target cache: mode={mode} query={query} initial_seed={initial_seed} "
                f"target={target} lds_seed={lds_seed}"
            )
            continue
        if mode == "unprompted_solo":
            out_dir = (
                eval_root
                / mode
                / "unprompted"
                / f"initial_seed_{initial_seed}"
                / "lds_unprompted"
                / das_algorithm
                / target
                / args.pred_tag
                / cache_csv.parent.name
            )
        else:
            out_dir = (
                eval_root
                / mode
                / f"query_{query}"
                / f"initial_seed_{initial_seed}"
                / "lds"
                / das_algorithm
                / target
                / args.pred_tag
                / cache_csv.parent.name
            )
        score = compute_fast_das_summary(
            legacy_root=legacy_root,
            cache_csv=cache_csv,
            score_dir=score_dir,
            out_dir=out_dir,
            algorithm=das_algorithm,
            target=target,
            mode=mode,
            prediction_subset=args.prediction_subset,
            prediction_sign=args.prediction_sign,
            duplicate_policy=args.duplicate_policy,
            overwrite=args.overwrite_das,
        )
        if score is None:
            continue
        row_key = (mode, query, initial_seed, target, lds_seed, das_algorithm)
        if row_key in present_das:
            continue
        rows.append(
            {
                "mode": mode,
                "query": query,
                "initial_seed": initial_seed,
                "algorithm": das_algorithm,
                "target": target,
                "lds_seed": lds_seed,
                "score": score,
                "path": out_dir / "lds_summary.json",
            }
        )
        present_das.add(row_key)
        computed += 1
    warnings.append(f"DAS rows added/reused for lambda={args.das_lambda}: {computed}")
    return warnings


def report(rows: list[dict[str, object]], warnings: list[str], args: argparse.Namespace) -> str:
    filtered = [
        row
        for row in rows
        if (not args.algorithms or str(row["algorithm"]) in args.algorithms)
        and (not args.targets or str(row["target"]) in args.targets)
    ]
    lines = []
    lines.append("LDS Metric Comparison")
    lines.append("=" * 80)
    lines.append(f"eval_experiment       : {args.eval_experiment}")
    lines.append(f"das_score_experiment  : {args.das_score_experiment}")
    lines.append(f"DAS lambda            : {args.das_lambda}")
    lines.append(f"targets               : {', '.join(args.targets) if args.targets else 'all'}")
    lines.append(f"algorithms            : {', '.join(args.algorithms) if args.algorithms else 'all'}")
    lines.append("")
    if warnings:
        lines.append("Notes")
        lines.append("-" * 80)
        for item in warnings[:80]:
            lines.append(f"- {item}")
        if len(warnings) > 80:
            lines.append(f"- ... {len(warnings) - 80} more warnings omitted")
        lines.append("")

    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    seed_grouped: dict[tuple[str, str, str, str, int], list[float]] = defaultdict(list)
    for row in filtered:
        key = (str(row["mode"]), str(row["query"]), str(row["target"]), str(row["algorithm"]))
        grouped[key].append(float(row["score"]))
        seed_key = (*key, int(row["initial_seed"]))
        seed_grouped[seed_key].append(float(row["score"]))

    lines.append("Per Prompt + Query + Target")
    lines.append("-" * 80)
    lines.append(f"{'mode':16s} {'query':18s} {'target':24s} {'algorithm':46s} {'n':>5s} {'mean':>9s} {'sem':>9s}")
    for key in sorted(grouped):
        vals = grouped[key]
        mode, query, target, algorithm = key
        lines.append(
            f"{mode:16s} {query:18s} {target:24s} {algorithm:46s} "
            f"{len(vals):5d} {fmt(mean(vals)):>9s} {fmt(sem(vals)):>9s}"
        )
    lines.append("")

    lines.append("Per Initial Seed")
    lines.append("-" * 80)
    lines.append(
        f"{'mode':16s} {'query':18s} {'target':24s} {'algorithm':46s} "
        f"{'seed':>5s} {'n':>5s} {'mean':>9s}"
    )
    for key in sorted(seed_grouped):
        vals = seed_grouped[key]
        mode, query, target, algorithm, initial_seed = key
        lines.append(
            f"{mode:16s} {query:18s} {target:24s} {algorithm:46s} "
            f"{initial_seed:5d} {len(vals):5d} {fmt(mean(vals)):>9s}"
        )
    lines.append("")

    overall: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in filtered:
        overall[(str(row["target"]), str(row["algorithm"]))].append(float(row["score"]))
    lines.append("Overall Average")
    lines.append("-" * 80)
    lines.append(f"{'target':24s} {'algorithm':46s} {'n':>5s} {'mean':>9s} {'sem':>9s}")
    for key in sorted(overall):
        vals = overall[key]
        target, algorithm = key
        lines.append(f"{target:24s} {algorithm:46s} {len(vals):5d} {fmt(mean(vals)):>9s} {fmt(sem(vals)):>9s}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--eval-experiment", default="experiment_67")
    parser.add_argument("--das-score-experiment", default="experiment1_67")
    parser.add_argument("--train-seed", type=int, default=67)
    parser.add_argument("--das-lambda", default="200")
    parser.add_argument("--targets", default="endpoint_contarfactual traj_contarfactual simple_loss trajectory_state_mse")
    parser.add_argument("--algorithms", default="")
    parser.add_argument("--lds-m", type=int, default=64)
    parser.add_argument("--lds-pct", type=int, default=25)
    parser.add_argument("--pred-tag", default="pred_kept_sign_m1")
    parser.add_argument("--prediction-subset", choices=["kept", "removed"], default="kept")
    parser.add_argument("--prediction-sign", type=float, default=-1.0)
    parser.add_argument("--duplicate-policy", choices=["max", "sum", "mean"], default="max")
    parser.add_argument("--no-das", action="store_true", help="Only summarize existing eval summaries.")
    parser.add_argument("--overwrite-das", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    args.targets = str_items(args.targets)
    args.algorithms = set(str_items(args.algorithms)) if args.algorithms.strip() else set()

    script = Path(__file__).resolve()
    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else script.parents[3]
    eval_root = repo_root / "diffusion_jax_refined" / "cifar2" / "result" / args.eval_experiment / "eval"
    if not eval_root.is_dir():
        raise FileNotFoundError(f"Missing eval root: {eval_root}")

    rows = collect_existing(eval_root)
    warnings = [f"existing eval summaries found: {len(rows)}"]
    if not args.no_das:
        warnings.extend(add_das_rows(args, rows, repo_root))

    text = report(rows, warnings, args)
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_path = (
            eval_root
            / "reports"
            / f"compare_12_eval_metrics_with_das_lambda_{damping_tag(args.das_lambda)}.txt"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(text)
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
