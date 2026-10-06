"""Compare 20-t trajectory inverse-noise DAS with original final-EMA DAS."""

import json

import numpy as np
from scipy.stats import wilcoxon

from exp_config import DAS_LAMBDAS, LDS_DIR, LDS_METRICS
from trajectory_inverse_noise_das_config import lambda_tag


TRAJECTORY_RESULT = (
    LDS_DIR / "trajectory_inverse_noise_das_20t_100q_lambda_sweep.json"
)
TEXT_OUTPUT = (
    LDS_DIR / "trajectory_inverse_noise_das_20t_vs_original_das_100q.txt"
)
JSON_OUTPUT = (
    LDS_DIR / "trajectory_inverse_noise_das_20t_vs_original_das_100q.json"
)


def original_result_path(metric, lam):
    return LDS_DIR / f"das_ema_{metric}_lambda_{lambda_tag(lam)}.json"


def load_original(metric, lam):
    path = original_result_path(metric, lam)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing original DAS LDS result: {path}\n"
            "Run: python -u 06_lds_eval.py --method das_ema "
            f"--metric {metric} --lambda {float(lam):g}"
        )
    with open(path) as handle:
        payload = json.load(handle)
    queries = payload["queries"]
    by_query = {
        int(item["query_id"]): float(item["spearman"])
        for item in queries
    }
    if sorted(by_query) != list(range(100)):
        raise ValueError(f"original DAS result does not cover q00-q99: {path}")
    return np.asarray([by_query[query_id] for query_id in range(100)])


def paired_summary(trajectory, original):
    difference = trajectory - original
    finite = np.isfinite(difference)
    try:
        pvalue = float(
            wilcoxon(difference[finite], alternative="two-sided").pvalue
        )
    except ValueError:
        pvalue = 1.0
    return {
        "trajectory_mean": float(np.nanmean(trajectory)),
        "trajectory_std": float(np.nanstd(trajectory)),
        "original_mean": float(np.nanmean(original)),
        "original_std": float(np.nanstd(original)),
        "mean_difference": float(np.nanmean(difference)),
        "wilcoxon_two_sided_p": pvalue,
        "trajectory_wins": int(np.count_nonzero(difference > 0)),
        "ties": int(np.count_nonzero(difference == 0)),
        "query_count": int(np.count_nonzero(finite)),
        "trajectory_per_query": trajectory.tolist(),
        "original_per_query": original.tolist(),
        "difference_per_query": difference.tolist(),
    }


def main():
    if not TRAJECTORY_RESULT.is_file():
        raise FileNotFoundError(
            f"missing 20-t trajectory DAS result: {TRAJECTORY_RESULT}\n"
            "Run the 20-t launcher first."
        )
    with open(TRAJECTORY_RESULT) as handle:
        trajectory_payload = json.load(handle)

    result = {
        "trajectory_result": str(TRAJECTORY_RESULT),
        "original_method": "das_ema",
        "query_ids": list(range(100)),
        "prediction_sign": -1,
        "test": "paired two-sided Wilcoxon signed-rank",
        "same_lambda": {},
        "best_lambda_summary": {},
    }
    lines = [
        "20-T TRAJECTORY INVERSE-NOISE DAS VS ORIGINAL DAS",
        "=" * 126,
        "q00-q99 | final EMA | projected4096 | sign=-1 | "
        "paired two-sided Wilcoxon",
        "All rows compare the same lambda unless explicitly marked as best.",
    ]

    for metric in LDS_METRICS:
        lines.extend(
            [
                "",
                f"TARGET: {metric}",
                "-" * 126,
                "lambda    trajectory mean+/-std      original mean+/-std"
                "        delta       p-wilcoxon    wins/100",
            ]
        )
        metric_result = {}
        trajectory_means = {}
        original_means = {}
        for lam_raw in DAS_LAMBDAS:
            lam = float(lam_raw)
            tag = lambda_tag(lam)
            trajectory = np.asarray(
                trajectory_payload["results"][tag][metric]["negative"][
                    "per_query"
                ],
                dtype=np.float64,
            )
            original = load_original(metric, lam)
            if trajectory.shape != (100,) or original.shape != (100,):
                raise ValueError(f"unexpected per-query shape for {metric}, lambda={lam}")
            summary = paired_summary(trajectory, original)
            metric_result[tag] = summary
            trajectory_means[lam] = summary["trajectory_mean"]
            original_means[lam] = summary["original_mean"]
            lines.append(
                f"{lam:8g}  "
                f"{summary['trajectory_mean']:+.6f}+/-"
                f"{summary['trajectory_std']:.6f}   "
                f"{summary['original_mean']:+.6f}+/-"
                f"{summary['original_std']:.6f}   "
                f"{summary['mean_difference']:+.6f}   "
                f"{summary['wilcoxon_two_sided_p']:.6g}   "
                f"{summary['trajectory_wins']:03d}/100"
            )
        result["same_lambda"][metric] = metric_result

        trajectory_best_lambda = max(
            trajectory_means, key=trajectory_means.get
        )
        original_best_lambda = max(original_means, key=original_means.get)
        matched = metric_result[lambda_tag(trajectory_best_lambda)]
        best_summary = {
            "trajectory_best_lambda": trajectory_best_lambda,
            "trajectory_best_mean": trajectory_means[trajectory_best_lambda],
            "original_best_lambda": original_best_lambda,
            "original_best_mean": original_means[original_best_lambda],
            "original_mean_at_trajectory_best_lambda": matched["original_mean"],
            "same_lambda_mean_difference": matched["mean_difference"],
            "same_lambda_wilcoxon_two_sided_p": matched[
                "wilcoxon_two_sided_p"
            ],
            "same_lambda_trajectory_wins": matched["trajectory_wins"],
        }
        result["best_lambda_summary"][metric] = best_summary
        lines.extend(
            [
                "BEST:",
                f"  trajectory: lambda={trajectory_best_lambda:g} "
                f"mean={trajectory_means[trajectory_best_lambda]:+.6f}",
                f"  original:   lambda={original_best_lambda:g} "
                f"mean={original_means[original_best_lambda]:+.6f}",
                f"  matched at trajectory lambda={trajectory_best_lambda:g}: "
                f"delta={matched['mean_difference']:+.6f} "
                f"p={matched['wilcoxon_two_sided_p']:.6g} "
                f"wins={matched['trajectory_wins']}/100",
            ]
        )

    TEXT_OUTPUT.write_text("\n".join(lines) + "\n")
    with open(JSON_OUTPUT, "w") as handle:
        json.dump(result, handle, indent=2)
    print("\n".join(lines))
    print(f"\n[saved] {TEXT_OUTPUT}")
    print(f"[saved] {JSON_OUTPUT}")


if __name__ == "__main__":
    main()
