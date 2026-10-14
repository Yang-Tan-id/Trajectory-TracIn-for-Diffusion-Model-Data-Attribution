"""Relate Top-K predicted-noise changes to endpoint and trajectory changes."""

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, pearsonr, spearmanr


TARGETS = (
    "trajectory_mse",
    "endpoint_mse",
    "simple_loss_abs_change",
    "simple_loss_increase",
)
SCOPES = {
    "all": lambda row: True,
    "prompted": lambda row: row["family"] == "prompted",
    "unprompted": lambda row: row["family"] == "unprompted",
}


def load_prepare_module():
    path = Path(__file__).with_name("117_prepare_retrac_tracindas_topk_100q.py")
    spec = spec_from_file_location("retrac_topk_prepare", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return {
            "n": int(len(x)),
            "spearman": float("nan"),
            "spearman_p_two_sided": float("nan"),
            "pearson": float("nan"),
            "pearson_p_two_sided": float("nan"),
            "log_pearson": float("nan"),
            "log_pearson_p_two_sided": float("nan"),
        }
    rho = spearmanr(x, y)
    raw = pearsonr(x, y)
    tiny = np.finfo(np.float64).tiny
    if np.all(x > 0.0) and np.all(y > 0.0):
        log = pearsonr(np.log10(np.maximum(x, tiny)), np.log10(np.maximum(y, tiny)))
        log_statistic = float(log.statistic)
        log_pvalue = float(log.pvalue)
    else:
        log_statistic = float("nan")
        log_pvalue = float("nan")
    return {
        "n": int(len(x)),
        "spearman": float(rho.statistic),
        "spearman_p_two_sided": float(rho.pvalue),
        "pearson": float(raw.statistic),
        "pearson_p_two_sided": float(raw.pvalue),
        "log_pearson": log_statistic,
        "log_pearson_p_two_sided": log_pvalue,
    }


def difference_alignment(noise, target):
    noise = np.asarray(noise, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(noise) & np.isfinite(target)
    noise = noise[valid]
    target = target[valid]
    nonzero = (noise != 0.0) & (target != 0.0)
    agreements = np.sign(noise[nonzero]) == np.sign(target[nonzero])
    count = int(agreements.sum())
    total = int(len(agreements))
    return {
        **correlation(noise, target),
        "sign_agreement": float(count / total) if total else float("nan"),
        "sign_agreement_count": count,
        "sign_comparison_count": total,
        "sign_agreement_p_greater_than_half": (
            float(binomtest(count, total, 0.5, alternative="greater").pvalue)
            if total
            else float("nan")
        ),
    }


def main():
    prepare = load_prepare_module()
    with open(prepare.OUT_ROOT / "jobs.json") as handle:
        jobs = json.load(handle)

    rows = []
    for job in jobs:
        path = Path(job["job_dir"]) / "evaluation.json"
        with open(path) as handle:
            row = json.load(handle)
        if "noise_trajectory_mse" not in row or "simple_loss_increase" not in row:
            raise KeyError(
                f"{path} lacks the current alignment metrics; rerun the Top-K launcher "
                "with --skip-prepare --eval-only"
            )
        row["removal_fraction_tag"] = job["removal_fraction_tag"]
        row["removal_fraction"] = float(job["removal_fraction"])
        rows.append(row)

    methods = [item["tag"] for item in prepare.METHODS]
    result = {
        "definition": (
            "noise_trajectory_mse is the mean predicted-noise MSE between the "
            "removal EMA model and base EMA model on the 100 saved base-model "
            "trajectory states"
        ),
        "simple_loss_definition": (
            "removal-minus-base EMA simple loss on the same query endpoint and "
            "the fixed LDS 10-timestep x MC4 bank"
        ),
        "correlations": {},
        "pairwise_method_difference_alignment": {},
    }
    lines = [
        "TOP-K NOISE-TRAJECTORY ALIGNMENT",
        "noise = removal-EMA vs base-EMA predicted-noise MSE on the fixed base trajectory",
        "correlations are across queries within each fixed method and removal fraction",
        "",
    ]

    for scope, predicate in SCOPES.items():
        lines.extend(
            [
                f"SCOPE: {scope}",
                "method                                  remove target                    n  spearman       p-rho  pearson  log-pearson",
            ]
        )
        result["correlations"][scope] = {}
        for method in methods:
            result["correlations"][scope][method] = {}
            for fraction in prepare.REMOVAL_FRACTIONS:
                tag = prepare.fraction_tag(fraction)
                selected = [
                    row
                    for row in rows
                    if row["method_tag"] == method
                    and row["removal_fraction_tag"] == tag
                    and predicate(row)
                ]
                result["correlations"][scope][method][tag] = {}
                noise = [row["noise_trajectory_mse"] for row in selected]
                for target in TARGETS:
                    entry = correlation(noise, [row[target] for row in selected])
                    result["correlations"][scope][method][tag][target] = entry
                    lines.append(
                        f"{method:39s} {100*fraction:>5.0f}% {target:23s} "
                        f"{entry['n']:3d} {entry['spearman']:+.4f} "
                        f"{entry['spearman_p_two_sided']:.3g} "
                        f"{entry['pearson']:+.4f} {entry['log_pearson']:+.4f}"
                    )
        lines.append("")

    by_key = {
        (row["method_tag"], row["removal_fraction_tag"], int(row["query_id"])): row
        for row in rows
    }
    lines.extend(
        [
            "PAIRWISE METHOD-DIFFERENCE ALIGNMENT (all 100 queries)",
            "right-minus-left: does the method that changes predicted noise more also change the generated output more?",
            "left -> right                            remove target                    rho       p-rho  sign-agree       p>0.5",
        ]
    )
    for left_index, left in enumerate(methods):
        for right in methods[left_index + 1 :]:
            pair = f"{left}__to__{right}"
            result["pairwise_method_difference_alignment"][pair] = {}
            for fraction in prepare.REMOVAL_FRACTIONS:
                tag = prepare.fraction_tag(fraction)
                left_rows = [by_key[(left, tag, query_id)] for query_id in range(100)]
                right_rows = [by_key[(right, tag, query_id)] for query_id in range(100)]
                noise_difference = [
                    right_row["noise_trajectory_mse"] - left_row["noise_trajectory_mse"]
                    for left_row, right_row in zip(left_rows, right_rows)
                ]
                result["pairwise_method_difference_alignment"][pair][tag] = {}
                for target in TARGETS:
                    target_difference = [
                        right_row[target] - left_row[target]
                        for left_row, right_row in zip(left_rows, right_rows)
                    ]
                    entry = difference_alignment(noise_difference, target_difference)
                    result["pairwise_method_difference_alignment"][pair][tag][target] = entry
                    lines.append(
                        f"{left[:17]:17s}->{right[:17]:17s} {100*fraction:>5.0f}% "
                        f"{target:23s} {entry['spearman']:+.4f} "
                        f"{entry['spearman_p_two_sided']:.3g} "
                        f"{entry['sign_agreement']:.3f} "
                        f"{entry['sign_agreement_p_greater_than_half']:.3g}"
                    )

    json_path = prepare.OUT_ROOT / "noise_output_simple_loss_alignment.json"
    text_path = prepare.OUT_ROOT / "noise_output_simple_loss_alignment.txt"
    with open(json_path, "w") as handle:
        json.dump(result, handle, indent=2)
    with open(text_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"[saved] {json_path}", flush=True)
    print(f"[saved] {text_path}", flush=True)


if __name__ == "__main__":
    main()
