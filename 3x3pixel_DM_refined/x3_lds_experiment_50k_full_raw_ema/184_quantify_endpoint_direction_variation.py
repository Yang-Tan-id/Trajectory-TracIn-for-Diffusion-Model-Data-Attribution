"""Quantify whether endpoint response variation across directions is small."""

import json

import numpy as np

from endpoint_direction_mc_config import *


TOLERANCES = (0.10, 0.20, 0.30, 0.40, 0.50)
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
BOOTSTRAP_REPEATS = 10000
BOOTSTRAP_SEED = 996100


def bootstrap_source_mean_ci(values_by_source):
    values = np.asarray(values_by_source, dtype=np.float64)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    samples = generator.choice(
        len(values), size=(BOOTSTRAP_REPEATS, len(values)), replace=True
    )
    bootstrap = values[samples].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "source_cluster_bootstrap_95ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "per_source": values.tolist(),
    }


def analyze_response(response_by_source):
    pooled_ratios = []
    pooled_abs_relative = []
    pooled_cv_by_t = []
    pooled_q95_abs_relative_by_t = []
    integrated_ratios = []
    source_coverage = {tolerance: [] for tolerance in TOLERANCES}
    source_integrated_coverage = {tolerance: [] for tolerance in TOLERANCES}
    source_cv = []
    for source_blocks in response_by_source:
        source_ratios = []
        source_cv_values = []
        source_direction_integrated = []
        for response in source_blocks:
            mean_by_t = response.mean(axis=0)
            ratio = response / np.maximum(mean_by_t[None, :], NSDL_EPS)
            abs_relative = np.abs(ratio - 1.0)
            cv_by_t = response.std(axis=0) / np.maximum(mean_by_t, NSDL_EPS)
            q95_by_t = np.quantile(abs_relative, 0.95, axis=0)
            direction_integrated = response.mean(axis=1)
            direction_integrated /= max(direction_integrated.mean(), NSDL_EPS)
            pooled_ratios.append(ratio.reshape(-1))
            pooled_abs_relative.append(abs_relative.reshape(-1))
            pooled_cv_by_t.append(cv_by_t)
            pooled_q95_abs_relative_by_t.append(q95_by_t)
            integrated_ratios.append(direction_integrated)
            source_ratios.append(ratio.reshape(-1))
            source_cv_values.append(cv_by_t)
            source_direction_integrated.append(direction_integrated)
        source_ratios = np.concatenate(source_ratios)
        source_abs_relative = np.abs(source_ratios - 1.0)
        source_integrated = np.concatenate(source_direction_integrated)
        source_integrated_abs_relative = np.abs(source_integrated - 1.0)
        source_cv.append(float(np.mean(np.concatenate(source_cv_values))))
        for tolerance in TOLERANCES:
            source_coverage[tolerance].append(
                float(np.mean(source_abs_relative <= tolerance))
            )
            source_integrated_coverage[tolerance].append(
                float(np.mean(source_integrated_abs_relative <= tolerance))
            )
    ratios = np.concatenate(pooled_ratios)
    abs_relative = np.concatenate(pooled_abs_relative)
    cv_by_t = np.concatenate(pooled_cv_by_t)
    q95_by_t = np.concatenate(pooled_q95_abs_relative_by_t)
    integrated = np.concatenate(integrated_ratios)
    integrated_abs_relative = np.abs(integrated - 1.0)
    return {
        "direction_over_fixed_t_mean_quantiles": {
            str(q): float(np.quantile(ratios, q)) for q in QUANTILES
        },
        "absolute_relative_deviation_quantiles": {
            str(q): float(np.quantile(abs_relative, q)) for q in QUANTILES
        },
        "fixed_t_coverage": {
            str(tolerance): bootstrap_source_mean_ci(
                source_coverage[tolerance]
            )
            for tolerance in TOLERANCES
        },
        "fixed_t_cv": {
            **bootstrap_source_mean_ci(source_cv),
            "quantiles_over_all_branch_t": {
                str(q): float(np.quantile(cv_by_t, q)) for q in QUANTILES
            },
            "max_over_all_branch_t": float(np.max(cv_by_t)),
        },
        "per_t_95pct_direction_band": {
            "quantiles_over_all_branch_t": {
                str(q): float(np.quantile(q95_by_t, q)) for q in QUANTILES
            },
            "max_over_all_branch_t": float(np.max(q95_by_t)),
        },
        "all_t_integrated_direction_over_mean_quantiles": {
            str(q): float(np.quantile(integrated, q)) for q in QUANTILES
        },
        "all_t_integrated_absolute_relative_deviation_quantiles": {
            str(q): float(np.quantile(integrated_abs_relative, q))
            for q in QUANTILES
        },
        "all_t_integrated_coverage": {
            str(tolerance): bootstrap_source_mean_ci(
                source_integrated_coverage[tolerance]
            )
            for tolerance in TOLERANCES
        },
    }


def print_analysis(label, result):
    deviation = result["absolute_relative_deviation_quantiles"]
    coverage = result["fixed_t_coverage"]
    cv = result["fixed_t_cv"]
    band = result["per_t_95pct_direction_band"]
    integrated_deviation = result[
        "all_t_integrated_absolute_relative_deviation_quantiles"
    ]
    print(f"\n[{label}] fixed-t direction variation")
    print(
        f"CV mean={cv['mean']:.4f} "
        f"95%CI=[{cv['source_cluster_bootstrap_95ci'][0]:.4f},"
        f"{cv['source_cluster_bootstrap_95ci'][1]:.4f}] "
        f"CV p95={cv['quantiles_over_all_branch_t']['0.95']:.4f} "
        f"CV max={cv['max_over_all_branch_t']:.4f}"
    )
    print(
        "absolute relative deviation: "
        f"median={deviation['0.5']:.4f} "
        f"p90={deviation['0.9']:.4f} "
        f"p95={deviation['0.95']:.4f} "
        f"p99={deviation['0.99']:.4f}"
    )
    for tolerance in TOLERANCES:
        metric = coverage[str(tolerance)]
        print(
            f"within ±{int(100 * tolerance):2d}%: "
            f"{metric['mean']:.4f} "
            f"95%CI=[{metric['source_cluster_bootstrap_95ci'][0]:.4f},"
            f"{metric['source_cluster_bootstrap_95ci'][1]:.4f}]"
        )
    print(
        "per-t band containing 95% of directions: "
        f"median ±{band['quantiles_over_all_branch_t']['0.5']:.4f}, "
        f"p90 ±{band['quantiles_over_all_branch_t']['0.9']:.4f}, "
        f"worst ±{band['max_over_all_branch_t']:.4f}"
    )
    print(
        "after averaging all t, absolute direction deviation: "
        f"median={integrated_deviation['0.5']:.4f} "
        f"p95={integrated_deviation['0.95']:.4f} "
        f"p99={integrated_deviation['0.99']:.4f}"
    )


def main():
    response_by_source_l2 = []
    response_by_source_squared = []
    for source_index in nsdl_datapoint_indices():
        source_l2 = []
        source_squared = []
        source_dir = edmc_source_dir(source_index)
        for block_index in range(len(NTCD_TIMESTAMP_BLOCKS)):
            path = source_dir / f"block_{block_index}_responses.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as archive:
                squared = archive["actual_sq_l2"].astype(np.float64)
            source_squared.append(squared)
            source_l2.append(np.sqrt(np.maximum(squared, 0.0)))
        response_by_source_l2.append(source_l2)
        response_by_source_squared.append(source_squared)
    output = {
        "definition": {
            "fixed_t_mean": "mean over 100 pollution directions separately at each noise level",
            "coverage": "fraction of direction responses within the stated relative tolerance of the fixed-t direction mean",
            "bootstrap": "10000 resamples clustered by the 10 source datapoints",
        },
        "l2": analyze_response(response_by_source_l2),
        "squared_l2": analyze_response(response_by_source_squared),
    }
    output_path = EDMC_ROOT / "direction_variation_equivalence.json"
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print_analysis("L2", output["l2"])
    print_analysis("squared L2", output["squared_l2"])
    print(f"\n[saved] {output_path}")


if __name__ == "__main__":
    main()
