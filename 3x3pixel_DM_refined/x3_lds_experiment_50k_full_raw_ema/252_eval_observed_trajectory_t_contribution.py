"""Determine whether observed trajectory-MSE LDS variation comes from small or large t."""

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from exp_config import LDS_DIR


def safe_rho(x, y):
    return float(spearmanr(x, y).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-ids", default="0-9")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.query_ids.split("-"))
    labels = ("small_t_q1", "q2", "q3", "large_t_q4")
    aggregate = {source: {label: [] for label in labels} for source in ("ema", "raw")}
    result = {"queries": {}}

    for qid in range(lo, hi + 1):
        path = LDS_DIR / "observed_trajectory_per_t" / f"q{qid:02d}.npz"
        data = np.load(path)
        t = data["trajectory_t"].astype(np.int64)
        order = np.argsort(t)
        groups = np.array_split(order, 4)
        query_result = {}
        for source in ("ema", "raw"):
            mse = data[f"mse_{source}"].astype(np.float64)
            full = mse.mean(axis=1)
            mean_total = float(mse.mean())
            var_full = float(np.var(full))
            entries = []
            for label, indices in zip(labels, groups):
                # Contribution retains the original 1/T weighting, so covariance
                # contributions sum exactly to one (up to floating-point error).
                weighted_group = mse[:, indices].sum(axis=1) / mse.shape[1]
                magnitude_fraction = float(mse[:, indices].mean() * len(indices) / (mean_total * mse.shape[1]))
                covariance_contribution = float(
                    np.cov(weighted_group, full, ddof=0)[0, 1] / var_full
                ) if var_full > 0 else float("nan")
                entry = {
                    "label": label,
                    "t_min": int(t[indices].min()),
                    "t_max": int(t[indices].max()),
                    "magnitude_fraction": magnitude_fraction,
                    "variance_covariance_contribution": covariance_contribution,
                    "spearman_with_full_target": safe_rho(mse[:, indices].mean(axis=1), full),
                }
                entries.append(entry)
                aggregate[source][label].append(entry)
            query_result[source] = entries
        result["queries"][f"q{qid:02d}"] = query_result

    result["mean"] = {}
    for source in ("ema", "raw"):
        result["mean"][source] = {}
        print(f"\n[{source.upper()}] mean across q{lo:02d}-q{hi:02d}")
        print("range          magnitude%  LDS-variance%  rho(full)")
        for label in labels:
            rows = aggregate[source][label]
            summary = {
                key: float(np.mean([row[key] for row in rows]))
                for key in (
                    "magnitude_fraction",
                    "variance_covariance_contribution",
                    "spearman_with_full_target",
                )
            }
            result["mean"][source][label] = summary
            print(
                f"{label:14s} {100*summary['magnitude_fraction']:9.2f} "
                f"{100*summary['variance_covariance_contribution']:13.2f} "
                f"{summary['spearman_with_full_target']:+10.4f}"
            )

    output = LDS_DIR / f"observed_trajectory_t_contribution_q{lo:02d}_q{hi:02d}.json"
    with open(output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"\n[saved] {output}")


if __name__ == "__main__":
    main()
