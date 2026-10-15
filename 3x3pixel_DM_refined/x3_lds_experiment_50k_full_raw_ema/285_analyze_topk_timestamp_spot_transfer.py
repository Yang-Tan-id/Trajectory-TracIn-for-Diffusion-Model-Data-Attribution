"""Test whether useful predicted-noise timestamps transfer across removal settings.

This consumes the per-snapshot arrays already produced by
``topk_removal_eval_worker.py``.  No model evaluation is required.
"""

import csv
import json
from importlib.util import module_from_spec, spec_from_file_location
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


SCOPES = {
    "all": lambda row: True,
    "prompted": lambda row: row["family"] == "prompted",
    "unprompted": lambda row: row["family"] == "unprompted",
}
TARGETS = ("endpoint_mse", "trajectory_mse")
SPOT_COUNTS = (1, 5, 10, 20)


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
        return float("nan"), float("nan")
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def setting_label(method, fraction_tag):
    return f"{method}/{fraction_tag}"


def main():
    prepare = load_prepare_module()
    with open(prepare.OUT_ROOT / "jobs.json") as handle:
        jobs = json.load(handle)
    with open(prepare.QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    query_by_id = {int(row["query_id"]): row for row in manifest}

    rows = []
    reference_timesteps = None
    for job in jobs:
        job_dir = Path(job["job_dir"])
        evaluation_path = job_dir / "evaluation.json"
        noise_path = job_dir / "predicted_noise_mse_per_snapshot.npy"
        if not evaluation_path.is_file() or not noise_path.is_file():
            raise FileNotFoundError(
                f"missing timestamp evaluation for {job_dir}; rerun eval-only first"
            )
        with open(evaluation_path) as handle:
            evaluation = json.load(handle)
        query_id = int(job["query_id"])
        query = query_by_id[query_id]
        timesteps = np.asarray(
            np.load(Path(query["dir"]) / "trajectory_t.npy"), dtype=np.int64
        ).reshape(-1)
        noise = np.asarray(np.load(noise_path), dtype=np.float64).reshape(-1)
        if noise.shape != timesteps.shape:
            raise ValueError(
                f"noise/timestep mismatch in {job_dir}: {noise.shape} vs {timesteps.shape}"
            )
        if reference_timesteps is None:
            reference_timesteps = timesteps
        elif not np.array_equal(reference_timesteps, timesteps):
            raise ValueError(f"timestamp grid mismatch for q{query_id:02d}")
        rows.append(
            {
                "query_id": query_id,
                "family": evaluation["family"],
                "method": job["method_tag"],
                "fraction_tag": job["removal_fraction_tag"],
                "fraction": float(job["removal_fraction"]),
                "noise": noise,
                **{target: float(evaluation[target]) for target in TARGETS},
            }
        )

    methods = [item["tag"] for item in prepare.METHODS]
    settings = [
        (method, prepare.fraction_tag(fraction))
        for method in methods
        for fraction in prepare.REMOVAL_FRACTIONS
    ]
    per_timestamp_rows = []
    curves = {}
    setting_data = {}

    for scope, predicate in SCOPES.items():
        curves[scope] = {}
        setting_data[scope] = {}
        for method, fraction_tag in settings:
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["method"] == method
                    and row["fraction_tag"] == fraction_tag
                    and predicate(row)
                ],
                key=lambda row: row["query_id"],
            )
            label = setting_label(method, fraction_tag)
            noise = np.stack([row["noise"] for row in selected], axis=0)
            targets = {
                target: np.asarray([row[target] for row in selected], dtype=np.float64)
                for target in TARGETS
            }
            setting_data[scope][label] = {"noise": noise, **targets}
            curves[scope][label] = {}
            for target in TARGETS:
                rho_values = []
                p_values = []
                for position, timestep in enumerate(reference_timesteps):
                    rho, p_value = correlation(noise[:, position], targets[target])
                    rho_values.append(rho)
                    p_values.append(p_value)
                    per_timestamp_rows.append(
                        {
                            "scope": scope,
                            "method": method,
                            "fraction_tag": fraction_tag,
                            "target": target,
                            "position": position,
                            "timestep": int(timestep),
                            "n": len(selected),
                            "spearman": rho,
                            "p_two_sided": p_value,
                            "noise_mean": float(np.mean(noise[:, position])),
                            "noise_std": float(np.std(noise[:, position])),
                        }
                    )
                curves[scope][label][target] = np.asarray(
                    rho_values, dtype=np.float64
                )

    stability = {}
    transfer = {}
    lines = [
        "X3 TOP-K PER-TIMESTAMP NOISE/OUTPUT CORRELATION",
        "Each curve is Spearman across queries for one method x removal fraction.",
        "Transfer selects timestamp spots using the other two methods only.",
        "",
    ]
    for scope in SCOPES:
        stability[scope] = {}
        transfer[scope] = {}
        labels = list(setting_data[scope])
        lines.append(f"SCOPE: {scope}")
        for target in TARGETS:
            pairwise_curve_rhos = []
            top10_overlaps = []
            pair_entries = []
            for left, right in combinations(labels, 2):
                left_curve = curves[scope][left][target]
                right_curve = curves[scope][right][target]
                curve_rho, curve_p = correlation(left_curve, right_curve)
                left_top = set(np.argsort(-left_curve, kind="stable")[:10].tolist())
                right_top = set(np.argsort(-right_curve, kind="stable")[:10].tolist())
                overlap = len(left_top & right_top) / 10.0
                pairwise_curve_rhos.append(curve_rho)
                top10_overlaps.append(overlap)
                pair_entries.append(
                    {
                        "left": left,
                        "right": right,
                        "curve_spearman": curve_rho,
                        "curve_spearman_p_two_sided": curve_p,
                        "top10_overlap_fraction": overlap,
                    }
                )
            stability[scope][target] = {
                "pairs": pair_entries,
                "mean_curve_spearman": float(np.nanmean(pairwise_curve_rhos)),
                "min_curve_spearman": float(np.nanmin(pairwise_curve_rhos)),
                "mean_top10_overlap_fraction": float(np.mean(top10_overlaps)),
                "min_top10_overlap_fraction": float(np.min(top10_overlaps)),
            }
            lines.append(
                f"  {target}: cross-setting curve rho="
                f"{np.nanmean(pairwise_curve_rhos):+.4f} "
                f"(min {np.nanmin(pairwise_curve_rhos):+.4f}); "
                f"top10 overlap={np.mean(top10_overlaps):.3f} "
                f"(min {np.min(top10_overlaps):.3f})"
            )

            target_transfer = {}
            for heldout in labels:
                heldout_method = heldout.split("/", 1)[0]
                selector_curves = [
                    curves[scope][label][target]
                    for label in labels
                    if label.split("/", 1)[0] != heldout_method
                ]
                selector = np.nanmean(np.stack(selector_curves, axis=0), axis=0)
                heldout_data = setting_data[scope][heldout]
                full_rho, full_p = correlation(
                    heldout_data["noise"].mean(axis=1), heldout_data[target]
                )
                entry = {
                    "full_100_timestamp_spearman": full_rho,
                    "full_100_timestamp_p_two_sided": full_p,
                    "spot_counts": {},
                }
                for count in SPOT_COUNTS:
                    positions = np.argsort(-selector, kind="stable")[:count]
                    score = heldout_data["noise"][:, positions].mean(axis=1)
                    rho, p_value = correlation(score, heldout_data[target])
                    entry["spot_counts"][str(count)] = {
                        "positions": positions.tolist(),
                        "timesteps": reference_timesteps[positions].astype(int).tolist(),
                        "spearman": rho,
                        "p_two_sided": p_value,
                    }
                target_transfer[heldout] = entry
            transfer[scope][target] = target_transfer

            lines.append(
                "    held-out method transfer: spots   mean-rho  mean-delta-vs-full"
            )
            full_values = np.asarray(
                [
                    target_transfer[label]["full_100_timestamp_spearman"]
                    for label in labels
                ],
                dtype=np.float64,
            )
            for count in SPOT_COUNTS:
                values = np.asarray(
                    [
                        target_transfer[label]["spot_counts"][str(count)]["spearman"]
                        for label in labels
                    ],
                    dtype=np.float64,
                )
                lines.append(
                    f"      {count:3d}                 {np.nanmean(values):+.4f} "
                    f"{np.nanmean(values - full_values):+.4f}"
                )
            lines.append(f"      100 (full)          {np.nanmean(full_values):+.4f}")

        mean_endpoint_curve = np.nanmean(
            np.stack(
                [curves[scope][label]["endpoint_mse"] for label in labels], axis=0
            ),
            axis=0,
        )
        best_positions = np.argsort(-mean_endpoint_curve, kind="stable")[:15]
        lines.append("  consensus endpoint spots:")
        lines.append(
            "    "
            + ", ".join(
                f"t={int(reference_timesteps[position])}:rho={mean_endpoint_curve[position]:+.3f}"
                for position in best_positions
            )
        )
        lines.append("")

    output_root = prepare.OUT_ROOT / "timestamp_spot_transfer"
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "per_timestamp_correlations.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(per_timestamp_rows[0]))
        writer.writeheader()
        writer.writerows(per_timestamp_rows)

    result = {
        "definition": (
            "At each saved base-trajectory timestamp, correlate removal-vs-base "
            "predicted-noise MSE with output MSE across queries."
        ),
        "timesteps": reference_timesteps.astype(int).tolist(),
        "stability": stability,
        "leave_one_method_out_transfer": transfer,
    }
    json_path = output_root / "timestamp_spot_transfer.json"
    text_path = output_root / "timestamp_spot_transfer.txt"
    with open(json_path, "w") as handle:
        json.dump(result, handle, indent=2)
    with open(text_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        for axis, scope in zip(axes, SCOPES):
            labels = list(curves[scope])
            for label in labels:
                axis.plot(
                    reference_timesteps,
                    curves[scope][label]["endpoint_mse"],
                    alpha=0.32,
                    linewidth=1.0,
                )
            mean_curve = np.nanmean(
                np.stack(
                    [curves[scope][label]["endpoint_mse"] for label in labels],
                    axis=0,
                ),
                axis=0,
            )
            axis.plot(
                reference_timesteps,
                mean_curve,
                color="black",
                linewidth=2.5,
                label="mean across 9 settings",
            )
            axis.axhline(0.0, color="gray", linewidth=0.8)
            axis.set_title(scope)
            axis.set_ylabel("Spearman vs endpoint MSE")
            axis.grid(alpha=0.2)
            axis.legend()
        axes[-1].set_xlabel("diffusion timestep")
        figure.tight_layout()
        plot_path = output_root / "endpoint_spearman_by_timestamp.png"
        figure.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    except ImportError:
        plot_path = None

    print("\n".join(lines), flush=True)
    print(f"[saved] {csv_path}", flush=True)
    print(f"[saved] {json_path}", flush=True)
    print(f"[saved] {text_path}", flush=True)
    if plot_path is not None:
        print(f"[saved] {plot_path}", flush=True)


if __name__ == "__main__":
    main()
