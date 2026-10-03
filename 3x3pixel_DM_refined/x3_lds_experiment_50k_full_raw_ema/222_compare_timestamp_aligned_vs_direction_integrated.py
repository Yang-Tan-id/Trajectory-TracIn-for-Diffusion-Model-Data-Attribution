"""Compare legacy timestamp-aligned TracIn-DAS with direction-integrated 100-t."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from direction_integrated_tracin_das_config import DITD_QUERY_IDS, ditd_methods
from exp_config import ATTR_DIR, LDS_DIR, LDS_METRICS, MASK_DIR
from tracin_das_config import tracin_das_methods


TRAIN_T_COUNT = 100
OLD_METHODS = tracin_das_methods(
    noise_mode="checkpoint",
    parameter_projection="projected4096",
    train_noise_mode="aligned",
    query_mc=1,
)
NEW_METHODS = ditd_methods(TRAIN_T_COUNT)


def load_scores(method):
    values = []
    for query_id in DITD_QUERY_IDS:
        path = ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        values.append(np.load(path).astype(np.float64))
    return np.stack(values)


def evaluate(scores, membership, observed, sign):
    return np.asarray(
        [
            spearmanr(sign * (membership @ scores[position]), observed[query_id]).statistic
            for position, query_id in enumerate(DITD_QUERY_IDS)
        ],
        dtype=np.float64,
    )


def format_values(values):
    return " ".join(f"{value:+.6f}" for value in values)


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    report = {
        "comparison": {
            "old": (
                "checkpoint-noise projected4096 TracIn-DAS; train t/noise "
                "aligned to each of 100 query timestamps; one noise term"
            ),
            "new": (
                "direction-integrated projected4096 TracIn-DAS; train loss "
                "averaged over 100 t; noise direction aligned but train t is "
                "not aligned to query t; 100 directions per checkpoint"
            ),
            "warning": (
                "This is a full-method comparison, not an isolated t-alignment "
                "ablation, because direction count also changes from 1 to 100."
            ),
        },
        "results": {},
    }
    lines = [
        "OLD: timestamp-aligned projected4096 TracIn-DAS (100 query t, MC1)",
        "NEW: direction-aligned/train-100t-averaged projected4096 TracIn-DAS (100 directions)",
        "NOTE: direction count also differs, so this is not a one-variable ablation.",
        "",
    ]

    for contraction in ("linear", "termwise_squared", "timestamp_sum_squared"):
        old_method = OLD_METHODS[contraction]
        new_method = NEW_METHODS[contraction]
        old_scores = load_scores(old_method)
        new_scores = load_scores(new_method)
        contraction_result = {
            "old_method": old_method,
            "new_method": new_method,
            "targets": {},
        }
        lines.extend(
            [
                "=" * 112,
                f"CONTRACTION: {contraction}",
                f"old method: {old_method}",
                f"new method: {new_method}",
            ]
        )
        for metric in LDS_METRICS:
            observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
            metric_result = {}
            lines.append(f"\nTARGET: {metric}")
            lines.append(
                "sign  old_mean±std          new_mean±std          delta(new-old)"
            )
            for sign_name, sign in (("negative", -1.0), ("positive", 1.0)):
                old = evaluate(old_scores, membership, observed, sign)
                new = evaluate(new_scores, membership, observed, sign)
                delta = new - old
                metric_result[sign_name] = {
                    "old_mean": float(np.nanmean(old)),
                    "old_std": float(np.nanstd(old)),
                    "new_mean": float(np.nanmean(new)),
                    "new_std": float(np.nanstd(new)),
                    "delta_mean": float(np.nanmean(delta)),
                    "old_per_query": old.tolist(),
                    "new_per_query": new.tolist(),
                    "delta_per_query": delta.tolist(),
                }
                lines.append(
                    f"{sign_name:8s} "
                    f"{np.nanmean(old):+.6f}±{np.nanstd(old):.6f}  "
                    f"{np.nanmean(new):+.6f}±{np.nanstd(new):.6f}  "
                    f"{np.nanmean(delta):+.6f}"
                )
                lines.append(f"  old   q00-q09: {format_values(old)}")
                lines.append(f"  new   q00-q09: {format_values(new)}")
                lines.append(f"  delta q00-q09: {format_values(delta)}")
            contraction_result["targets"][metric] = metric_result
        report["results"][contraction] = contraction_result
        lines.append("")

    stem = "tracin_das_timestamp_aligned_vs_direction_integrated_100t_q00_q09"
    json_path = LDS_DIR / f"{stem}.json"
    text_path = LDS_DIR / f"{stem}.txt"
    LDS_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    text_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"[saved] {json_path}", flush=True)
    print(f"[saved] {text_path}", flush=True)


if __name__ == "__main__":
    main()
