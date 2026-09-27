"""Save and evaluate fixed-alpha next/previous Traj score combinations."""

import argparse
import csv
import json
import os
import re

import numpy as np

from checkpoint_counterfactual_metrics import spearman_correlation
from exp_config import *


CONTRACTIONS = ("linear", "termwise_squared", "timestamp_sum_squared")
ALPHAS = (-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0)
PRIMARY_METRIC = "traj_ref_raw"


def alpha_tag(alpha):
    sign = "p" if float(alpha) >= 0 else "m"
    magnitude = str(abs(float(alpha))).replace(".", "p")
    return f"{sign}{magnitude}"


def combined_method(contraction, alpha, output_suffix=""):
    suffix = f"_{output_suffix}" if output_suffix else ""
    return f"traj_next_previous_affine_{contraction}_alpha_{alpha_tag(alpha)}{suffix}"


def selected_method(contraction, selection, output_suffix=""):
    suffix = f"_{output_suffix}" if output_suffix else ""
    return f"traj_next_previous_affine_{contraction}_{selection}_traj_ref_raw{suffix}"


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-method-suffix",
        default="",
        help="suffix on both next and previous source attribution methods",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="suffix on generated alpha and selected attribution methods",
    )
    parser.add_argument(
        "--result-stem",
        default="traj_next_previous_alpha_sweep",
        help="basename for LDS JSON and per-query CSV outputs",
    )
    args = parser.parse_args()
    for label, value in (
        ("source method suffix", args.source_method_suffix),
        ("output suffix", args.output_suffix),
        ("result stem", args.result_stem),
    ):
        if value and not re.fullmatch(r"[A-Za-z0-9_]+", value):
            raise ValueError(
                f"{label} may contain only letters, digits, and underscores"
            )
    source_suffix = (
        f"_{args.source_method_suffix}" if args.source_method_suffix else ""
    )

    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    observed = {
        metric: np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
        for metric in LDS_METRICS
    }
    query_count = observed[PRIMARY_METRIC].shape[0]
    if query_count != 100:
        raise ValueError(f"expected 100 LDS queries, found {query_count}")

    predictions = {
        contraction: {
            alpha: np.empty((query_count, membership.shape[0]), dtype=np.float64)
            for alpha in ALPHAS
        }
        for contraction in CONTRACTIONS
    }
    for contraction in CONTRACTIONS:
        next_method = f"traj_projected_first_raw_{contraction}{source_suffix}"
        previous_method = f"traj_projected_backward_first_raw_{contraction}{source_suffix}"
        for query_id in range(query_count):
            next_path = ATTR_DIR / next_method / f"q{query_id:02d}" / "scores.npy"
            previous_path = (
                ATTR_DIR / previous_method / f"q{query_id:02d}" / "scores.npy"
            )
            if not next_path.is_file():
                raise FileNotFoundError(next_path)
            if not previous_path.is_file():
                raise FileNotFoundError(previous_path)
            next_score = np.load(next_path).astype(np.float64).reshape(-1)
            previous_score = np.load(previous_path).astype(np.float64).reshape(-1)
            if next_score.shape != (N_TRAIN,) or previous_score.shape != (N_TRAIN,):
                raise ValueError(
                    f"q{query_id:02d} {contraction}: next={next_score.shape}, "
                    f"previous={previous_score.shape}"
                )
            if not np.isfinite(next_score).all() or not np.isfinite(previous_score).all():
                raise ValueError(f"q{query_id:02d} {contraction}: non-finite score")

            for alpha in ALPHAS:
                score = (1.0 - alpha) * next_score + alpha * previous_score
                saved_score = score.astype(np.float32)
                method = combined_method(contraction, alpha, args.output_suffix)
                output = ATTR_DIR / method / f"q{query_id:02d}"
                output.mkdir(parents=True, exist_ok=True)
                np.save(output / "scores.npy", saved_score)
                predictions[contraction][alpha][query_id] = (
                    membership @ saved_score.astype(np.float64)
                )
                atomic_json(
                    output / "info.json",
                    {
                        "method": method,
                        "query_id": query_id,
                        "contraction": contraction,
                        "alpha": alpha,
                        "formula": "(1-alpha)*next + alpha*previous",
                        "next_method": next_method,
                        "previous_method": previous_method,
                        "next_checkpoint_pairs": (
                            "defined by next source info.json"
                            if args.source_method_suffix
                            else "c=1..49, target=c+1; final excluded"
                        ),
                        "previous_checkpoint_pairs": (
                            "defined by previous source info.json"
                            if args.source_method_suffix
                            else "c=2..50, target=c-1; first excluded"
                        ),
                        "score_dtype": "float32",
                    },
                )
        print(f"[scores saved] {contraction} q00-q99", flush=True)

    rows = []
    results = {}
    for contraction in CONTRACTIONS:
        contraction_results = {}
        for metric, target in observed.items():
            alpha_results = []
            for alpha in ALPHAS:
                query_rows = []
                for query_id in range(query_count):
                    prediction = -predictions[contraction][alpha][query_id]
                    rho = spearman_correlation(prediction, target[query_id])
                    query_rows.append({"query_id": query_id, "spearman": rho})
                    rows.append(
                        {
                            "contraction": contraction,
                            "metric": metric,
                            "alpha": alpha,
                            "alpha_branch": (
                                "negative" if alpha < 0 else "positive" if alpha > 0 else "next_base"
                            ),
                            "query_id": query_id,
                            "spearman": rho,
                        }
                    )
                alpha_results.append(
                    {
                        "alpha": alpha,
                        "method": combined_method(
                            contraction, alpha, args.output_suffix
                        ),
                        "mean": float(
                            np.nanmean([item["spearman"] for item in query_rows])
                        ),
                        "queries": query_rows,
                    }
                )

            negative = [item for item in alpha_results if item["alpha"] < 0]
            positive = [item for item in alpha_results if item["alpha"] > 0]
            next_base = next(item for item in alpha_results if item["alpha"] == 0)
            best_negative = max(negative, key=lambda item: item["mean"])
            best_positive = max(positive, key=lambda item: item["mean"])
            best_overall = max(alpha_results, key=lambda item: item["mean"])
            contraction_results[metric] = {
                "prediction": "-(membership @ combined_score)",
                "alpha_results": alpha_results,
                "next_base": {
                    "alpha": 0.0,
                    "mean": next_base["mean"],
                },
                "best_negative": {
                    "alpha": best_negative["alpha"],
                    "mean": best_negative["mean"],
                    "improvement_over_next": best_negative["mean"] - next_base["mean"],
                },
                "best_positive": {
                    "alpha": best_positive["alpha"],
                    "mean": best_positive["mean"],
                    "improvement_over_next": best_positive["mean"] - next_base["mean"],
                },
                "best_overall": {
                    "alpha": best_overall["alpha"],
                    "mean": best_overall["mean"],
                    "improvement_over_next": best_overall["mean"] - next_base["mean"],
                },
            }
        results[contraction] = contraction_results

    primary_selection = {}
    selection_rows = []
    for contraction in CONTRACTIONS:
        primary = results[contraction][PRIMARY_METRIC]
        alpha_results = primary["alpha_results"]
        global_alpha = float(primary["best_overall"]["alpha"])
        global_mean = float(primary["best_overall"]["mean"])
        global_method = selected_method(
            contraction, "global_best", args.output_suffix
        )
        per_query_method = selected_method(
            contraction, "per_query_best", args.output_suffix
        )
        per_query = []
        for query_id in range(query_count):
            candidates = [
                {
                    "alpha": float(item["alpha"]),
                    "spearman": float(item["queries"][query_id]["spearman"]),
                }
                for item in alpha_results
            ]
            best = max(
                candidates,
                key=lambda item: (
                    -np.inf if np.isnan(item["spearman"]) else item["spearman"]
                ),
            )
            per_query.append(
                {
                    "query_id": query_id,
                    "alpha": best["alpha"],
                    "spearman": best["spearman"],
                }
            )
            selection_rows.append(
                {
                    "contraction": contraction,
                    "query_id": query_id,
                    "per_query_best_alpha": best["alpha"],
                    "per_query_best_spearman": best["spearman"],
                    "global_best_alpha": global_alpha,
                    "global_best_mean_spearman": global_mean,
                }
            )

            selections = (
                (global_method, global_alpha, "one alpha shared by all queries"),
                (per_query_method, best["alpha"], "oracle alpha selected separately per query"),
            )
            for method, alpha, semantics in selections:
                source = (
                    ATTR_DIR
                    / combined_method(contraction, alpha, args.output_suffix)
                    / f"q{query_id:02d}"
                    / "scores.npy"
                )
                score = np.load(source).astype(np.float32)
                output = ATTR_DIR / method / f"q{query_id:02d}"
                output.mkdir(parents=True, exist_ok=True)
                np.save(output / "scores.npy", score)
                atomic_json(
                    output / "info.json",
                    {
                        "method": method,
                        "query_id": query_id,
                        "contraction": contraction,
                        "selection_metric": PRIMARY_METRIC,
                        "selection_semantics": semantics,
                        "selected_alpha": alpha,
                        "source_method": combined_method(
                            contraction, alpha, args.output_suffix
                        ),
                        "formula": "(1-alpha)*next + alpha*previous",
                    },
                )

        primary_selection[contraction] = {
            "global": {
                "alpha": global_alpha,
                "mean_spearman": global_mean,
                "method": global_method,
            },
            "per_query_oracle": {
                "mean_spearman": float(
                    np.nanmean([item["spearman"] for item in per_query])
                ),
                "method": per_query_method,
                "queries": per_query,
            },
        }

    payload = {
        "formula": "combined=(1-alpha)*next + alpha*previous",
        "alphas": list(ALPHAS),
        "alpha_interpretation": {
            "zero": "next baseline",
            "positive_0_to_1": "convex interpolation from next to previous",
            "negative": "affine extrapolation subtracting the previous direction",
            "one": "previous only",
        },
        "prediction": "-(membership @ combined_score)",
        "primary_metric": PRIMARY_METRIC,
        "query_count": query_count,
        "source_method_suffix": args.source_method_suffix,
        "output_suffix": args.output_suffix,
        "primary_metric_alpha_selection": primary_selection,
        "results": results,
    }
    json_path = LDS_DIR / f"{args.result_stem}.json"
    csv_path = LDS_DIR / f"{args.result_stem}_per_query.csv"
    selection_csv_path = (
        LDS_DIR / f"{args.result_stem}_selection_traj_ref_raw.csv"
    )
    atomic_json(json_path, payload)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "contraction",
                "metric",
                "alpha",
                "alpha_branch",
                "query_id",
                "spearman",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    with open(selection_csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "contraction",
                "query_id",
                "per_query_best_alpha",
                "per_query_best_spearman",
                "global_best_alpha",
                "global_best_mean_spearman",
            ),
        )
        writer.writeheader()
        writer.writerows(selection_rows)

    print(f"[primary metric: {PRIMARY_METRIC}]", flush=True)
    for contraction in CONTRACTIONS:
        result = results[contraction][PRIMARY_METRIC]
        selection = primary_selection[contraction]
        print(
            f"  {contraction}: next(alpha=0)={result['next_base']['mean']:+.6f} | "
            f"best negative alpha={result['best_negative']['alpha']:+g} "
            f"mean={result['best_negative']['mean']:+.6f} "
            f"delta={result['best_negative']['improvement_over_next']:+.6f} | "
            f"best positive alpha={result['best_positive']['alpha']:+g} "
            f"mean={result['best_positive']['mean']:+.6f} "
            f"delta={result['best_positive']['improvement_over_next']:+.6f}",
            flush=True,
        )
        print(
            f"    global alpha={selection['global']['alpha']:+g} "
            f"mean={selection['global']['mean_spearman']:+.6f} | "
            f"per-query oracle mean="
            f"{selection['per_query_oracle']['mean_spearman']:+.6f}",
            flush=True,
        )
    print(f"[saved] {json_path}", flush=True)
    print(f"[saved] {csv_path}", flush=True)
    print(f"[saved] {selection_csv_path}", flush=True)


if __name__ == "__main__":
    main()
