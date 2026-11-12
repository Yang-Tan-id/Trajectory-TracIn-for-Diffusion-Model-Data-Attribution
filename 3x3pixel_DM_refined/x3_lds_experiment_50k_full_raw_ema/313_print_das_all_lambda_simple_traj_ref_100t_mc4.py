#!/usr/bin/env python3
"""Evaluate the original x3 DAS lambda sweep on simple-loss and trajectory targets.

The simple-loss targets are loaded explicitly from the new 100-timestamp x MC4
bank, so this report cannot silently fall back to the old 10-timestamp bank.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from exp_config import ATTR_DIR, DAS_LAMBDAS, LDS_DIR, MASK_DIR


METHODS = ("das_ema", "das_raw")
METRIC_PATHS = {
    "simple_loss_ema_100t_mc4": LDS_DIR / "observed_simple_loss_ema_100t_mc4.npy",
    "simple_loss_raw_100t_mc4": LDS_DIR / "observed_simple_loss_raw_100t_mc4.npy",
    "traj_ref_ema": LDS_DIR / "observed_traj_ref_ema.npy",
    "traj_ref_raw": LDS_DIR / "observed_traj_ref_raw.npy",
}


def lambda_tag(value: float) -> str:
    return str(float(value)).replace(".", "p")


def parse_query_ids(text: str) -> list[int]:
    query_ids: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            query_ids.extend(range(start, end + 1))
        else:
            query_ids.append(int(part))
    if not query_ids:
        raise ValueError("--query-ids selected no queries")
    return query_ids


def load_scores(method: str, lam: float, query_ids: list[int]) -> np.ndarray:
    paths = [
        ATTR_DIR
        / method
        / f"q{query_id:02d}"
        / f"lambda_{lambda_tag(lam)}"
        / "scores.npy"
        for query_id in query_ids
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(
            f"missing {len(missing)} score files for {method}, lambda={lam:g}:\n{preview}"
        )
    return np.stack([np.load(path) for path in paths]).astype(np.float64)


def evaluate(
    predictions: np.ndarray,
    observed: np.ndarray,
    query_ids: list[int],
) -> dict:
    # Match 06_lds_eval.py exactly: pred = -(membership @ attribution).
    values = np.asarray(
        [
            spearmanr(-predictions[:, position], observed[query_id]).statistic
            for position, query_id in enumerate(query_ids)
        ],
        dtype=np.float64,
    )
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "per_query": {
            f"q{query_id:02d}": float(values[position])
            for position, query_id in enumerate(query_ids)
        },
    }


def render_report(payload: dict) -> str:
    metrics = list(METRIC_PATHS)
    lines = [
        "X3 ORIGINAL DAS — ALL LAMBDAS",
        "simple loss = explicit 100 timestamps x MC4 observed bank",
        "traj_ref = existing 100-snapshot trajectory target",
        "LDS convention = Spearman(-membership @ score, observed); sign=-1",
        f"queries = {len(payload['query_ids'])}",
        "",
    ]
    for method in METHODS:
        lines.extend(
            [
                f"METHOD: {method}",
                (
                    f"{'lambda':>9s}  "
                    + "  ".join(f"{metric:>34s}" for metric in metrics)
                ),
                "-" * (11 + 38 * len(metrics)),
            ]
        )
        for lam in payload["lambdas"]:
            row = payload["results"][method][str(lam)]
            values = "  ".join(
                f"{row[metric]['mean']:+.6f} +/- {row[metric]['std']:.6f}"
                for metric in metrics
            )
            lines.append(f"{lam:9g}  {values}")

        lines.extend(["", "BEST LAMBDA BY MEAN LDS"])
        for metric in metrics:
            best_lam = max(
                payload["lambdas"],
                key=lambda lam: payload["results"][method][str(lam)][metric]["mean"],
            )
            best = payload["results"][method][str(best_lam)][metric]
            lines.append(
                f"  {metric:34s} lambda={best_lam:g} "
                f"{best['mean']:+.6f} +/- {best['std']:.6f}"
            )
        lines.append("")

    lines.append("PER-QUERY LDS")
    for method in METHODS:
        for lam in payload["lambdas"]:
            lines.append(f"\nMETHOD={method} lambda={lam:g}")
            row = payload["results"][method][str(lam)]
            lines.append(
                "query  " + "  ".join(f"{metric:>34s}" for metric in metrics)
            )
            for query_id in payload["query_ids"]:
                query = f"q{query_id:02d}"
                lines.append(
                    f"{query:5s}  "
                    + "  ".join(
                        f"{row[metric]['per_query'][query]:+34.6f}"
                        for metric in metrics
                    )
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-ids", default="0-99")
    parser.add_argument(
        "--output-prefix",
        default=str(LDS_DIR / "das_all_lambda_simple100t_mc4_traj_ref_q00_q99"),
    )
    args = parser.parse_args()

    query_ids = parse_query_ids(args.query_ids)
    membership_path = MASK_DIR / "membership.npy"
    if not membership_path.is_file():
        raise FileNotFoundError(membership_path)
    membership = np.load(membership_path).astype(np.float64)

    observed = {}
    for metric, path in METRIC_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[metric] = np.load(path).astype(np.float64)
        if observed[metric].shape[1] != membership.shape[0]:
            raise ValueError(
                f"mask mismatch for {metric}: observed={observed[metric].shape}, "
                f"membership={membership.shape}"
            )

    payload = {
        "query_ids": query_ids,
        "lambdas": [float(value) for value in DAS_LAMBDAS],
        "prediction_sign": -1,
        "simple_loss_bank": {"timestamps": 100, "mc": 4},
        "metric_paths": {key: str(value) for key, value in METRIC_PATHS.items()},
        "results": {method: {} for method in METHODS},
    }

    for method in METHODS:
        for lam_raw in DAS_LAMBDAS:
            lam = float(lam_raw)
            scores = load_scores(method, lam, query_ids)
            predictions = membership @ scores.T
            payload["results"][method][str(lam)] = {
                metric: evaluate(predictions, values, query_ids)
                for metric, values in observed.items()
            }
            summary = payload["results"][method][str(lam)]
            print(
                f"[{method} lambda={lam:g}] "
                + " ".join(
                    f"{metric}={summary[metric]['mean']:+.6f}"
                    for metric in METRIC_PATHS
                ),
                flush=True,
            )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    text_path = output_prefix.with_suffix(".txt")
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    text_path.write_text(render_report(payload))
    print(f"[saved] {json_path}")
    print(f"[saved] {text_path}")


if __name__ == "__main__":
    main()
