#!/usr/bin/env python3
"""Average two timestamp-shared probe scores, then compute LDS."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
for root in (SHAPES_ROOT, REFINE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from analyze_predicted_noise_probe8_choose4 import (  # noqa: E402
    TARGETS,
    cache_group,
    load_target_data,
    spearman,
    write_csv,
)
from dataset_config import _prompt_tag  # noqa: E402
from legacy_jax.LDS.DM_cifar_lds import (  # noqa: E402
    resolve_score_inputs,
)
from legacy_jax.DM_counterfactual_retrain_from_attribution import (  # noqa: E402
    combine_attribution_scores,
)


VARIANTS = {
    "raw": "score",
    "query_l2": "score_query_normalized",
    "train_l2": "score_train_l2_normalized",
    "query_train_l2": "score_query_train_l2_normalized",
}
REDUCTIONS = {
    "linear": "predicted_noise_jvp_signed",
    "square": "predicted_noise_jvp_l2_squared",
    "root": "predicted_noise_jvp_probe_l2",
}


def load_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    inputs = resolve_score_inputs(str(path))
    indices, scores, _ = combine_attribution_scores(
        inputs, duplicate_policy="max"
    )
    order = np.argsort(indices)
    return (
        np.asarray(indices, dtype=np.int64)[order],
        np.asarray(scores, dtype=np.float64)[order],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument(
        "--suffix-a", default="timestamp_shared_own_trajectory"
    )
    parser.add_argument(
        "--suffix-b", default="timestamp_shared_seed20260918_own_trajectory"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    result_root = SHAPES_ROOT / "result" / args.experiment
    records = json.loads(
        (SHAPES_ROOT / "queries_seed_0_9.json").read_text()
    )["queries"]
    rows: list[dict[str, object]] = []

    for query, record in enumerate(records):
        prompt_tag = _prompt_tag(str(record["prompt"]))
        initial_seed = int(record["initial_seed"])
        score_root = (
            result_root
            / "attribution_score"
            / "prompted_solo"
            / f"train_seed_{args.train_seed}"
            / f"query_{prompt_tag}"
            / f"initial_seed_{initial_seed}"
        )
        eval_root = (
            result_root
            / "eval"
            / "prompted_solo"
            / f"query_{prompt_tag}"
            / f"initial_seed_{initial_seed}"
        )
        target_indices = None
        incidence = None
        true_values = None

        for reduction, base in REDUCTIONS.items():
            namespaces = (
                f"traj_tracin_{base}_{args.suffix_a}",
                f"traj_tracin_{base}_{args.suffix_b}",
            )
            for variant, score_dir_name in VARIANTS.items():
                indices_a, scores_a = load_scores(
                    score_root / namespaces[0] / score_dir_name
                )
                indices_b, scores_b = load_scores(
                    score_root / namespaces[1] / score_dir_name
                )
                if not np.array_equal(indices_a, indices_b):
                    raise ValueError(
                        f"Q{query} {reduction} {variant}: score indices differ"
                    )
                averaged_scores = 0.5 * (scores_a + scores_b)
                if target_indices is None:
                    target_indices = indices_a
                    incidence, true_values = load_target_data(
                        cache_group(eval_root), indices_a
                    )
                elif not np.array_equal(target_indices, indices_a):
                    raise ValueError(f"Q{query}: score indices differ across reductions")
                assert incidence is not None and true_values is not None
                native_prediction = averaged_scores @ incidence.T
                for sign_name, multiplier in (("p1", 1.0), ("m1", -1.0)):
                    for target in TARGETS:
                        rows.append(
                            {
                                "query": query,
                                "reduction": reduction,
                                "variant": variant,
                                "sign": sign_name,
                                "target": target,
                                "lds_percent": 100.0
                                * spearman(
                                    multiplier * native_prediction,
                                    true_values[target],
                                ),
                            }
                        )
        print(f"[query {query}/9] averaged two probe scores", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_query.csv", rows)
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["reduction"]),
                str(row["sign"]),
                str(row["target"]),
                str(row["variant"]),
            )
        ].append(float(row["lds_percent"]))
    mean_rows = [
        {
            "reduction": reduction,
            "sign": sign,
            "target": target,
            "variant": variant,
            "mean_lds_percent": float(np.mean(values)),
        }
        for (reduction, sign, target, variant), values in grouped.items()
    ]
    write_csv(args.out_dir / "ten_query_means.csv", mean_rows)

    lookup = {
        (row["reduction"], row["sign"], row["target"], row["variant"]): row[
            "mean_lds_percent"
        ]
        for row in mean_rows
    }
    print("\nTWO TIMESTAMP-SHARED PROBES — SCORE AVERAGE THEN LDS")
    for reduction in REDUCTIONS:
        for sign in ("p1", "m1"):
            print(f"\n{reduction.upper()} sign={sign}")
            print(
                f"{'TARGET':24s} {'RAW':>9s} {'QUERY-L2':>9s} "
                f"{'TRAIN-L2':>9s} {'BOTH-L2':>9s}"
            )
            print("-" * 68)
            for target in TARGETS:
                values = [
                    lookup[(reduction, sign, target, variant)]
                    for variant in VARIANTS
                ]
                print(
                    f"{target:24s} "
                    + " ".join(f"{value:8.3f}%" for value in values)
                )
    print(f"\n[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
