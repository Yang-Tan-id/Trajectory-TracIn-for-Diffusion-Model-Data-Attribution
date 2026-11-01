#!/usr/bin/env python3
"""Evaluate only the ten five-method Shape3D configs on 100x1 simple loss."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
REPO_ROOT = REFINE_ROOT.parent
for path in (SHAPES_ROOT, REFINE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_config import _prompt_tag
from diffusion_jax_refined.script.export_pint_lds_txt import method_configs


VARIANT_SCORE_DIR = {
    "raw": "score",
    "query_train_l2": "score_query_train_l2_normalized",
}


def parse_ids(text: str) -> list[int]:
    values = []
    for token in text.replace(",", " ").split():
        if "-" in token:
            left, right = token.split("-", 1)
            values.extend(range(int(left), int(right) + 1))
        else:
            values.append(int(token))
    return values


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument(
        "--query-file",
        type=Path,
        default=SHAPES_ROOT / "queries_in_distribution_plus_zero_seed_100_219.json",
    )
    parser.add_argument("--query-ids", default="0-99")
    parser.add_argument(
        "--cache-namespace", default="simple_loss_100x1_shared_seed0"
    )
    parser.add_argument("--expected-models", type=int, default=192)
    parser.add_argument("--expected-probes", type=int, default=100)
    parser.add_argument("--target-name", default="simple_loss_100x1")
    parser.add_argument(
        "--transforms",
        choices=("both", "raw-gradient", "adamw"),
        default="both",
        help="Evaluate both score transforms or only the selected transform.",
    )
    args = parser.parse_args()

    legacy_root = REFINE_ROOT / "legacy_jax"
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))
    from LDS.DM_cifar_lds import (
        build_score_vector,
        combine_attribution_scores,
        resolve_score_inputs,
        spearman_corr,
        sum_scores,
    )

    records = json.loads(args.query_file.read_text())["queries"]
    query_ids = parse_ids(args.query_ids)
    methods = method_configs("shape3d")
    result_root = SHAPES_ROOT / "result" / args.experiment
    eval_root = result_root / "eval" / "prompted_solo"
    score_base = (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{args.train_seed}"
    )

    completed = 0
    selected_transforms = {
        "both": ("raw_gradient", "adamw_full"),
        "raw-gradient": ("raw_gradient",),
        "adamw": ("adamw_full",),
    }[args.transforms]
    for query_id in query_ids:
        record = records[query_id]
        prompt = str(record["prompt"])
        seed = int(record["initial_seed"])
        tag = _prompt_tag(prompt)
        query_eval = eval_root / f"query_{tag}" / f"initial_seed_{seed}"
        cache_parent = (
            query_eval
            / "lds_target_cache"
            / "ddim_eta0"
            / args.cache_namespace
        )
        groups = sorted(path for path in cache_parent.iterdir() if path.is_dir())
        if len(groups) != 1:
            raise RuntimeError(
                f"Q{query_id}: expected one cache model group under "
                f"{cache_parent}; found {groups}"
            )
        group = groups[0]
        targets = []
        for model_id in range(args.expected_models):
            path = group / "simple_loss" / f"target_{model_id:04d}.json"
            payload = json.loads(path.read_text())
            details = payload.get("target_details", {})
            if int(details.get("simple_loss_num_mc_terms", -1)) != args.expected_probes:
                raise ValueError(
                    f"{path}: target does not contain {args.expected_probes} probes"
                )
            subset_dir = Path(payload["subset_dir"])
            targets.append(
                {
                    "subset_id": model_id,
                    "true_f": float(payload["true_f"]),
                    "subset_dir": subset_dir,
                    "kept": np.load(subset_dir / "kept_attribution_indices.npy"),
                    "checkpoint": payload.get("checkpoint", ""),
                }
            )
        true_f = np.asarray([row["true_f"] for row in targets], dtype=np.float64)
        query_scores = score_base / f"query_{tag}" / f"initial_seed_{seed}"

        for method in methods:
            for transform, namespace, preferred_sign in (
                ("raw_gradient", method.raw_namespace, method.raw_sign),
                ("adamw_full", method.adamw_namespace, method.adamw_sign),
            ):
                if transform not in selected_transforms:
                    continue
                score_dir = query_scores / namespace / VARIANT_SCORE_DIR[method.variant]
                inputs = resolve_score_inputs(str(score_dir))
                indices, scores, sources = combine_attribution_scores(
                    inputs, duplicate_policy="max"
                )
                score_map = build_score_vector(indices, scores)
                sign = 1.0 if preferred_sign == "p1" else -1.0
                predictions = np.asarray(
                    [sum_scores(row["kept"], score_map, sign) for row in targets],
                    dtype=np.float64,
                )
                lds = spearman_corr(predictions, true_f)
                rows = [
                    {
                        "subset_id": row["subset_id"],
                        "true_f": row["true_f"],
                        "pred_sum_tau": float(prediction),
                        "prediction_subset": "kept",
                        "prediction_sign": sign,
                        "checkpoint": row["checkpoint"],
                        "subset_dir": str(row["subset_dir"]),
                    }
                    for row, prediction in zip(targets, predictions)
                ]
                out_dir = (
                    query_eval
                    / "lds"
                    / f"{namespace}_{method.variant}"
                    / args.target_name
                    / f"pred_kept_sign_{preferred_sign}"
                    / group.name
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                write_csv(out_dir / "lds_results.csv", rows)
                atomic_json(
                    out_dir / "lds_summary.json",
                    {
                        "algorithm": f"{namespace}_{method.variant}",
                        "method_label": method.label,
                        "transform": transform,
                        "target_function": args.target_name,
                        "target_definition": (
                            f"mean simple denoising loss over {args.expected_probes} uniformly "
                            "spaced timesteps with one independent shared probe per timestep"
                        ),
                        "target_cache": str(group / "simple_loss"),
                        "score_sources": sources,
                        "num_models": len(rows),
                        "lds_spearman": float(lds),
                        "lds_percent": float(100.0 * lds),
                        "prediction_subset": "kept",
                        "prediction_sign": sign,
                    },
                )
                completed += 1
                print(
                    f"[LDS {completed}/{len(query_ids) * len(methods) * len(selected_transforms)}] "
                    f"Q{query_id} "
                    f"{transform} {method.label}={100.0 * lds:+.3f}%",
                    flush=True,
                )
    print(f"[done] selected five-method LDS records={completed}", flush=True)


if __name__ == "__main__":
    main()
