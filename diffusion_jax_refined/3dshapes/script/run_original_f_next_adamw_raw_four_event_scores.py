#!/usr/bin/env python3
"""Original-f/next linear scores comparing AdamW and raw event gradients."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "legacy_jax")]

from analyze_predicted_noise_probe12_sign_flips import rowwise_spearman
from analyze_predicted_noise_probe8_choose4 import cache_group, load_target_data
from dataset_config import _prompt_tag
from run_adamw_four_event_original_f_scores import event
from run_expected_residual_jacobian_scores import load_query_bank


METHODS = ("adamw_four", "adamw_e1", "raw_four", "raw_e1")
VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")
TARGETS = (
    "endpoint_contarfactual",
    "traj_contarfactual",
    "simple_loss",
    "noise_trajectory",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--attribution-points", type=int, default=5000)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    result_root = ROOT / "result" / args.experiment
    adamw_root = result_root / f"fixed_checkpoint_adamw_four_events_n{args.attribution_points}"
    raw_root = result_root / f"fixed_checkpoint_raw_four_events_n{args.attribution_points}"

    class QueryArgs:
        pass

    query_args = QueryArgs()
    query_args.experiment = args.experiment
    query_args.train_seed = args.train_seed
    query_args.epochs = args.epochs
    query, metadata = load_query_bank(
        query_args,
        "loss_direction_residual_rms_original_f",
        "trajectory_next_checkpoint_noise_mse",
        range(10),
    )
    checkpoint_indices = np.asarray(metadata["ckpt_indices"], dtype=np.int32)
    timesteps = np.asarray(metadata["timesteps"], dtype=np.int32)
    term_weights = np.asarray(metadata["term_weights"], dtype=np.float64)
    scores = {
        (method, variant): np.zeros((10, args.attribution_points), dtype=np.float64)
        for method in METHODS
        for variant in VARIANTS
    }
    score_indices = None

    for checkpoint in range(49):
        start_epoch = 4 * (checkpoint + 1)
        roots = {
            "adamw": adamw_root / f"epoch_{start_epoch}_{start_epoch + 4}",
            "raw": raw_root / f"epoch_{start_epoch}_{start_epoch + 4}",
        }
        event_banks = {}
        for family, source in roots.items():
            values = []
            for epoch in range(start_epoch + 1, start_epoch + 5):
                features, indices = event(source, epoch)
                if score_indices is None:
                    score_indices = indices
                elif not np.array_equal(score_indices, indices):
                    raise ValueError(
                        f"score-index mismatch: checkpoint={checkpoint} "
                        f"family={family} epoch={epoch}"
                    )
                values.append(features)
            event_banks[f"{family}_four"] = sum(values)
            event_banks[f"{family}_e1"] = values[0]

        term_ids = np.flatnonzero(checkpoint_indices == checkpoint)
        if len(term_ids) != 10:
            raise ValueError(
                f"checkpoint {checkpoint} expected 10 query terms, got {len(term_ids)}"
            )
        checkpoint_query = np.stack(
            [query[:, term_id, :] for term_id in term_ids], axis=0
        )
        # [timestamp, query, projection] -> [timestamp * query, projection]
        query_matrix = checkpoint_query.reshape(100, checkpoint_query.shape[-1])
        query_device = jax.device_put(jnp.asarray(query_matrix, dtype=jnp.float32))
        query_norm = jnp.linalg.norm(query_device, axis=1) + 1e-8
        weights = jnp.asarray(term_weights[term_ids], dtype=jnp.float64)

        for method, bank in event_banks.items():
            train_device = jax.device_put(jnp.asarray(bank, dtype=jnp.float32))
            train_norm = jnp.linalg.norm(train_device, axis=1) + 1e-8
            dots = train_device @ query_device.T
            variants = {
                "raw": dots,
                "query_l2": dots / query_norm[None, :],
                "train_l2": dots / train_norm[:, None],
                "query_train_l2": dots / train_norm[:, None] / query_norm[None, :],
            }
            for variant, values in variants.items():
                by_term = values.reshape(args.attribution_points, 10, 10)
                combined = jnp.sum(by_term * weights[None, :, None], axis=1)
                scores[(method, variant)] += np.asarray(combined, dtype=np.float64).T
        print(f"[score] checkpoint={checkpoint + 1}/49", flush=True)

    assert score_indices is not None
    records = json.loads((ROOT / "queries_seed_0_9.json").read_text())["queries"]
    rows = []
    for query_id, record in enumerate(records):
        eval_root = (
            result_root / "eval" / "prompted_solo"
            / f"query_{_prompt_tag(str(record['prompt']))}"
            / f"initial_seed_{int(record['initial_seed'])}"
        )
        incidence, true_values = load_target_data(cache_group(eval_root), score_indices)
        for method in METHODS:
            for variant in VARIANTS:
                predictions = scores[(method, variant)][query_id] @ incidence.T
                for target in TARGETS:
                    lds = 100.0 * float(
                        rowwise_spearman(predictions[None, :], true_values[target])[0]
                    )
                    rows.append(
                        {
                            "method": method,
                            "variant": variant,
                            "query": query_id,
                            "target": target,
                            "lds_percent": lds,
                            "prediction_sign": "p1",
                        }
                    )

    write_csv(args.out_dir / "per_query.csv", rows)
    print("ORIGINAL-F NEXT — ADAMW vs RAW — LINEAR FIXED P1")
    for method in METHODS:
        for variant in VARIANTS:
            values = []
            print(f"\n{method.upper()} — {variant.upper()}")
            for query_id in range(10):
                row_values = [
                    next(
                        row["lds_percent"]
                        for row in rows
                        if row["method"] == method
                        and row["variant"] == variant
                        and row["query"] == query_id
                        and row["target"] == target
                    )
                    for target in TARGETS
                ]
                values.append(row_values)
                print(f"Q{query_id} " + " ".join(f"{x:+8.3f}%" for x in row_values))
            print("MEAN " + " ".join(f"{x:+8.3f}%" for x in np.mean(values, axis=0)))
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
