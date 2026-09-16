#!/usr/bin/env python3
"""Compare flipped saved query pullbacks for one old/fresh probe pair."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT / "script") not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT / "script"))

from analyze_predicted_noise_old_fresh_term_signs import BANKS, write_csv
from run_predicted_noise_jvp_l2_squared import query_artifact_path


def load_query_pullbacks(args, bank: str, global_probe: int):
    probe_in_bank = global_probe if bank == "old12" else global_probe - 12
    path = query_artifact_path(
        args.experiment,
        args.train_seed,
        args.epochs,
        args.query_id,
        num_probes=12,
        probe_index=probe_in_bank - 1,
        query_namespace_pattern=BANKS[bank]["query_pattern"],
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        features = np.asarray(payload["query_features"], dtype=np.float64)
        metadata = {
            "ckpt_indices": np.asarray(payload["ckpt_indices"], dtype=np.int32),
            "timesteps": np.asarray(payload["timesteps"], dtype=np.int32),
            "snapshot_positions": np.asarray(
                payload["snapshot_positions"], dtype=np.int32
            ),
        }
    if features.ndim != 2:
        raise ValueError(f"expected query_features matrix, got {features.shape}: {path}")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    units = features / np.maximum(norms, 1e-12)
    return units, norms[:, 0], metadata, path


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "mean_abs": float(np.abs(values).mean()),
        "positive_fraction": float(np.mean(values > 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--source-run-id", default="3506389")
    parser.add_argument("--query-id", type=int, default=0)
    parser.add_argument("--old-global-probe", type=int, default=8)
    parser.add_argument("--fresh-global-probe", type=int, default=20)
    parser.add_argument("--output-tag", default="p8_p20")
    args = parser.parse_args()

    if not 1 <= args.old_global_probe <= 12:
        raise ValueError("old-global-probe must be in 1,...,12")
    if not 13 <= args.fresh_global_probe <= 24:
        raise ValueError("fresh-global-probe must be in 13,...,24")

    old, old_norms, old_meta, old_path = load_query_pullbacks(
        args, "old12", args.old_global_probe
    )
    fresh, fresh_norms, fresh_meta, fresh_path = load_query_pullbacks(
        args, "fresh12", args.fresh_global_probe
    )
    for key, expected in old_meta.items():
        if not np.array_equal(fresh_meta[key], expected):
            raise ValueError(f"old/fresh metadata differ for {key}")
    if old.shape != fresh.shape:
        raise ValueError(f"old/fresh query feature shapes differ: {old.shape}, {fresh.shape}")
    if old.shape[0] != 500:
        raise ValueError(f"expected 500 terms, got {old.shape[0]}")

    result_root = SHAPES_ROOT / "result" / args.experiment
    sign_path = (
        result_root
        / "eval"
        / "predicted_noise_24_individual_probe_timestamp_signs"
        / f"source_run_{args.source_run_id}"
        / "individual_probe_best_binary_vectors.npz"
    )
    if not sign_path.is_file():
        raise FileNotFoundError(sign_path)
    with np.load(sign_path, allow_pickle=False) as payload:
        query_ids = np.asarray(payload["query_ids"], dtype=np.int32)
        sign_timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        best_signs = np.asarray(payload["best_signs"], dtype=np.int8)
    qslots = np.flatnonzero(query_ids == args.query_id)
    if len(qslots) != 1:
        raise ValueError(f"query {args.query_id} not uniquely present in sign cache")
    qslot = int(qslots[0])
    timestep_slots = {int(value): slot for slot, value in enumerate(sign_timesteps)}
    old_signs = best_signs[args.old_global_probe - 1, qslot]
    fresh_signs = best_signs[args.fresh_global_probe - 1, qslot]
    term_slots = np.asarray(
        [timestep_slots[int(value)] for value in old_meta["timesteps"]],
        dtype=np.int64,
    )
    old_flipped = old * old_signs[term_slots, None]
    fresh_flipped = fresh * fresh_signs[term_slots, None]

    checkpoints = list(dict.fromkeys(int(x) for x in old_meta["ckpt_indices"]))
    if len(checkpoints) != 50:
        raise ValueError(f"expected 50 checkpoints, got {len(checkpoints)}")
    train_checkpoints = set(checkpoints[:25])
    train_mask = np.asarray(
        [int(x) in train_checkpoints for x in old_meta["ckpt_indices"]], dtype=bool
    )
    heldout_mask = ~train_mask
    anchor_sum = fresh_flipped[train_mask].sum(axis=0)
    anchor_resultant = float(np.linalg.norm(anchor_sum) / train_mask.sum())
    anchor = anchor_sum / max(float(np.linalg.norm(anchor_sum)), 1e-12)

    pair_cosine = np.einsum("td,td->t", fresh_flipped, old_flipped)
    fresh_anchor = fresh_flipped @ anchor
    old_anchor = old_flipped @ anchor

    checkpoint_slots = {value: slot for slot, value in enumerate(checkpoints)}
    rows = []
    for term, (checkpoint, timestep, position) in enumerate(
        zip(
            old_meta["ckpt_indices"],
            old_meta["timesteps"],
            old_meta["snapshot_positions"],
        )
    ):
        tslot = timestep_slots[int(timestep)]
        rows.append(
            {
                "term": term,
                "checkpoint_ordinal": checkpoint_slots[int(checkpoint)] + 1,
                "checkpoint_index": int(checkpoint),
                "timestep": int(timestep),
                "snapshot_position": int(position),
                "split": "anchor_train" if train_mask[term] else "heldout_test",
                "fresh_selected_sign": int(fresh_signs[tslot]),
                "old_selected_sign": int(old_signs[tslot]),
                "fresh_cosine_to_anchor": float(fresh_anchor[term]),
                "old_cosine_to_anchor": float(old_anchor[term]),
                "fresh_old_same_term_cosine": float(pair_cosine[term]),
                "fresh_query_feature_norm": float(fresh_norms[term]),
                "old_query_feature_norm": float(old_norms[term]),
            }
        )

    split_rows = []
    for split, mask in (("anchor_train", train_mask), ("heldout_test", heldout_mask)):
        for label, values in (
            (f"p{args.fresh_global_probe}", fresh_anchor),
            (f"p{args.old_global_probe}", old_anchor),
        ):
            split_rows.append(
                {
                    "split": split,
                    "probe": label,
                    "terms": int(mask.sum()),
                    **{f"cosine_to_anchor_{k}": v for k, v in summarize(values[mask]).items()},
                }
            )

    timestamp_rows = []
    for timestep in sorted(timestep_slots, reverse=True):
        mask = old_meta["timesteps"] == timestep
        heldout = mask & heldout_mask
        timestamp_rows.append(
            {
                "timestep": timestep,
                "terms": int(mask.sum()),
                **{f"pair_{k}": v for k, v in summarize(pair_cosine[mask]).items()},
                **{f"fresh_anchor_{k}": v for k, v in summarize(fresh_anchor[heldout]).items()},
                **{f"old_anchor_{k}": v for k, v in summarize(old_anchor[heldout]).items()},
            }
        )

    out_dir = (
        result_root
        / "eval"
        / "q0_probe_pair_projected_directions"
        / f"source_run_{args.source_run_id}"
        / args.output_tag
    )
    write_csv(out_dir / "per_term_cosines.csv", rows)
    write_csv(out_dir / "split_summary.csv", split_rows)
    write_csv(out_dir / "per_timestamp_summary.csv", timestamp_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "query": args.query_id,
        "old_global_probe": args.old_global_probe,
        "fresh_global_probe": args.fresh_global_probe,
        "old_artifact": str(old_path),
        "fresh_artifact": str(fresh_path),
        "projected_dimension": int(old.shape[1]),
        "anchor_resultant": anchor_resultant,
        "random_anchor_resultant_baseline": 1.0 / math.sqrt(int(train_mask.sum())),
        "isotropic_random_cosine_sd": 1.0 / math.sqrt(int(old.shape[1])),
        "heldout_pair": summarize(pair_cosine[heldout_mask]),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Q0 FLIPPED PROJECTED QUERY-PULLBACK DIRECTIONS")
    print(
        f"P{args.old_global_probe} vs P{args.fresh_global_probe} | "
        f"dimension={old.shape[1]} | anchor=P{args.fresh_global_probe}, checkpoints 1-25"
    )
    print(
        f"anchor resultant={anchor_resultant:.6f} | "
        f"random baseline={manifest['random_anchor_resultant_baseline']:.6f} | "
        f"isotropic cosine SD={manifest['isotropic_random_cosine_sd']:.6f}"
    )
    print(f"{'SPLIT':13s} {'PROBE':8s} {'MEAN COS':>10s} {'STD':>10s} {'|COS|':>10s} {'COS+':>8s}")
    print("-" * 68)
    for row in split_rows:
        print(
            f"{row['split']:13s} {row['probe']:8s} "
            f"{row['cosine_to_anchor_mean']:+10.6f} "
            f"{row['cosine_to_anchor_std']:10.6f} "
            f"{row['cosine_to_anchor_mean_abs']:10.6f} "
            f"{row['cosine_to_anchor_positive_fraction']:8.3f}"
        )
    heldout = manifest["heldout_pair"]
    print(
        f"heldout same-term P{args.fresh_global_probe}/P{args.old_global_probe}: "
        f"{heldout['mean']:+.6f} ± {heldout['std']:.6f}; "
        f"|cos|={heldout['mean_abs']:.6f}; cos+={heldout['positive_fraction']:.3f}"
    )
    print("\nPER TIMESTAMP — PAIR COSINE ACROSS 50 CHECKPOINTS")
    print(f"{'T':>4s} {'MEAN':>10s} {'STD':>10s} {'|COS|':>10s} {'COS+':>8s} {'P20→A':>10s} {'P8→A':>10s}")
    print("-" * 80)
    for row in timestamp_rows:
        print(
            f"{row['timestep']:4d} {row['pair_mean']:+10.6f} "
            f"{row['pair_std']:10.6f} {row['pair_mean_abs']:10.6f} "
            f"{row['pair_positive_fraction']:8.3f} "
            f"{row['fresh_anchor_mean']:+10.6f} {row['old_anchor_mean']:+10.6f}"
        )
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
