#!/usr/bin/env python3
"""Compare flipped raw output-probe directions for Q0 near-positive/negative probes."""

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
from analyze_predicted_noise_probe_independence import predicted_noise_probe_key
from run_predicted_noise_jvp_l2_squared import query_artifact_path


def parse_shape(text: str) -> tuple[int, ...]:
    shape = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not shape or any(value <= 0 for value in shape):
        raise argparse.ArgumentTypeError(f"invalid shape: {text!r}")
    return shape


def load_metadata(args, bank: str, global_probe: int):
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
        metadata = {
            "ckpt_indices": np.asarray(payload["ckpt_indices"], dtype=np.int32),
            "timesteps": np.asarray(payload["timesteps"], dtype=np.int32),
            "snapshot_positions": np.asarray(
                payload["snapshot_positions"], dtype=np.int32
            ),
        }
    return metadata, path


def save_heatmap(path: Path, matrix: np.ndarray, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib unavailable; heatmap skipped", flush=True)
        return
    limit = max(0.01, float(np.max(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    ax.axhline(24.5, color="black", linewidth=1.5)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(0, matrix.shape[0], 5), range(1, matrix.shape[0] + 1, 5))
    ax.set_xlabel("timestamp slot")
    ax.set_ylabel("checkpoint ordinal")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="cosine to P20 train-half anchor")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--source-run-id", default="3506389")
    parser.add_argument("--query-id", type=int, default=0)
    parser.add_argument("--negative-global-probe", type=int, default=1)
    parser.add_argument("--positive-global-probe", type=int, default=20)
    parser.add_argument("--fresh-probe-seed", type=int, default=20260915)
    parser.add_argument("--output-shape", type=parse_shape, default=(1, 64, 64, 3))
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    if args.query_id != 0:
        raise ValueError("this focused audit currently expects query 0")
    if not 1 <= args.negative_global_probe <= 12:
        raise ValueError("negative probe must be in old12 for this Q0 audit")
    if not 13 <= args.positive_global_probe <= 24:
        raise ValueError("positive probe must be in fresh12 for this Q0 audit")

    negative_meta, negative_artifact = load_metadata(
        args, "old12", args.negative_global_probe
    )
    positive_meta, positive_artifact = load_metadata(
        args, "fresh12", args.positive_global_probe
    )
    for key, reference in negative_meta.items():
        if not np.array_equal(positive_meta[key], reference):
            raise ValueError(f"old/fresh metadata differ for {key}")
    metadata = negative_meta
    term_count = len(metadata["ckpt_indices"])
    if term_count != 500:
        raise ValueError(f"expected 500 terms, got {term_count}")

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
        timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        best_signs = np.asarray(payload["best_signs"], dtype=np.int8)
    qslots = np.flatnonzero(query_ids == args.query_id)
    if len(qslots) != 1:
        raise ValueError(f"query {args.query_id} not uniquely present in sign cache")
    qslot = int(qslots[0])
    timestep_slots = {int(value): slot for slot, value in enumerate(timesteps)}
    negative_signs = best_signs[args.negative_global_probe - 1, qslot]
    positive_signs = best_signs[args.positive_global_probe - 1, qslot]

    generate_pair = jax.jit(
        jax.vmap(
            lambda key: jax.random.normal(key, args.output_shape, dtype=jnp.float32)
        )
    )
    dimension = int(np.prod(args.output_shape))
    negative_units = np.empty((term_count, dimension), dtype=np.float32)
    positive_units = np.empty_like(negative_units)
    negative_probe_index = args.negative_global_probe - 1
    positive_probe_index = args.positive_global_probe - 13
    for term, (checkpoint, timestep, position) in enumerate(
        zip(
            metadata["ckpt_indices"],
            metadata["timesteps"],
            metadata["snapshot_positions"],
        )
    ):
        keys = jnp.stack(
            [
                predicted_noise_probe_key(
                    args.train_seed,
                    int(checkpoint),
                    int(timestep),
                    int(position),
                    negative_probe_index,
                ),
                predicted_noise_probe_key(
                    args.fresh_probe_seed,
                    int(checkpoint),
                    int(timestep),
                    int(position),
                    positive_probe_index,
                ),
            ]
        )
        pair = np.asarray(jax.device_get(generate_pair(keys)), dtype=np.float32).reshape(
            2, dimension
        )
        pair /= np.maximum(np.linalg.norm(pair, axis=1, keepdims=True), 1e-12)
        tslot = timestep_slots[int(timestep)]
        negative_units[term] = float(negative_signs[tslot]) * pair[0]
        positive_units[term] = float(positive_signs[tslot]) * pair[1]
        if (term + 1) % 50 == 0:
            print(f"[output directions] terms={term + 1}/{term_count}", flush=True)

    checkpoint_values = list(dict.fromkeys(int(x) for x in metadata["ckpt_indices"]))
    if len(checkpoint_values) != 50:
        raise ValueError(f"expected 50 checkpoints, got {len(checkpoint_values)}")
    train_checkpoints = set(checkpoint_values[:25])
    train_mask = np.asarray(
        [int(value) in train_checkpoints for value in metadata["ckpt_indices"]],
        dtype=bool,
    )
    heldout_mask = ~train_mask
    anchor_sum = positive_units[train_mask].sum(axis=0, dtype=np.float64)
    anchor_resultant = float(np.linalg.norm(anchor_sum) / np.count_nonzero(train_mask))
    anchor = anchor_sum / max(np.linalg.norm(anchor_sum), 1e-12)

    positive_cos = positive_units @ anchor
    negative_cos = negative_units @ anchor
    pair_cos = np.sum(positive_units * negative_units, axis=1)
    dc_anchor = np.ones(dimension, dtype=np.float64) / math.sqrt(dimension)
    positive_dc = positive_units @ dc_anchor
    negative_dc = negative_units @ dc_anchor

    rows = []
    matrices = {
        "near100_p20": np.full((50, 10), np.nan, dtype=np.float64),
        "near0_p1": np.full((50, 10), np.nan, dtype=np.float64),
    }
    checkpoint_slots = {value: slot for slot, value in enumerate(checkpoint_values)}
    for term, (checkpoint, timestep, position) in enumerate(
        zip(
            metadata["ckpt_indices"],
            metadata["timesteps"],
            metadata["snapshot_positions"],
        )
    ):
        cslot = checkpoint_slots[int(checkpoint)]
        tslot = timestep_slots[int(timestep)]
        matrices["near100_p20"][cslot, tslot] = positive_cos[term]
        matrices["near0_p1"][cslot, tslot] = negative_cos[term]
        rows.append(
            {
                "term": term,
                "checkpoint_ordinal": cslot + 1,
                "checkpoint_index": int(checkpoint),
                "timestep": int(timestep),
                "snapshot_position": int(position),
                "split": "anchor_train" if train_mask[term] else "heldout_test",
                "near100_selected_sign": int(positive_signs[tslot]),
                "near0_selected_sign": int(negative_signs[tslot]),
                "near100_cosine_to_anchor": float(positive_cos[term]),
                "near0_cosine_to_anchor": float(negative_cos[term]),
                "near100_near0_same_term_cosine": float(pair_cos[term]),
                "near100_cosine_to_dc": float(positive_dc[term]),
                "near0_cosine_to_dc": float(negative_dc[term]),
            }
        )

    summary_rows = []
    for split_name, mask in (("anchor_train", train_mask), ("heldout_test", heldout_mask)):
        for label, values in (
            ("near100_p20", positive_cos),
            ("near0_p1", negative_cos),
        ):
            selected = values[mask]
            summary_rows.append(
                {
                    "split": split_name,
                    "probe": label,
                    "terms": int(np.count_nonzero(mask)),
                    "mean_cosine_to_anchor": float(selected.mean()),
                    "std_cosine_to_anchor": float(selected.std()),
                    "positive_cosine_fraction": float(np.mean(selected > 0.0)),
                    "mean_abs_cosine_to_anchor": float(np.mean(np.abs(selected))),
                }
            )

    out_dir = (
        result_root
        / "eval"
        / "q0_extreme_probe_output_directions"
        / f"source_run_{args.source_run_id}"
    )
    write_csv(out_dir / "per_term_cosines.csv", rows)
    write_csv(out_dir / "split_summary.csv", summary_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "anchor_and_cosines.npz",
        anchor=anchor.astype(np.float32),
        positive_cosine=positive_cos,
        negative_cosine=negative_cos,
        same_term_pair_cosine=pair_cos,
        train_mask=train_mask,
        checkpoint_indices=metadata["ckpt_indices"],
        timesteps=metadata["timesteps"],
        snapshot_positions=metadata["snapshot_positions"],
    )
    for label, matrix in matrices.items():
        save_heatmap(
            out_dir / f"{label}_cosine_to_p20_train_anchor.png",
            matrix,
            f"Q0 {label}: flipped raw-v cosine to P20 train-half anchor",
        )

    summary = {
        "query": args.query_id,
        "near0_global_probe": args.negative_global_probe,
        "near100_global_probe": args.positive_global_probe,
        "near0_artifact": str(negative_artifact),
        "near100_artifact": str(positive_artifact),
        "anchor_definition": "normalized mean of flipped near100 P20 unit probes over checkpoints 1-25",
        "anchor_resultant_norm": anchor_resultant,
        "random_resultant_baseline": 1.0 / math.sqrt(np.count_nonzero(train_mask)),
        "heldout_same_term_pair_cosine_mean": float(pair_cos[heldout_mask].mean()),
        "heldout_same_term_pair_cosine_std": float(pair_cos[heldout_mask].std()),
        "heldout_near100_dc_cosine_mean": float(positive_dc[heldout_mask].mean()),
        "heldout_near0_dc_cosine_mean": float(negative_dc[heldout_mask].mean()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    print("Q0 FLIPPED RAW OUTPUT-PROBE DIRECTIONS")
    print(
        f"anchor=P20 near100, checkpoints 1-25 | resultant={anchor_resultant:.6f} "
        f"random_baseline={summary['random_resultant_baseline']:.6f}"
    )
    print(f"{'SPLIT':13s} {'PROBE':12s} {'MEAN COS':>10s} {'STD':>10s} {'COS+':>8s}")
    print("-" * 60)
    for row in summary_rows:
        print(
            f"{str(row['split']):13s} {str(row['probe']):12s} "
            f"{float(row['mean_cosine_to_anchor']):+10.6f} "
            f"{float(row['std_cosine_to_anchor']):10.6f} "
            f"{float(row['positive_cosine_fraction']):8.3f}"
        )
    print(
        f"heldout same-term P20/P1 cosine: "
        f"{summary['heldout_same_term_pair_cosine_mean']:+.6f} "
        f"± {summary['heldout_same_term_pair_cosine_std']:.6f}"
    )
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
