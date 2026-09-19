#!/usr/bin/env python3
"""Materialize raw E1/four-event banks without timestep alignment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_adamw_four_event_original_f_scores import event
from run_expected_residual_jacobian_scores import load_query_bank


SEMANTICS = "fixed_checkpoint_raw_event_gradient_no_timestamp_alignment"


def atomic_save(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--attribution-points", type=int, default=5000)
    args = parser.parse_args()

    source_root = (
        ROOT / "result" / args.experiment
        / f"fixed_checkpoint_raw_four_events_n{args.attribution_points}"
    )

    class QueryArgs:
        pass

    query_args = QueryArgs()
    query_args.experiment = args.experiment
    query_args.train_seed = args.train_seed
    query_args.epochs = 200
    _, metadata = load_query_bank(
        query_args,
        "loss_direction_residual_rms_original_f",
        "trajectory_next_checkpoint_noise_mse",
        [0],
    )
    checkpoint_indices = np.asarray(metadata["ckpt_indices"], dtype=np.int32)
    all_timesteps = np.asarray(metadata["timesteps"], dtype=np.int32)
    all_weights = np.asarray(metadata["term_weights"], dtype=np.float64)

    last_indices = None
    for checkpoint in range(49):
        start_epoch = 4 * (checkpoint + 1)
        source = source_root / f"epoch_{start_epoch}_{start_epoch + 4}"
        events = []
        score_indices = None
        for epoch in range(start_epoch + 1, start_epoch + 5):
            features, indices = event(source, epoch)
            if score_indices is None:
                score_indices = indices
            elif not np.array_equal(score_indices, indices):
                raise ValueError(f"score-index mismatch in {source}")
            events.append(features)
        assert score_indices is not None
        last_indices = score_indices
        mask = checkpoint_indices == checkpoint
        timesteps = all_timesteps[mask]
        weights = all_weights[mask]
        if len(timesteps) != 10:
            raise ValueError(
                f"checkpoint {checkpoint} expected 10 query timestamps, got {len(timesteps)}"
            )
        banks = {"four": sum(events), "e1": events[0]}
        for method, bank in banks.items():
            output = (
                ROOT / "result" / args.experiment / "model" / "prompted_solo"
                / f"seed_{args.train_seed}_train_gradient"
                / f"traj_tracin_raw4_{method}"
                / "train_datapoint_gradient_artifact.npz.parts"
                / f"ckpt_{checkpoint:04d}.npz"
            )
            if not output.is_file():
                atomic_save(
                    output,
                    train_features=bank[None, :, :],
                    score_indices=score_indices,
                    ckpt_indices=np.full(10, checkpoint, dtype=np.int32),
                    timesteps=timesteps,
                    term_weights=weights,
                    train_feature_semantics=np.asarray(SEMANTICS),
                    timestamp_shared_train_feature=np.asarray(1, dtype=np.int8),
                )
        print(f"[materialize raw] checkpoint={checkpoint + 1}/49", flush=True)

    assert last_indices is not None
    for method in ("four", "e1"):
        output = (
            ROOT / "result" / args.experiment / "model" / "prompted_solo"
            / f"seed_{args.train_seed}_train_gradient"
            / f"traj_tracin_raw4_{method}"
            / "train_datapoint_gradient_artifact.npz.parts"
            / "ckpt_0049.npz"
        )
        if not output.is_file():
            atomic_save(
                output,
                train_features=np.zeros(
                    (1, args.attribution_points, 4096), dtype=np.float32
                ),
                score_indices=last_indices,
                ckpt_indices=np.empty((0,), dtype=np.int32),
                timesteps=np.empty((0,), dtype=np.int32),
                term_weights=np.empty((0,), dtype=np.float64),
                train_feature_semantics=np.asarray(SEMANTICS),
                timestamp_shared_train_feature=np.asarray(1, dtype=np.int8),
            )


if __name__ == "__main__":
    main()
