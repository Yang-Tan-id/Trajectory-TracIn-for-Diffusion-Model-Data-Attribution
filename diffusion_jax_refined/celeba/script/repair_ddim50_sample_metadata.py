#!/usr/bin/env python3
"""Restore omitted DDIM-50 metadata after trajectory-only staging.

The original transfer manifest for the CelebA 250-task removal run copied the
trajectory arrays but not their sampler metadata.  This command validates the
exact saved timestep schedule before writing the missing per-seed metadata; it
never overwrites conflicting metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np


CELEBA_ROOT = Path(__file__).resolve().parents[1]
REFINED_ROOT = CELEBA_ROOT.parent
LEGACY_ROOT = REFINED_ROOT / "legacy_jax"
for path in (CELEBA_ROOT, LEGACY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_config_attr40 import _prompt_path_tag
from DM___sampler import inference_timesteps, selected_inference_timesteps


def load_queries(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text())
    records = payload["queries"] if isinstance(payload, dict) else payload
    return {
        int(record.get("query_id", record.get("id", position))): record
        for position, record in enumerate(records)
    }


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in re.split(r"[\s,]+", value) if item]


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-file",
        type=Path,
        default=CELEBA_ROOT / "queries_attr40_val25_x4.json",
    )
    parser.add_argument("--query-ids", default="0-69")
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--training-timesteps", type=int, default=1000)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--trajectory-steps", type=int, default=50)
    args = parser.parse_args()

    if args.query_ids == "0-69":
        query_ids = list(range(70))
    else:
        query_ids = parse_ints(args.query_ids)
    queries = load_queries(args.query_file)
    inference = inference_timesteps(
        args.training_timesteps, args.inference_steps
    )
    expected = np.asarray(
        sorted(
            selected_inference_timesteps(inference, args.trajectory_steps),
            reverse=True,
        ),
        dtype=np.int32,
    )
    result_root = CELEBA_ROOT / "result" / args.experiment
    sample_root = result_root / f"sample_ddim_eta0_{args.inference_steps}" / "cifar"
    checkpoint_stem = f"seed_{args.train_seed}_epoch_{args.epochs:04d}"
    repaired = 0
    already_valid = 0

    for query_id in query_ids:
        record = queries[query_id]
        seed = int(record.get("initial_seed", record.get("seed")))
        run_root = (
            sample_root
            / f"prompt_{_prompt_path_tag(str(record['prompt']))}"
            / f"model_prompted_solo__ckpt_{checkpoint_stem}"
        )
        seed_dir = run_root / f"seed_{seed:06d}"
        trajectory_path = seed_dir / "trajectory_xt.npy"
        timestep_path = seed_dir / "trajectory_t.npy"
        for required in (trajectory_path, timestep_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        timesteps = np.asarray(np.load(timestep_path), dtype=np.int32).reshape(-1)
        trajectory = np.load(trajectory_path, mmap_mode="r")
        if trajectory.shape[0] != len(timesteps):
            raise ValueError(
                f"{seed_dir}: trajectory/timestep length mismatch "
                f"{trajectory.shape[0]} != {len(timesteps)}"
            )
        if not np.array_equal(timesteps, expected):
            raise ValueError(
                f"{seed_dir}: cannot certify DDIM-{args.inference_steps}; "
                f"actual timesteps={timesteps.tolist()}, expected={expected.tolist()}"
            )

        info_path = seed_dir / "seed_info.json"
        if info_path.is_file():
            info = json.loads(info_path.read_text())
            if (
                info.get("trajectory_sampler") != "ddim_eta0"
                or int(info.get("num_inference_steps", -1))
                != args.inference_steps
            ):
                raise ValueError(f"conflicting sampler metadata in {info_path}")
            already_valid += 1
            continue
        write_json_atomic(
            info_path,
            {
                "seed": seed,
                "prompt": str(record["prompt"]),
                "model_tag": "prompted_solo",
                "batch_size": int(trajectory.shape[1]),
                "save_timesteps": [int(value) for value in timesteps],
                "trajectory_xt_shape": list(trajectory.shape),
                "trajectory_sampler": "ddim_eta0",
                "num_inference_steps": args.inference_steps,
                "metadata_reconstructed_after_staging": True,
            },
        )
        repaired += 1

    print(
        f"[metadata] certified DDIM-{args.inference_steps}: "
        f"repaired={repaired} already_valid={already_valid} total={len(query_ids)}"
    )


if __name__ == "__main__":
    main()
