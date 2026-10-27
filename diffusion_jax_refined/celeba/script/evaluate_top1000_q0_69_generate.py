#!/usr/bin/env python3
"""Generate paired CelebA Top-1000 removal trajectories and pixel metrics."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
CELEBA_ROOT = SCRIPT_DIR.parent
REFINED_ROOT = CELEBA_ROOT.parent
LEGACY_ROOT = REFINED_ROOT / "legacy_jax"
for path in (CELEBA_ROOT, REFINED_ROOT, LEGACY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_config_attr40 import DATA_ROOT, _prompt_path_tag
from LDS.DM_cifar_lds import CifarTargetEvaluator


METHODS = (
    "endpoint_pollute_adamw_timestamp_both_l2_full10k_pos",
    "endpoint_tracin_adamw_linear_raw_full10k_neg",
    "endpoint_tracin_adamw_linear_both_l2_full10k_neg",
    "retrac_adamw_query_train_l2_full10k_neg",
    "random_full10k_seeded",
)
METHOD_LABELS = {
    METHODS[0]: "PINT-BOTH-L2",
    METHODS[1]: "TRACIN-RAW",
    METHODS[2]: "TRACIN-BOTH-L2",
    METHODS[3]: "RETRAC-BOTH-L2",
    METHODS[4]: "RANDOM",
}
RANDOM_METHOD = METHODS[4]


def parse_ids(text: str) -> list[int]:
    values = [int(value) for value in text.replace(",", " ").split() if value]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one query id")
    return values


def load_queries(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["queries"] if isinstance(payload, dict) else payload
    return {
        int(record.get("query_id", record.get("id", position))): dict(record)
        for position, record in enumerate(records)
    }


def query_seed(record: dict[str, Any]) -> int:
    return int(record.get("initial_seed", record.get("seed")))


def select_steps(step_count: int, count: int) -> np.ndarray:
    if count < 2 or count > step_count:
        raise ValueError("saved trajectory steps must be in [2, full step count]")
    indices = np.rint(np.linspace(0, step_count - 1, count)).astype(np.int64)
    if len(np.unique(indices)) != count:
        raise AssertionError("trajectory step selection contains duplicates")
    return indices


def display_image(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3:
        raise ValueError(f"expected image-shaped value, found {image.shape}")
    image = image.astype(np.float32)
    if float(np.nanmin(image)) < -0.05:
        image = (image + 1.0) / 2.0
    elif float(np.nanmax(image)) > 1.5:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def save_png(path: Path, value: np.ndarray) -> None:
    image = display_image(value)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    Image.fromarray(np.rint(image[..., :3] * 255.0).astype(np.uint8), mode="RGB").save(path)


def paired_mse(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, float, float]:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} vs {right.shape}")
    axes = tuple(range(1, left.ndim))
    per_step = np.mean(np.square(left - right), axis=axes, dtype=np.float64)
    mse = float(np.mean(per_step, dtype=np.float64))
    return per_step, mse, float(np.sqrt(mse))


def endpoint_pixel_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    left = display_image(reference).astype(np.float64)
    right = display_image(candidate).astype(np.float64)
    if left.shape != right.shape:
        raise ValueError(f"endpoint image shape mismatch: {left.shape} vs {right.shape}")
    mse = float(np.mean(np.square(left - right), dtype=np.float64))
    return mse, float(np.sqrt(mse))


def find_run_dir(roots: Sequence[Path], method: str, query_id: int) -> Path:
    matches_by_target: dict[Path, Path] = {}
    for root in roots:
        for path in sorted((root / method).glob(f"query_{query_id:03d}_seed_*")):
            matches_by_target[path.resolve()] = path
    matches = sorted(matches_by_target.values())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one run for {method} Q{query_id} under "
            f"{', '.join(map(str, roots))}, found {len(matches)}"
        )
    return matches[0]


def local_checkpoint(run_dir: Path, train_seed: int, epochs: int) -> Path:
    path = run_dir / "model" / f"seed_{train_seed}_epoch_{epochs:04d}.ckpt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", type=parse_ids, default=list(range(70)))
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--trajectory-steps", type=int, default=10)
    parser.add_argument("--prefer-device", default="gpu")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    queries = load_queries(args.query_file)
    result_root = CELEBA_ROOT / "result" / args.experiment
    removal_root = result_root / "endpoint_top1000_removal_q0_69"
    historical_root = result_root / "endpoint_top1000_removal_q0_9"
    random_root = result_root / "endpoint_random_top1000_removal_q0_69"
    output_dir = args.output_dir or (
        result_root / "analysis" / "endpoint_top1000_q0_69_five_method_metrics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base_checkpoint = (
        result_root
        / "model"
        / "prompted_jax"
        / f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt"
    )
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)

    for query_id in args.query_ids:
        if query_id not in queries:
            raise KeyError(f"query {query_id} is absent from {args.query_file}")
        record = queries[query_id]
        seed = query_seed(record)
        prompt = str(record["prompt"])
        query_output = output_dir / f"query_{query_id:03d}_seed_{seed:03d}"
        query_output.mkdir(parents=True, exist_ok=True)
        complete_path = query_output / "METRICS_GENERATION_COMPLETE"
        if complete_path.is_file():
            print(f"[skip complete] Q{query_id}", flush=True)
            continue

        reference_root = (
            result_root
            / "sample_ddim_eta0_50"
            / "cifar"
            / f"prompt_{_prompt_path_tag(prompt)}"
            / f"model_prompted_solo__ckpt_{base_checkpoint.stem}"
        )
        evaluator = CifarTargetEvaluator(
            code_file=str(LEGACY_ROOT / "DM__training_CELEBA_pixel.py"),
            base_checkpoint=str(base_checkpoint),
            prompt=prompt,
            prefer_device=args.prefer_device,
            data_root=DATA_ROOT,
            target_function="traj_counterfactual",
            sample_root=str(reference_root),
            sample_seed=seed,
            sample_index=0,
            max_trajectory_steps=None,
            trajectory_reduction="mean",
            trajectory_projection=None,
            simple_loss_timesteps=[0],
            simple_loss_noise_seeds=None,
            simple_loss_num_mc=1,
            simple_loss_mc_seed=0,
            trajectory_sampler="ddim_eta0",
        )
        indices = select_steps(len(evaluator.t_seq), args.trajectory_steps)
        timesteps = np.asarray(evaluator.t_seq, dtype=np.int32)[indices]
        reference = np.asarray(evaluator.xt_ref, dtype=np.float32)[indices]
        np.savez_compressed(
            query_output / "original_trajectory.npz",
            states=reference,
            timesteps=timesteps,
        )
        save_png(query_output / "original_endpoint.png", reference[-1])

        rows: list[dict[str, Any]] = []
        for method in METHODS:
            roots = (
                (random_root,)
                if method == RANDOM_METHOD
                else (removal_root, historical_root)
            )
            run_dir = find_run_dir(roots, method, query_id)
            checkpoint = local_checkpoint(run_dir, args.train_seed, args.epochs)
            output_npz = query_output / f"{method}_trajectory.npz"
            output_png = query_output / f"{method}_endpoint.png"
            if output_npz.is_file() and output_png.is_file():
                with np.load(output_npz) as payload:
                    candidate = np.asarray(payload["states"], dtype=np.float32)
                print(f"[reuse] Q{query_id} {METHOD_LABELS[method]}", flush=True)
            else:
                adapter = evaluator._make_adapter(str(checkpoint))
                candidate = np.asarray(
                    evaluator._sample_model_space_trajectory(
                        adapter,
                        seed=seed,
                        timesteps_to_save=timesteps,
                    ),
                    dtype=np.float32,
                )
                np.savez_compressed(
                    output_npz,
                    states=candidate,
                    timesteps=timesteps,
                )
                save_png(output_png, candidate[-1])
                del adapter
                gc.collect()
                print(f"[sampled] Q{query_id} {METHOD_LABELS[method]}", flush=True)

            per_step, trajectory_mse, trajectory_rmse = paired_mse(reference, candidate)
            endpoint_mse, endpoint_rmse = endpoint_pixel_metrics(
                reference[-1], candidate[-1]
            )
            rows.append(
                {
                    "query_id": query_id,
                    "seed": seed,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "endpoint_mse": endpoint_mse,
                    "endpoint_rmse": endpoint_rmse,
                    "trajectory_mse": trajectory_mse,
                    "trajectory_rmse": trajectory_rmse,
                    "trajectory_steps": len(timesteps),
                    "checkpoint": str(checkpoint.resolve()),
                }
            )
            (query_output / f"{method}_per_step_mse.json").write_text(
                json.dumps(
                    {
                        "query_id": query_id,
                        "method": method,
                        "timesteps": [int(value) for value in timesteps],
                        "per_step_mse": [float(value) for value in per_step],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        write_csv(query_output / "pixel_trajectory_metrics.csv", rows)
        complete_path.write_text("complete\n", encoding="utf-8")
        print(f"[complete] Q{query_id}", flush=True)
        del evaluator
        gc.collect()

    print(f"[done] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
