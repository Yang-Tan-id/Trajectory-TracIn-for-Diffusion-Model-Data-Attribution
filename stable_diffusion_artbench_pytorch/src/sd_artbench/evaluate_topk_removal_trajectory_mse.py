"""Evaluate endpoint and DDIM latent-trajectory MSE for Top-K removal LoRAs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image

from .config import ArtBenchConfig
from .train_lora import set_attention_dropout


METHODS = (
    "endpoint_pollute_query_train_l2",
    "endpoint_tracin_raw",
    "endpoint_tracin_query_train_l2",
    "retrac_query_train_l2",
)


def parse_csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def parse_args() -> argparse.Namespace:
    root = ArtBenchConfig().index_root.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-manifest",
        type=Path,
        default=root / "outputs/artbench2-queries-60/manifest.json",
    )
    parser.add_argument(
        "--full-lora-dir",
        type=Path,
        default=root / "outputs/artbench2-lora/final",
    )
    parser.add_argument(
        "--removal-root",
        type=Path,
        default=root / "outputs/topk-removal-adamw-q0-9",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs/topk-removal-trajectory-mse-q0-9",
    )
    parser.add_argument("--query-indices", type=parse_csv_ints, default=list(range(10)))
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--model", default="lambdalabs/miniSD-diffusers")
    return parser.parse_args()


def load_queries(path: Path, indices: Sequence[int]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"query manifest must be a non-empty list: {path}")
    records: list[dict[str, Any]] = []
    for index in indices:
        if not 0 <= index < len(payload):
            raise IndexError(f"query index outside manifest: {index}")
        record = dict(payload[index])
        record["query_index"] = index
        record["seed"] = int(record["seed"])
        source = Path(record["file"])
        if not source.is_absolute():
            source = path.parent / source
        record["file"] = str(source.resolve())
        records.append(record)
    return records


def mean_squared_error(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    if left64.shape != right64.shape:
        raise ValueError(f"shape mismatch: {left64.shape} vs {right64.shape}")
    return float(np.mean(np.square(left64 - right64), dtype=np.float64))


def trajectory_metrics(
    reference: np.ndarray, candidate: np.ndarray
) -> tuple[np.ndarray, float, float]:
    reference64 = np.asarray(reference, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    if reference64.shape != candidate64.shape:
        raise ValueError(
            f"trajectory shape mismatch: {reference64.shape} vs {candidate64.shape}"
        )
    if reference64.ndim < 2:
        raise ValueError("trajectory must have shape [steps, ...]")
    axes = tuple(range(1, reference64.ndim))
    per_step = np.mean(np.square(reference64 - candidate64), axis=axes)
    mse = float(np.mean(per_step, dtype=np.float64))
    return per_step, mse, float(np.sqrt(mse))


def image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def expected_timesteps(training_timesteps: int, steps: int) -> list[int]:
    if steps < 1 or training_timesteps % steps:
        raise ValueError(
            f"training timesteps {training_timesteps} must be divisible by steps {steps}"
        )
    stride = training_timesteps // steps
    return [training_timesteps - 1 - stride * index for index in range(steps)]


@torch.no_grad()
def sample_with_trajectory(
    pipeline: StableDiffusionPipeline,
    *,
    prompt: str,
    seed: int,
    steps: int,
    guidance_scale: float,
    timesteps: Sequence[int],
) -> tuple[Image.Image, np.ndarray]:
    states: list[np.ndarray] = []

    def record_state(_step: int, _timestep: int, latents: torch.Tensor) -> None:
        states.append(latents[0].detach().float().cpu().numpy().copy())

    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    output = pipeline(
        prompt,
        height=256,
        width=256,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        callback=record_state,
        callback_steps=1,
    )
    actual = [int(value) for value in pipeline.scheduler.timesteps.tolist()]
    if actual != list(timesteps):
        raise RuntimeError(f"DDIM schedule mismatch: actual={actual}, expected={timesteps}")
    if len(states) != steps:
        raise RuntimeError(f"expected {steps} trajectory states, recorded {len(states)}")
    image = output.images[0].convert("RGB")
    extrema = image.getextrema()
    if all(low == 0 and high == 0 for low, high in extrema):
        raise RuntimeError("generation produced an all-black image")
    return image, np.stack(states, axis=0)


def load_lora(pipeline: StableDiffusionPipeline, path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(path)
    pipeline.unet.load_attn_procs(path)
    set_attention_dropout(pipeline.unet, 0.0)
    pipeline.unet.eval()


def removal_lora(root: Path, method: str, query_index: int, top_k: int) -> Path:
    run_dir = root / method / f"query-{query_index:03d}" / f"topk-{top_k:04d}"
    if not (run_dir / "TRAINING_COMPLETE").is_file():
        raise FileNotFoundError(run_dir / "TRAINING_COMPLETE")
    path = run_dir / "model/final"
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "ARTBENCH Q0-Q9 TOP-1000 REMOVAL — ENDPOINT / TRAJECTORY MSE",
        "Larger endpoint and trajectory distance is better.",
        "",
        (
            f"{'QUERY':<7} {'METHOD':<31} {'END-MSE':>12} {'END-RMSE':>12} "
            f"{'TRAJ-MSE':>12} {'TRAJ-RMSE':>12}"
        ),
        "-" * 94,
    ]
    for row in rows:
        lines.append(
            f"Q{row['query_index']:<6d} {row['method']:<31} "
            f"{row['endpoint_mse']:>12.8f} {row['endpoint_rmse']:>12.8f} "
            f"{row['trajectory_mse']:>12.8f} {row['trajectory_rmse']:>12.8f}"
        )
    lines.extend(["", "METHOD SUMMARY", "-" * 94])
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        lines.append(
            f"{method:<31} "
            f"END-MSE={np.mean([r['endpoint_mse'] for r in selected]):.8f} "
            f"END-RMSE={np.mean([r['endpoint_rmse'] for r in selected]):.8f} "
            f"TRAJ-MSE={np.mean([r['trajectory_mse'] for r in selected]):.8f} "
            f"TRAJ-RMSE={np.mean([r['trajectory_rmse'] for r in selected]):.8f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.top_k < 1 or args.steps < 1:
        raise ValueError("top-k and steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("trajectory evaluation requires a CUDA GPU")
    records = load_queries(args.query_manifest, args.query_indices)
    device = torch.device("cuda")
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    training_timesteps = int(pipeline.scheduler.config.num_train_timesteps)
    timesteps = expected_timesteps(training_timesteps, args.steps)
    stride = training_timesteps // args.steps
    pipeline.scheduler = DDIMScheduler.from_config(
        pipeline.scheduler.config, steps_offset=stride - 1
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    timestep_rows: list[dict[str, Any]] = []

    for record in records:
        query_index = int(record["query_index"])
        query_dir = args.output_dir / f"query-{query_index:03d}"
        query_dir.mkdir(parents=True, exist_ok=True)
        print(f"[reference] Q{query_index} full-data LoRA", flush=True)
        load_lora(pipeline, args.full_lora_dir)
        reference_image, reference_trajectory = sample_with_trajectory(
            pipeline,
            prompt=str(record["prompt"]),
            seed=int(record["seed"]),
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            timesteps=timesteps,
        )
        reference_image.save(query_dir / "full-data.png")
        np.savez_compressed(
            query_dir / "full-data-trajectory.npz",
            states=reference_trajectory.astype(np.float16),
            timesteps=np.asarray(timesteps, dtype=np.int64),
        )
        source_path = Path(record["file"])
        source_mse = (
            mean_squared_error(image_array(Image.open(source_path)), image_array(reference_image))
            if source_path.is_file()
            else None
        )
        if source_mse is not None and source_mse > 1e-8:
            print(
                f"[warning] Q{query_index} replay differs from query image: "
                f"source_mse={source_mse:.8g}",
                flush=True,
            )

        for method in METHODS:
            print(f"[candidate] Q{query_index} {method}", flush=True)
            lora_dir = removal_lora(
                args.removal_root, method, query_index, args.top_k
            )
            load_lora(pipeline, lora_dir)
            candidate_image, candidate_trajectory = sample_with_trajectory(
                pipeline,
                prompt=str(record["prompt"]),
                seed=int(record["seed"]),
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                timesteps=timesteps,
            )
            candidate_image.save(query_dir / f"{method}.png")
            np.savez_compressed(
                query_dir / f"{method}-trajectory.npz",
                states=candidate_trajectory.astype(np.float16),
                timesteps=np.asarray(timesteps, dtype=np.int64),
            )
            endpoint_mse = mean_squared_error(
                image_array(reference_image), image_array(candidate_image)
            )
            per_step, trajectory_mse, trajectory_rmse = trajectory_metrics(
                reference_trajectory, candidate_trajectory
            )
            row = {
                "query_index": query_index,
                "method": method,
                "seed": int(record["seed"]),
                "endpoint_mse": endpoint_mse,
                "endpoint_rmse": float(np.sqrt(endpoint_mse)),
                "trajectory_mse": trajectory_mse,
                "trajectory_rmse": trajectory_rmse,
                "source_replay_mse": source_mse,
            }
            rows.append(row)
            for step, (timestep, mse) in enumerate(zip(timesteps, per_step)):
                timestep_rows.append(
                    {
                        "query_index": query_index,
                        "method": method,
                        "step": step,
                        "timestep": timestep,
                        "mse": float(mse),
                    }
                )
            print(
                f"[metric] Q{query_index} {method} endpoint_mse={endpoint_mse:.8f} "
                f"trajectory_mse={trajectory_mse:.8f}",
                flush=True,
            )

    with (args.output_dir / "endpoint_trajectory_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "per_timestep_mse.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timestep_rows[0]))
        writer.writeheader()
        writer.writerows(timestep_rows)
    write_report(args.output_dir / "endpoint_trajectory_report.txt", rows)
    metadata = {
        "query_manifest": str(args.query_manifest.resolve()),
        "full_lora_dir": str(args.full_lora_dir.resolve()),
        "removal_root": str(args.removal_root.resolve()),
        "query_indices": args.query_indices,
        "methods": list(METHODS),
        "top_k": args.top_k,
        "steps": args.steps,
        "timesteps": timesteps,
        "guidance_scale": args.guidance_scale,
        "trajectory_definition": "mean over DDIM post-step latent-state elementwise MSE",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
