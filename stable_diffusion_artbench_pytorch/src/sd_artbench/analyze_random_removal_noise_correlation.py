"""Correlate random-removal endpoint/trajectory changes with noise changes.

For each query, the full-data LoRA is sampled once while recording every UNet
input and its classifier-free-guided predicted noise.  The matching random
top-k-removal LoRA is then evaluated on those *same full-data trajectory
inputs*.  This isolates the model's predicted-noise change from differences
between the two generated latent trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from scipy.stats import pearsonr, spearmanr

from .config import ArtBenchConfig
from .evaluate_topk_removal_trajectory_mse import (
    expected_timesteps,
    load_lora,
    load_queries,
    parse_csv_ints,
    removal_lora,
)


RANDOM_METHOD = "random_full5000_seeded"


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
    parser.add_argument("--removal-root", type=Path, required=True)
    parser.add_argument("--endpoint-trajectory-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs/random-removal-noise-correlation-q0-59",
    )
    parser.add_argument("--method", default=RANDOM_METHOD)
    parser.add_argument(
        "--query-indices", type=parse_csv_ints, default=list(range(60))
    )
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--model", default="lambdalabs/miniSD-diffusers")
    return parser.parse_args()


def load_endpoint_trajectory_metrics(
    path: Path, method: str, query_indices: Sequence[int]
) -> dict[int, dict[str, float]]:
    selected: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != method:
                continue
            query_index = int(row["query_index"])
            if query_index not in query_indices:
                continue
            if query_index in selected:
                raise RuntimeError(f"duplicate {method} metric for Q{query_index}")
            selected[query_index] = {
                "endpoint_mse": float(row["endpoint_mse"]),
                "trajectory_mse": float(row["trajectory_mse"]),
            }
    missing = sorted(set(query_indices) - set(selected))
    if missing:
        raise RuntimeError(f"missing {method} endpoint/trajectory metrics: {missing}")
    return selected


class FullTrajectoryNoiseCapture:
    """Capture scaled UNet inputs and guided outputs used by the pipeline."""

    def __init__(self, guidance_scale: float) -> None:
        self.guidance_scale = float(guidance_scale)
        self.inputs: list[dict[str, torch.Tensor | int]] = []
        self.noises: list[np.ndarray] = []
        self._pending: dict[str, torch.Tensor | int] | None = None

    def pre_hook(
        self,
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        sample = args[0] if args else kwargs["sample"]
        timestep = args[1] if len(args) > 1 else kwargs["timestep"]
        embeddings = kwargs["encoder_hidden_states"]
        timestep_value = int(timestep.flatten()[0].item()) if torch.is_tensor(timestep) else int(timestep)
        self._pending = {
            "sample": sample.detach().to(device="cpu", dtype=torch.float16).clone(),
            "timestep": timestep_value,
            "encoder_hidden_states": embeddings.detach()
            .to(device="cpu", dtype=torch.float16)
            .clone(),
        }

    def post_hook(
        self,
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        if self._pending is None:
            raise RuntimeError("UNet post-hook ran without a captured input")
        raw = output.sample.detach().float()
        if raw.shape[0] == 2:
            unconditional, conditional = raw.chunk(2)
            guided = unconditional + self.guidance_scale * (conditional - unconditional)
        elif raw.shape[0] == 1 and self.guidance_scale <= 1.0:
            guided = raw
        else:
            raise RuntimeError(f"unexpected classifier-free-guidance batch: {raw.shape}")
        self.inputs.append(self._pending)
        self.noises.append(guided[0].cpu().numpy().copy())
        self._pending = None


@torch.no_grad()
def capture_full_data_noise(
    pipeline: StableDiffusionPipeline,
    *,
    prompt: str,
    seed: int,
    steps: int,
    guidance_scale: float,
) -> tuple[list[dict[str, torch.Tensor | int]], np.ndarray]:
    capture = FullTrajectoryNoiseCapture(guidance_scale)
    pre = pipeline.unet.register_forward_pre_hook(capture.pre_hook, with_kwargs=True)
    post = pipeline.unet.register_forward_hook(capture.post_hook, with_kwargs=True)
    try:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        pipeline(
            prompt,
            height=256,
            width=256,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
    finally:
        pre.remove()
        post.remove()
    if len(capture.inputs) != steps or len(capture.noises) != steps:
        raise RuntimeError(
            f"expected {steps} captured UNet calls, found "
            f"{len(capture.inputs)} inputs and {len(capture.noises)} outputs"
        )
    return capture.inputs, np.stack(capture.noises)


@torch.no_grad()
def predicted_noise_mse_on_reference_inputs(
    pipeline: StableDiffusionPipeline,
    reference_inputs: Sequence[dict[str, torch.Tensor | int]],
    reference_noise: np.ndarray,
    guidance_scale: float,
) -> np.ndarray:
    values = []
    for step, state in enumerate(reference_inputs):
        sample = state["sample"]
        embeddings = state["encoder_hidden_states"]
        if not torch.is_tensor(sample) or not torch.is_tensor(embeddings):
            raise TypeError("captured UNet tensors are invalid")
        sample = sample.to(device=pipeline.device, dtype=pipeline.unet.dtype)
        embeddings = embeddings.to(device=pipeline.device, dtype=pipeline.unet.dtype)
        raw = pipeline.unet(
            sample,
            int(state["timestep"]),
            encoder_hidden_states=embeddings,
        ).sample.float()
        if raw.shape[0] == 2:
            unconditional, conditional = raw.chunk(2)
            guided = unconditional + guidance_scale * (conditional - unconditional)
        elif raw.shape[0] == 1 and guidance_scale <= 1.0:
            guided = raw
        else:
            raise RuntimeError(f"unexpected candidate UNet output: {raw.shape}")
        reference = torch.from_numpy(reference_noise[step]).to(
            device=guided.device, dtype=torch.float32
        )
        values.append(float(torch.mean((guided[0] - reference) ** 2).item()))
    return np.asarray(values, dtype=np.float64)


def correlation(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    pearson = pearsonr(left, right)
    spearman = spearmanr(left, right)
    return {
        "pearson": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def read_completed(path: Path, steps: int) -> dict[int, list[dict[str, str]]]:
    if not path.is_file():
        return {}
    grouped: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(int(row["query_index"]), []).append(row)
    return {
        query: sorted(rows, key=lambda row: int(row["step"]))
        for query, rows in grouped.items()
        if len(rows) == steps
        and [int(row["step"]) for row in rows] == list(range(steps))
    }


def write_noise_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "query_index",
        "seed",
        "step",
        "timestep",
        "predicted_noise_mse",
        "endpoint_mse",
        "trajectory_mse",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["query_index"], row["step"])))


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    query_indices: Sequence[int],
    timesteps: Sequence[int],
) -> None:
    by_query = {query: [] for query in query_indices}
    for row in rows:
        by_query[int(row["query_index"])].append(row)
    noise = np.asarray(
        [
            [float(row["predicted_noise_mse"]) for row in sorted(by_query[q], key=lambda x: x["step"])]
            for q in query_indices
        ]
    )
    endpoint = np.asarray([float(by_query[q][0]["endpoint_mse"]) for q in query_indices])
    trajectory = np.asarray([float(by_query[q][0]["trajectory_mse"]) for q in query_indices])
    correlation_rows = []
    for step, timestep in enumerate(timesteps):
        end = correlation(noise[:, step], endpoint)
        traj = correlation(noise[:, step], trajectory)
        correlation_rows.append(
            {
                "step": step,
                "timestep": timestep,
                "queries": len(query_indices),
                "endpoint_pearson": end["pearson"],
                "endpoint_pearson_p": end["pearson_p"],
                "endpoint_spearman": end["spearman"],
                "endpoint_spearman_p": end["spearman_p"],
                "trajectory_pearson": traj["pearson"],
                "trajectory_pearson_p": traj["pearson_p"],
                "trajectory_spearman": traj["spearman"],
                "trajectory_spearman_p": traj["spearman_p"],
            }
        )
    correlation_path = output_dir / "correlation_by_timestep.csv"
    with correlation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(correlation_rows[0]))
        writer.writeheader()
        writer.writerows(correlation_rows)

    lines = [
        "ARTBENCH RANDOM TOP-1000 REMOVAL — PREDICTED-NOISE CORRELATION",
        "=" * 146,
        f"queries={len(query_indices)} method={RANDOM_METHOD}",
        "Predicted-noise MSE compares random-removal and full-data LoRAs on",
        "identical full-data DDIM trajectory inputs at every timestep.",
        "",
        f"{'STEP':>4} {'TIME':>5} {'END PEARSON':>13} {'P':>11} {'END SPEARMAN':>14} {'P':>11} "
        f"{'TRAJ PEARSON':>14} {'P':>11} {'TRAJ SPEARMAN':>15} {'P':>11}",
        "-" * 146,
    ]
    for row in correlation_rows:
        lines.append(
            f"{row['step']:>4d} {row['timestep']:>5d} "
            f"{row['endpoint_pearson']:>+13.6f} {row['endpoint_pearson_p']:>11.4g} "
            f"{row['endpoint_spearman']:>+14.6f} {row['endpoint_spearman_p']:>11.4g} "
            f"{row['trajectory_pearson']:>+14.6f} {row['trajectory_pearson_p']:>11.4g} "
            f"{row['trajectory_spearman']:>+15.6f} {row['trajectory_spearman_p']:>11.4g}"
        )
    (output_dir / "correlation_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    x = np.asarray(timesteps)
    for axis, prefix, title in (
        (axes[0], "endpoint", "Endpoint difference"),
        (axes[1], "trajectory", "Trajectory difference"),
    ):
        axis.plot(x, [row[f"{prefix}_pearson"] for row in correlation_rows], label="Pearson", color="#4C78A8")
        axis.plot(x, [row[f"{prefix}_spearman"] for row in correlation_rows], label="Spearman", color="#F2A65A")
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Diffusion timestep")
        axis.grid(axis="y", alpha=0.22)
        axis.invert_xaxis()
    axes[0].set_ylabel("Correlation across 60 queries")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "correlation_by_timestep.png", dpi=220)
    figure.savefig(output_dir / "correlation_by_timestep.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("predicted-noise evaluation requires CUDA")
    records = load_queries(args.query_manifest, args.query_indices)
    metrics = load_endpoint_trajectory_metrics(
        args.endpoint_trajectory_csv, args.method, args.query_indices
    )
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
    pipeline.set_progress_bar_config(disable=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = args.output_dir / "predicted_noise_per_query_timestep.csv"
    completed = read_completed(per_query_path, args.steps)
    rows: list[dict[str, Any]] = [
        dict(row) for query_rows in completed.values() for row in query_rows
    ]
    for position, record in enumerate(records, start=1):
        query_index = int(record["query_index"])
        if query_index in completed:
            print(f"[skip {position}/{len(records)}] Q{query_index} complete", flush=True)
            continue
        print(f"[reference {position}/{len(records)}] Q{query_index}", flush=True)
        load_lora(pipeline, args.full_lora_dir)
        reference_inputs, reference_noise = capture_full_data_noise(
            pipeline,
            prompt=str(record["prompt"]),
            seed=int(record["seed"]),
            steps=args.steps,
            guidance_scale=args.guidance_scale,
        )
        actual_timesteps = [int(state["timestep"]) for state in reference_inputs]
        if actual_timesteps != timesteps:
            raise RuntimeError(
                f"Q{query_index} schedule mismatch: {actual_timesteps} vs {timesteps}"
            )
        print(f"[random {position}/{len(records)}] Q{query_index}", flush=True)
        load_lora(
            pipeline,
            removal_lora(args.removal_root, args.method, query_index, args.top_k),
        )
        noise_mse = predicted_noise_mse_on_reference_inputs(
            pipeline, reference_inputs, reference_noise, args.guidance_scale
        )
        metric = metrics[query_index]
        for step, (timestep, value) in enumerate(zip(timesteps, noise_mse)):
            rows.append(
                {
                    "query_index": query_index,
                    "seed": int(record["seed"]),
                    "step": step,
                    "timestep": timestep,
                    "predicted_noise_mse": float(value),
                    "endpoint_mse": metric["endpoint_mse"],
                    "trajectory_mse": metric["trajectory_mse"],
                }
            )
        write_noise_rows(per_query_path, rows)
        print(
            f"[saved {position}/{len(records)}] Q{query_index} "
            f"noise_mean={noise_mse.mean():.8g}",
            flush=True,
        )

    completed = read_completed(per_query_path, args.steps)
    missing = sorted(set(args.query_indices) - set(completed))
    if missing:
        raise RuntimeError(f"incomplete queries after evaluation: {missing}")
    final_rows = [dict(row) for query_rows in completed.values() for row in query_rows]
    write_outputs(args.output_dir, final_rows, args.query_indices, timesteps)
    metadata = {
        "query_manifest": str(args.query_manifest.resolve()),
        "full_lora_dir": str(args.full_lora_dir.resolve()),
        "removal_root": str(args.removal_root.resolve()),
        "endpoint_trajectory_csv": str(args.endpoint_trajectory_csv.resolve()),
        "method": args.method,
        "query_indices": args.query_indices,
        "top_k": args.top_k,
        "steps": args.steps,
        "timesteps": timesteps,
        "guidance_scale": args.guidance_scale,
        "predicted_noise_definition": (
            "guided UNet output MSE between random-removal and full-data LoRAs "
            "on identical scaled full-data DDIM trajectory inputs"
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
