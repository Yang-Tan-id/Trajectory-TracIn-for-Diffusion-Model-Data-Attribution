"""Cross-evaluate ArtBench queries on all random-removal models.

For each evaluation query, every independently trained random-removal LoRA is
compared with the full-data LoRA. Correlations are computed within a query
across random models, not by pairing query Qk only with random model Rk.
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
    image_array,
    load_lora,
    load_queries,
    mean_squared_error,
    parse_csv_ints,
    removal_lora,
    sample_with_trajectory,
    trajectory_metrics,
)

RANDOM_METHOD = "random_full5000_seeded"
PAIR_FIELDS = (
    "evaluation_query_index", "evaluation_seed", "random_model_index",
    "step", "timestep", "predicted_noise_mse", "endpoint_mse", "trajectory_mse",
)


def parse_args() -> argparse.Namespace:
    root = ArtBenchConfig().index_root.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-manifest", type=Path,
                        default=root / "outputs/artbench2-queries-60/manifest.json")
    parser.add_argument("--full-lora-dir", type=Path,
                        default=root / "outputs/artbench2-lora/final")
    parser.add_argument("--removal-root", type=Path, required=True)
    # Backward-compatible no-op for commands written for the old diagonal run.
    parser.add_argument("--endpoint-trajectory-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=root / "outputs/random-removal-noise-correlation-cross-q0-59")
    parser.add_argument("--method", default=RANDOM_METHOD)
    parser.add_argument("--query-indices", type=parse_csv_ints, default=list(range(60)),
                        help="Evaluation query prompts/seeds.")
    parser.add_argument("--model-indices", type=parse_csv_ints, default=list(range(60)),
                        help="Random-removal models applied to every query.")
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--model", default="lambdalabs/miniSD-diffusers")
    return parser.parse_args()


class FullTrajectoryNoiseCapture:
    def __init__(self, guidance_scale: float) -> None:
        self.guidance_scale = float(guidance_scale)
        self.inputs: list[dict[str, torch.Tensor | int]] = []
        self.noises: list[np.ndarray] = []
        self.pending: dict[str, torch.Tensor | int] | None = None

    def pre_hook(self, _module: torch.nn.Module, args: tuple[Any, ...],
                 kwargs: dict[str, Any]) -> None:
        sample = args[0] if args else kwargs["sample"]
        timestep = args[1] if len(args) > 1 else kwargs["timestep"]
        embeddings = kwargs["encoder_hidden_states"]
        timestep_value = (int(timestep.flatten()[0].item())
                          if torch.is_tensor(timestep) else int(timestep))
        self.pending = {
            "sample": sample.detach().to("cpu", torch.float16).clone(),
            "timestep": timestep_value,
            "encoder_hidden_states": embeddings.detach().to("cpu", torch.float16).clone(),
        }

    def post_hook(self, _module: torch.nn.Module, _args: tuple[Any, ...],
                  _kwargs: dict[str, Any], output: Any) -> None:
        if self.pending is None:
            raise RuntimeError("UNet post-hook ran without a captured input")
        raw = output.sample.detach().float()
        if raw.shape[0] == 2:
            unconditional, conditional = raw.chunk(2)
            guided = unconditional + self.guidance_scale * (conditional - unconditional)
        elif raw.shape[0] == 1 and self.guidance_scale <= 1.0:
            guided = raw
        else:
            raise RuntimeError(f"unexpected guidance batch: {raw.shape}")
        self.inputs.append(self.pending)
        self.noises.append(guided[0].cpu().numpy().copy())
        self.pending = None


@torch.no_grad()
def capture_full_data_noise(
    pipeline: StableDiffusionPipeline, *, prompt: str, seed: int, steps: int,
    guidance_scale: float,
) -> tuple[Any, np.ndarray, list[dict[str, torch.Tensor | int]], np.ndarray]:
    capture = FullTrajectoryNoiseCapture(guidance_scale)
    states: list[np.ndarray] = []

    def record_state(_step: int, _timestep: int, latents: torch.Tensor) -> None:
        states.append(latents[0].detach().float().cpu().numpy().copy())

    pre = pipeline.unet.register_forward_pre_hook(capture.pre_hook, with_kwargs=True)
    post = pipeline.unet.register_forward_hook(capture.post_hook, with_kwargs=True)
    try:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        output = pipeline(
            prompt, height=256, width=256, num_inference_steps=steps,
            guidance_scale=guidance_scale, generator=generator,
            callback=record_state, callback_steps=1,
        )
    finally:
        pre.remove()
        post.remove()
    if not (len(capture.inputs) == len(capture.noises) == len(states) == steps):
        raise RuntimeError(
            f"expected {steps} steps; inputs={len(capture.inputs)} "
            f"noise={len(capture.noises)} states={len(states)}"
        )
    return (output.images[0].convert("RGB"), np.stack(states), capture.inputs,
            np.stack(capture.noises))


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
        raw = pipeline.unet(
            sample.to(pipeline.device, pipeline.unet.dtype), int(state["timestep"]),
            encoder_hidden_states=embeddings.to(pipeline.device, pipeline.unet.dtype),
        ).sample.float()
        if raw.shape[0] == 2:
            unconditional, conditional = raw.chunk(2)
            guided = unconditional + guidance_scale * (conditional - unconditional)
        elif raw.shape[0] == 1 and guidance_scale <= 1.0:
            guided = raw
        else:
            raise RuntimeError(f"unexpected candidate output: {raw.shape}")
        reference = torch.from_numpy(reference_noise[step]).to(guided.device, torch.float32)
        values.append(float(torch.mean((guided[0] - reference) ** 2).item()))
    return np.asarray(values, dtype=np.float64)


def correlation(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if len(left) < 2:
        return {name: float("nan") for name in
                ("pearson", "pearson_p", "spearman", "spearman_p")}
    pearson = pearsonr(left, right)
    spearman = spearmanr(left, right)
    return {"pearson": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
            "spearman": float(spearman.statistic), "spearman_p": float(spearman.pvalue)}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def completed_pairs(rows: Sequence[dict[str, Any]], steps: int) -> set[tuple[int, int]]:
    grouped: dict[tuple[int, int], set[int]] = {}
    for row in rows:
        key = (int(row["evaluation_query_index"]), int(row["random_model_index"]))
        grouped.setdefault(key, set()).add(int(row["step"]))
    expected = set(range(steps))
    return {key for key, seen in grouped.items() if seen == expected}


def write_pair_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    # Explicit int conversion fixes mixed cached-string/new-int sorting.
    key = lambda row: (int(row["evaluation_query_index"]),
                       int(row["random_model_index"]), int(row["step"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=key))


def compute_correlation_rows(
    rows: Sequence[dict[str, Any]], query_indices: Sequence[int],
    model_indices: Sequence[int], timesteps: Sequence[int],
) -> list[dict[str, Any]]:
    lookup = {(int(row["evaluation_query_index"]), int(row["random_model_index"]),
               int(row["step"])): row for row in rows}
    output = []
    for query in query_indices:
        endpoint = np.asarray([float(lookup[(query, model, 0)]["endpoint_mse"])
                               for model in model_indices])
        trajectory = np.asarray([float(lookup[(query, model, 0)]["trajectory_mse"])
                                 for model in model_indices])
        for step, timestep in enumerate(timesteps):
            noise = np.asarray([float(lookup[(query, model, step)]["predicted_noise_mse"])
                                for model in model_indices])
            end, traj = correlation(noise, endpoint), correlation(noise, trajectory)
            output.append({
                "evaluation_query_index": query, "step": step, "timestep": timestep,
                "models": len(model_indices),
                "endpoint_pearson": end["pearson"], "endpoint_pearson_p": end["pearson_p"],
                "endpoint_spearman": end["spearman"], "endpoint_spearman_p": end["spearman_p"],
                "trajectory_pearson": traj["pearson"], "trajectory_pearson_p": traj["pearson_p"],
                "trajectory_spearman": traj["spearman"], "trajectory_spearman_p": traj["spearman_p"],
            })
    return output


def write_outputs(output_dir: Path, rows: list[dict[str, Any]],
                  query_indices: Sequence[int], model_indices: Sequence[int],
                  timesteps: Sequence[int]) -> None:
    correlations = compute_correlation_rows(rows, query_indices, model_indices, timesteps)
    path = output_dir / "correlation_per_query_by_timestep.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(correlations[0]))
        writer.writeheader()
        writer.writerows(correlations)

    summaries = []
    names = ("endpoint_pearson", "endpoint_spearman",
             "trajectory_pearson", "trajectory_spearman")
    for step, timestep in enumerate(timesteps):
        selected = [row for row in correlations if int(row["step"]) == step]
        summary: dict[str, Any] = {"step": step, "timestep": timestep,
                                  "queries": len(query_indices),
                                  "models_per_query": len(model_indices)}
        for name in names:
            values = np.asarray([float(row[name]) for row in selected])
            summary[f"{name}_mean"] = float(np.nanmean(values))
            summary[f"{name}_median"] = float(np.nanmedian(values))
        summaries.append(summary)
    path = output_dir / "correlation_summary_by_timestep.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    lines = ["ARTBENCH RANDOM TOP-1000 — CROSS-MODEL NOISE CORRELATION", "=" * 96,
             f"queries={len(query_indices)} models_per_query={len(model_indices)} "
             f"pairs={len(query_indices) * len(model_indices)}",
             "Each correlation is within one evaluation query across random models.", "",
             f"{'STEP':>4} {'TIME':>5} {'END-P':>11} {'END-S':>11} {'TRAJ-P':>11} {'TRAJ-S':>11}",
             "-" * 70]
    for row in summaries:
        lines.append(f"{row['step']:>4d} {row['timestep']:>5d} "
                     f"{row['endpoint_pearson_mean']:>+11.6f} "
                     f"{row['endpoint_spearman_mean']:>+11.6f} "
                     f"{row['trajectory_pearson_mean']:>+11.6f} "
                     f"{row['trajectory_spearman_mean']:>+11.6f}")
    (output_dir / "correlation_report.txt").write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    x = np.asarray(timesteps)
    for axis, prefix, title in ((axes[0], "endpoint", "Endpoint difference"),
                                (axes[1], "trajectory", "Trajectory difference")):
        axis.plot(x, [row[f"{prefix}_pearson_mean"] for row in summaries],
                  label="Mean Pearson", color="#4C78A8")
        axis.plot(x, [row[f"{prefix}_spearman_mean"] for row in summaries],
                  label="Mean Spearman", color="#F2A65A")
        axis.axhline(0, color="#777777", linewidth=.8)
        axis.set(title=title, xlabel="Diffusion timestep")
        axis.grid(axis="y", alpha=.22)
        axis.invert_xaxis()
    axes[0].set_ylabel("Mean within-query correlation across random models")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "correlation_summary_by_timestep.png", dpi=220)
    figure.savefig(output_dir / "correlation_summary_by_timestep.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("predicted-noise evaluation requires CUDA")
    records = load_queries(args.query_manifest, args.query_indices)
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False,
    )
    training_timesteps = int(pipeline.scheduler.config.num_train_timesteps)
    timesteps = expected_timesteps(training_timesteps, args.steps)
    stride = training_timesteps // args.steps
    pipeline.scheduler = DDIMScheduler.from_config(
        pipeline.scheduler.config, steps_offset=stride - 1)
    pipeline = pipeline.to(torch.device("cuda"))
    pipeline.set_progress_bar_config(disable=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = args.output_dir / "cross_model_per_query_timestep.csv"
    rows: list[dict[str, Any]] = [dict(row) for row in read_rows(pair_path)]
    complete = completed_pairs(rows, args.steps)
    pair_total, pair_position = len(records) * len(args.model_indices), 0
    for record in records:
        query = int(record["query_index"])
        pending = [model for model in args.model_indices if (query, model) not in complete]
        if not pending:
            pair_position += len(args.model_indices)
            print(f"[skip query] Q{query} all models complete", flush=True)
            continue
        print(f"[reference] evaluation Q{query}; pending_models={len(pending)}", flush=True)
        load_lora(pipeline, args.full_lora_dir)
        reference_image, reference_trajectory, reference_inputs, reference_noise = (
            capture_full_data_noise(
                pipeline, prompt=str(record["prompt"]), seed=int(record["seed"]),
                steps=args.steps, guidance_scale=args.guidance_scale))
        actual = [int(state["timestep"]) for state in reference_inputs]
        if actual != timesteps:
            raise RuntimeError(f"Q{query} schedule mismatch: {actual} vs {timesteps}")

        for model in args.model_indices:
            pair_position += 1
            if (query, model) in complete:
                print(f"[skip pair {pair_position}/{pair_total}] Q{query} R{model}", flush=True)
                continue
            print(f"[pair {pair_position}/{pair_total}] eval=Q{query} model=R{model}", flush=True)
            load_lora(pipeline, removal_lora(args.removal_root, args.method, model, args.top_k))
            candidate_image, candidate_trajectory = sample_with_trajectory(
                pipeline, prompt=str(record["prompt"]), seed=int(record["seed"]),
                steps=args.steps, guidance_scale=args.guidance_scale, timesteps=timesteps)
            endpoint_mse = mean_squared_error(image_array(reference_image),
                                              image_array(candidate_image))
            _, trajectory_mse, _ = trajectory_metrics(reference_trajectory,
                                                       candidate_trajectory)
            noise_mse = predicted_noise_mse_on_reference_inputs(
                pipeline, reference_inputs, reference_noise, args.guidance_scale)
            for step, (timestep, value) in enumerate(zip(timesteps, noise_mse)):
                rows.append({"evaluation_query_index": query,
                             "evaluation_seed": int(record["seed"]),
                             "random_model_index": model, "step": step,
                             "timestep": timestep, "predicted_noise_mse": float(value),
                             "endpoint_mse": endpoint_mse,
                             "trajectory_mse": trajectory_mse})
            write_pair_rows(pair_path, rows)
            complete.add((query, model))
            print(f"[saved] Q{query} R{model} endpoint={endpoint_mse:.8g} "
                  f"trajectory={trajectory_mse:.8g} noise_mean={noise_mse.mean():.8g}",
                  flush=True)

    expected = {(query, model) for query in args.query_indices for model in args.model_indices}
    missing = sorted(expected - completed_pairs(rows, args.steps))
    if missing:
        raise RuntimeError(f"incomplete query-model pairs: {missing}")
    write_outputs(args.output_dir, rows, args.query_indices, args.model_indices, timesteps)
    metadata = {
        "query_manifest": str(args.query_manifest.resolve()),
        "full_lora_dir": str(args.full_lora_dir.resolve()),
        "removal_root": str(args.removal_root.resolve()), "method": args.method,
        "evaluation_query_indices": args.query_indices,
        "random_model_indices": args.model_indices,
        "query_model_pairs": len(args.query_indices) * len(args.model_indices),
        "top_k": args.top_k, "steps": args.steps, "timesteps": timesteps,
        "guidance_scale": args.guidance_scale,
        "correlation_unit": "within each evaluation query across random models",
        "predicted_noise_definition": "guided UNet output MSE on identical scaled full-data DDIM trajectory inputs",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[done] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
