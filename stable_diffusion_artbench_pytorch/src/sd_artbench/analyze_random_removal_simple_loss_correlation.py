"""Relate ArtBench query simple-loss changes to random-removal output changes."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from diffusers import DDPMScheduler, StableDiffusionPipeline
from PIL import Image
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
FIELDS = (
    "evaluation_query_index", "evaluation_seed", "random_model_index",
    "step", "timestep", "full_simple_loss", "removed_simple_loss",
    "signed_loss_change", "absolute_loss_change", "endpoint_mse",
    "trajectory_mse",
)


def parse_args() -> argparse.Namespace:
    root = ArtBenchConfig().index_root.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-manifest", type=Path,
        default=root / "outputs/artbench2-queries-60/manifest.json",
    )
    parser.add_argument(
        "--full-lora-dir", type=Path,
        default=root / "outputs/artbench2-lora/final",
    )
    parser.add_argument("--removal-root", type=Path, required=True)
    parser.add_argument(
        "--noise-correlation-root", type=Path,
        default=root / "outputs/random-removal-noise-correlation-cross-q0-59",
        help="Existing cross-model result containing endpoint/trajectory MSE.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "outputs/random-removal-simple-loss-correlation-q0-9",
    )
    parser.add_argument("--method", default=RANDOM_METHOD)
    parser.add_argument("--query-indices", type=parse_csv_ints, default=list(range(10)))
    parser.add_argument("--model-indices", type=parse_csv_ints, default=list(range(60)))
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--probe-seed", type=int, default=2026)
    parser.add_argument("--model", default="lambdalabs/miniSD-diffusers")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_output_metrics(root: Path) -> dict[tuple[int, int], tuple[float, float]]:
    result: dict[tuple[int, int], tuple[float, float]] = {}
    paths = sorted(root.glob("**/cross_model_per_query_timestep.csv"))
    if not paths:
        raise FileNotFoundError(f"no cross-model CSV files under {root}")
    for path in paths:
        for row in read_rows(path):
            key = (int(row["evaluation_query_index"]), int(row["random_model_index"]))
            value = (float(row["endpoint_mse"]), float(row["trajectory_mse"]))
            previous = result.get(key)
            if previous is not None and previous != value:
                raise RuntimeError(f"conflicting endpoint/trajectory values for {key}")
            result[key] = value
    return result


def completed_pairs(rows: Sequence[dict[str, Any]], steps: int) -> set[tuple[int, int]]:
    grouped: dict[tuple[int, int], set[int]] = {}
    for row in rows:
        key = (int(row["evaluation_query_index"]), int(row["random_model_index"]))
        grouped.setdefault(key, set()).add(int(row["step"]))
    expected = set(range(steps))
    return {key for key, seen in grouped.items() if seen == expected}


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    key = lambda row: (
        int(row["evaluation_query_index"]), int(row["random_model_index"]),
        int(row["step"]),
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=key))
    temporary.replace(path)


def image_tensor(path: Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = 256.0 / min(width, height)
        resized = image.resize(
            (round(width * scale), round(height * scale)), Image.Resampling.BILINEAR
        )
        left = (resized.width - 256) // 2
        top = (resized.height - 256) // 2
        cropped = resized.crop((left, top, left + 256, top + 256))
        array = np.asarray(cropped, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device, dtype)


@torch.no_grad()
def build_shared_probes(
    pipeline: StableDiffusionPipeline, scheduler: DDPMScheduler,
    image_path: Path, prompt: str, timesteps: Sequence[int], seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device, dtype = pipeline.device, pipeline.unet.dtype
    pixels = image_tensor(image_path, device, pipeline.vae.dtype)
    latent_generator = torch.Generator(device=device).manual_seed(seed)
    latent = pipeline.vae.encode(pixels).latent_dist.sample(generator=latent_generator)
    latent = latent * pipeline.vae.config.scaling_factor

    input_ids = pipeline.tokenizer(
        prompt, max_length=pipeline.tokenizer.model_max_length,
        padding="max_length", truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    hidden = pipeline.text_encoder(input_ids)[0].to(dtype=dtype)

    cpu_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    noise_cpu = torch.randn(
        (len(timesteps),) + tuple(latent.shape[1:]), generator=cpu_generator,
        dtype=torch.float32,
    )
    noise = noise_cpu.to(device=device, dtype=dtype)
    t_values = torch.as_tensor(timesteps, device=device, dtype=torch.long)
    repeated_latent = latent.to(dtype=dtype).expand(len(timesteps), -1, -1, -1)
    noisy = scheduler.add_noise(repeated_latent, noise, t_values)
    hidden = hidden.expand(len(timesteps), -1, -1)
    if scheduler.config.prediction_type == "epsilon":
        target = noise
    elif scheduler.config.prediction_type == "v_prediction":
        target = scheduler.get_velocity(repeated_latent, noise, t_values)
    else:
        raise ValueError(f"unsupported prediction type: {scheduler.config.prediction_type}")
    return noisy, t_values, hidden, target


@torch.no_grad()
def simple_losses(
    pipeline: StableDiffusionPipeline, probes: tuple[torch.Tensor, ...],
    batch_size: int,
) -> np.ndarray:
    noisy, timesteps, hidden, target = probes
    values = []
    for start in range(0, len(timesteps), batch_size):
        stop = min(start + batch_size, len(timesteps))
        prediction = pipeline.unet(
            noisy[start:stop], timesteps[start:stop],
            encoder_hidden_states=hidden[start:stop],
        ).sample.float()
        losses = torch.mean(
            (prediction - target[start:stop].float()) ** 2,
            dim=tuple(range(1, prediction.ndim)),
        )
        values.extend(losses.cpu().tolist())
    return np.asarray(values, dtype=np.float64)


def correlation(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return {"pearson": float("nan"), "spearman": float("nan")}
    return {
        "pearson": float(pearsonr(left, right).statistic),
        "spearman": float(spearmanr(left, right).statistic),
    }


def summarize(
    output_dir: Path, rows: Sequence[dict[str, Any]],
    query_indices: Sequence[int], model_indices: Sequence[int],
    timesteps: Sequence[int],
) -> None:
    lookup = {
        (int(row["evaluation_query_index"]), int(row["random_model_index"]),
         int(row["step"])): row for row in rows
    }
    per_query = []
    per_timestep = []
    for query in query_indices:
        endpoint = np.asarray([
            float(lookup[(query, model, 0)]["endpoint_mse"]) for model in model_indices
        ])
        trajectory = np.asarray([
            float(lookup[(query, model, 0)]["trajectory_mse"]) for model in model_indices
        ])
        signed = np.asarray([
            np.mean([
                float(lookup[(query, model, step)]["signed_loss_change"])
                for step in range(len(timesteps))
            ]) for model in model_indices
        ])
        absolute = np.asarray([
            np.mean([
                float(lookup[(query, model, step)]["absolute_loss_change"])
                for step in range(len(timesteps))
            ]) for model in model_indices
        ])
        se, st = correlation(signed, endpoint), correlation(signed, trajectory)
        ae, at = correlation(absolute, endpoint), correlation(absolute, trajectory)
        per_query.append({
            "evaluation_query_index": query, "models": len(model_indices),
            "positive_signed_changes": int(np.sum(signed > 0)),
            "signed_endpoint_pearson": se["pearson"],
            "signed_endpoint_spearman": se["spearman"],
            "signed_trajectory_pearson": st["pearson"],
            "signed_trajectory_spearman": st["spearman"],
            "absolute_endpoint_pearson": ae["pearson"],
            "absolute_endpoint_spearman": ae["spearman"],
            "absolute_trajectory_pearson": at["pearson"],
            "absolute_trajectory_spearman": at["spearman"],
        })
        for step, timestep in enumerate(timesteps):
            step_signed = np.asarray([
                float(lookup[(query, model, step)]["signed_loss_change"])
                for model in model_indices
            ])
            step_absolute = np.abs(step_signed)
            se, st = correlation(step_signed, endpoint), correlation(step_signed, trajectory)
            ae, at = correlation(step_absolute, endpoint), correlation(step_absolute, trajectory)
            per_timestep.append({
                "evaluation_query_index": query, "step": step,
                "timestep": int(timestep), "models": len(model_indices),
                "signed_endpoint_pearson": se["pearson"],
                "signed_endpoint_spearman": se["spearman"],
                "signed_trajectory_pearson": st["pearson"],
                "signed_trajectory_spearman": st["spearman"],
                "absolute_endpoint_pearson": ae["pearson"],
                "absolute_endpoint_spearman": ae["spearman"],
                "absolute_trajectory_pearson": at["pearson"],
                "absolute_trajectory_spearman": at["spearman"],
            })

    def write_csv(path: Path, values: Sequence[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)

    write_csv(output_dir / "simple_loss_correlation_per_query.csv", per_query)
    write_csv(output_dir / "simple_loss_correlation_per_query_timestep.csv", per_timestep)
    metrics = [key for key in per_query[0] if key.endswith(("pearson", "spearman"))]
    lines = [
        "ARTBENCH RANDOM TOP-1000 — SIMPLE LOSS 50x1 CORRELATION",
        "=" * 145,
        f"queries={len(query_indices)} models_per_query={len(model_indices)} "
        f"pairs={len(query_indices) * len(model_indices)}",
        "50x1 uses the same 50 DDIM-aligned timesteps as the denoiser-deviation experiment, with one shared noise per timestep.",
        "Each correlation is within one query across random-removal models.", "",
        f"{'QUERY':<7}{'N':>5}{'POS':>8}  {'SGN END-P':>10}{'SGN END-S':>11}"
        f"{'SGN TRJ-P':>11}{'SGN TRJ-S':>11}  {'ABS END-P':>10}"
        f"{'ABS END-S':>11}{'ABS TRJ-P':>11}{'ABS TRJ-S':>11}",
        "-" * 145,
    ]
    for row in per_query:
        lines.append(
            f"Q{row['evaluation_query_index']:<6}{row['models']:>5}"
            f"{row['positive_signed_changes']:>5}/{row['models']:<2}  "
            f"{row['signed_endpoint_pearson']:>+10.4f}"
            f"{row['signed_endpoint_spearman']:>+11.4f}"
            f"{row['signed_trajectory_pearson']:>+11.4f}"
            f"{row['signed_trajectory_spearman']:>+11.4f}  "
            f"{row['absolute_endpoint_pearson']:>+10.4f}"
            f"{row['absolute_endpoint_spearman']:>+11.4f}"
            f"{row['absolute_trajectory_pearson']:>+11.4f}"
            f"{row['absolute_trajectory_spearman']:>+11.4f}"
        )
    lines.append("-" * 145)
    for label, reducer in (("MEAN", np.nanmean), ("STD", np.nanstd)):
        lines.append(
            f"{label:<20}" + "".join(
                f"{reducer([row[name] for row in per_query]):>+11.4f}"
                for name in metrics
            )
        )
    (output_dir / "simple_loss_correlation_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("simple-loss evaluation requires CUDA")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    records = load_queries(args.query_manifest, args.query_indices)
    output_metrics = load_output_metrics(args.noise_correlation_root)
    expected = {(query, model) for query in args.query_indices for model in args.model_indices}
    missing_metrics = sorted(expected - set(output_metrics))
    if missing_metrics:
        raise RuntimeError(f"missing endpoint/trajectory metrics: {missing_metrics}")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False,
    ).to(torch.device("cuda"))
    scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")
    timesteps = expected_timesteps(int(scheduler.config.num_train_timesteps), args.steps)
    pipeline.set_progress_bar_config(disable=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "simple_loss_per_query_model_timestep.csv"
    rows: list[dict[str, Any]] = [dict(row) for row in read_rows(data_path)]
    complete = completed_pairs(rows, args.steps)
    pair_total, pair_position = len(records) * len(args.model_indices), 0

    for record in records:
        query = int(record["query_index"])
        seed = int(record["seed"])
        probe_seed = args.probe_seed + 1009 * query
        probes = build_shared_probes(
            pipeline, scheduler, Path(record["file"]), str(record["prompt"]),
            timesteps, probe_seed,
        )
        load_lora(pipeline, args.full_lora_dir)
        full_losses = simple_losses(pipeline, probes, args.batch_size)
        print(f"[reference] Q{query} loss_mean={full_losses.mean():.8g}", flush=True)

        for model in args.model_indices:
            pair_position += 1
            if (query, model) in complete:
                print(f"[skip {pair_position}/{pair_total}] Q{query} R{model}", flush=True)
                continue
            load_lora(
                pipeline,
                removal_lora(args.removal_root, args.method, model, args.top_k),
            )
            removed_losses = simple_losses(pipeline, probes, args.batch_size)
            signed = removed_losses - full_losses
            endpoint_mse, trajectory_mse = output_metrics[(query, model)]
            for step, timestep in enumerate(timesteps):
                rows.append({
                    "evaluation_query_index": query, "evaluation_seed": seed,
                    "random_model_index": model, "step": step,
                    "timestep": int(timestep),
                    "full_simple_loss": float(full_losses[step]),
                    "removed_simple_loss": float(removed_losses[step]),
                    "signed_loss_change": float(signed[step]),
                    "absolute_loss_change": float(abs(signed[step])),
                    "endpoint_mse": endpoint_mse,
                    "trajectory_mse": trajectory_mse,
                })
            write_rows(data_path, rows)
            complete.add((query, model))
            print(
                f"[saved {pair_position}/{pair_total}] Q{query} R{model} "
                f"signed_mean={signed.mean():+.8g} abs_mean={np.abs(signed).mean():.8g}",
                flush=True,
            )
        del probes
        gc.collect()
        torch.cuda.empty_cache()

    missing = sorted(expected - completed_pairs(rows, args.steps))
    if missing:
        raise RuntimeError(f"incomplete pairs: {missing}")
    summarize(args.output_dir, rows, args.query_indices, args.model_indices, timesteps)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "query_manifest": str(args.query_manifest.resolve()),
        "full_lora_dir": str(args.full_lora_dir.resolve()),
        "removal_root": str(args.removal_root.resolve()),
        "noise_correlation_root": str(args.noise_correlation_root.resolve()),
        "query_indices": args.query_indices, "model_indices": args.model_indices,
        "steps": args.steps, "timesteps": timesteps,
        "probe_seed": args.probe_seed, "model": args.model,
        "definition": "conditional diffusion training simple loss on shared query-latent forward-noise probes",
    }, indent=2) + "\n", encoding="utf-8")
    print((args.output_dir / "simple_loss_correlation_report.txt").read_text())
    print(f"[done] {args.output_dir}")


if __name__ == "__main__":
    main()
