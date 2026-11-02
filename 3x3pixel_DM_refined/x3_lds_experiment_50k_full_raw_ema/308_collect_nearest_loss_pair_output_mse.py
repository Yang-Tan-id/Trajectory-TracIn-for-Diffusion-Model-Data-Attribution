"""Regenerate and compare the nearest-simple-loss model pair per query."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from exp_config import (
    BASE_CH,
    BASE_CSV,
    DDIM_STEPS,
    EPOCHS,
    LDS_DIR,
    MASK_DIR,
    MODEL_DIR,
    QUERY_DIR,
    T,
    TIME_DIM,
    TRAJ_SNAPSHOTS,
)


def model_path(mask_record: dict, family: str) -> Path:
    return (
        MODEL_DIR
        / "subsets"
        / f"seed_{int(mask_record['lds_seed']):02d}"
        / f"subset_{int(mask_record['subset_id']):02d}"
        / family
        / f"epoch_{EPOCHS:04d}.pt"
    )


def build_model(path: Path, source: str, device: torch.device, dataset) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = base.CondEpsModel(3, len(dataset.vocab), BASE_CH, TIME_DIM).to(device)
    key = "model_state" if source == "raw" else "ema_model_state"
    model.load_state_dict(checkpoint[key], strict=True)
    return model.eval()


def condition_for(record: dict, dataset, device: torch.device) -> torch.Tensor:
    condition = torch.zeros((1, len(dataset.vocab)), device=device)
    if record["family"] != "unprompted":
        for label in record.get("labels", []):
            condition[0, dataset.vocab[label]] = 1.0
    return condition


def nearest_pair(values: np.ndarray) -> tuple[int, int]:
    order = np.argsort(values, kind="stable")
    position = int(np.argmin(np.diff(values[order])))
    return int(order[position]), int(order[position + 1])


@torch.no_grad()
def generate_trajectory(
    model: torch.nn.Module,
    schedule,
    condition: torch.Tensor,
    x_t: torch.Tensor,
    device: torch.device,
) -> list[torch.Tensor]:
    save_steps = np.linspace(
        0, DDIM_STEPS - 1, TRAJ_SNAPSHOTS, dtype=int
    ).tolist()
    return base.ddim_sample(
        model=model,
        sched=schedule,
        cond=condition,
        shape=tuple(x_t.shape),
        seed=0,
        steps=DDIM_STEPS,
        eta=0.0,
        device=str(device),
        x_T=x_t,
        save_steps=save_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--query-shard-index", type=int, default=0)
    parser.add_argument("--query-shard-count", type=int, default=1)
    parser.add_argument("--sources", default="ema,raw")
    args = parser.parse_args()

    if not 0 <= args.query_shard_index < args.query_shard_count:
        raise ValueError("invalid query shard")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    schedule = base.make_linear_schedule(T, device=device)
    masks = json.loads((MASK_DIR / "manifest.json").read_text())
    queries = json.loads((QUERY_DIR / "manifest.json").read_text())
    sources = [value.strip() for value in args.sources.split(",") if value.strip()]

    losses = {}
    for source in sources:
        path = LDS_DIR / f"observed_simple_loss_{source}_100t_mc4.npy"
        if not path.is_file():
            path = LDS_DIR / f"observed_simple_loss_{source}.npy"
        losses[source] = np.load(path).astype(np.float64)
        if losses[source].shape != (len(queries), len(masks)):
            raise ValueError(f"unexpected shape at {path}: {losses[source].shape}")

    output_dir = LDS_DIR / "nearest_simple_loss_pair_output_mse_shards"
    output_dir.mkdir(parents=True, exist_ok=True)
    query_ids = list(range(args.query_shard_index, len(queries), args.query_shard_count))
    started = time.perf_counter()

    for query_position, query_id in enumerate(query_ids, start=1):
        output_path = output_dir / f"q{query_id:02d}.json"
        if output_path.is_file():
            print(f"[skip] {output_path}", flush=True)
            continue

        query = queries[query_id]
        query_dir = Path(query["dir"])
        reference_trajectory = np.load(query_dir / "trajectory_xt.npy")
        x_t = torch.from_numpy(reference_trajectory[0]).to(
            device=device, dtype=torch.float32
        )
        condition = condition_for(query, dataset, device)
        source_results = {}

        for source in sources:
            subset_a, subset_b = nearest_pair(losses[source][query_id])
            trajectories = []
            for subset_id in (subset_a, subset_b):
                path = model_path(masks[subset_id], query["family"])
                if not path.is_file():
                    raise FileNotFoundError(path)
                model = build_model(path, source, device, dataset)
                trajectories.append(
                    generate_trajectory(model, schedule, condition, x_t, device)
                )
                del model

            trajectory_mses = np.asarray(
                [
                    F.mse_loss(state_a, state_b).item()
                    for state_a, state_b in zip(trajectories[0], trajectories[1])
                ],
                dtype=np.float64,
            )
            source_results[source] = {
                "subset_a": subset_a,
                "subset_b": subset_b,
                "loss_a": float(losses[source][query_id, subset_a]),
                "loss_b": float(losses[source][query_id, subset_b]),
                "loss_gap": abs(
                    float(
                        losses[source][query_id, subset_a]
                        - losses[source][query_id, subset_b]
                    )
                ),
                "pairwise_endpoint_mse": float(trajectory_mses[-1]),
                "pairwise_trajectory_mse": float(np.mean(trajectory_mses)),
                "pairwise_trajectory_mse_per_snapshot": trajectory_mses.tolist(),
            }

        payload = {
            "query_id": query_id,
            "family": query["family"],
            "sources": source_results,
        }
        temporary_path = output_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
        temporary_path.replace(output_path)
        elapsed = time.perf_counter() - started
        eta = elapsed / query_position * (len(query_ids) - query_position)
        brief = " ".join(
            f"{source}:pair={result['subset_a']:03d}/{result['subset_b']:03d} "
            f"loss-gap={result['loss_gap']:.2e} end={result['pairwise_endpoint_mse']:.2e} "
            f"traj={result['pairwise_trajectory_mse']:.2e}"
            for source, result in source_results.items()
        )
        print(
            f"[nearest-pair gpu={args.gpu}] q{query_id:02d} {brief} "
            f"elapsed={elapsed / 3600:.2f}h eta={eta / 3600:.2f}h",
            flush=True,
        )


if __name__ == "__main__":
    main()
