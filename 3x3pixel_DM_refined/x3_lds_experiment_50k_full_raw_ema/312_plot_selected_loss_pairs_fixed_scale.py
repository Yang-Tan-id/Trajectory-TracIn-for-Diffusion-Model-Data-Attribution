"""Plot selected nearest-loss LDS model pairs along their DDIM trajectories.

For each query, the main overview contains 15 snapshots for subset A, the
same 15 snapshots for subset B, and a shared-scale dark-to-bright absolute
difference heatmap.  A standalone difference strip is also saved.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
)


DEFAULT_PAIRS = {
    22: [(137, 138), (12, 188)],
    71: [(91, 154)],
    81: [(46, 179)],
    92: [(14, 183)],
}


def parse_pairs(text: str | None) -> dict[int, list[tuple[int, int]]]:
    if not text:
        return {query_id: list(pairs) for query_id, pairs in DEFAULT_PAIRS.items()}
    pairs: dict[int, list[tuple[int, int]]] = {}
    for item in text.split(","):
        query_id, subset_a, subset_b = (int(value) for value in item.split(":"))
        pairs.setdefault(query_id, []).append((subset_a, subset_b))
    return pairs


def parse_query_ids(text: str | None) -> list[int]:
    if not text:
        return list(DEFAULT_PAIRS)
    if "-" in text and "," not in text:
        lo, hi = (int(value) for value in text.split("-", 1))
        return list(range(lo, hi + 1))
    return [int(value) for value in text.split(",")]


def closest_loss_pairs(values: np.ndarray, count: int) -> list[tuple[int, int]]:
    rows, cols = np.triu_indices(values.size, k=1)
    gaps = np.abs(values[rows] - values[cols])
    order = np.argsort(gaps, kind="stable")[:count]
    return [(int(rows[index]), int(cols[index])) for index in order]


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


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    schedule,
    condition: torch.Tensor,
    x_t: torch.Tensor,
    device: torch.device,
    save_steps: list[int],
) -> list[torch.Tensor]:
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


def rgb_image(state: torch.Tensor) -> np.ndarray:
    return np.clip(
        state[0].detach().float().cpu().permute(1, 2, 0).numpy(), 0.0, 1.0
    )


def difference_image(state_a: torch.Tensor, state_b: torch.Tensor) -> np.ndarray:
    # One scalar per spatial pixel; averaging RGB gives a readable 3x3 heatmap.
    return (
        (state_a - state_b)
        .abs()[0]
        .mean(dim=0)
        .detach()
        .float()
        .cpu()
        .numpy()
    )


def style_axis(axis) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def save_plots(
    output_dir: Path,
    query_id: int,
    source: str,
    subset_a: int,
    subset_b: int,
    loss_a: float,
    loss_b: float,
    save_steps: list[int],
    trajectory_a: list[torch.Tensor],
    trajectory_b: list[torch.Tensor],
    difference_vmax: float,
    loss_rank: int | None = None,
) -> dict[str, float]:
    rgb_a = [rgb_image(state) for state in trajectory_a]
    rgb_b = [rgb_image(state) for state in trajectory_b]
    differences = [
        difference_image(state_a, state_b)
        for state_a, state_b in zip(trajectory_a, trajectory_b)
    ]
    per_step_mse = np.asarray(
        [
            F.mse_loss(state_a, state_b).item()
            for state_a, state_b in zip(trajectory_a, trajectory_b)
        ],
        dtype=np.float64,
    )
    difference_max = max(float(difference_vmax), np.finfo(np.float32).eps)
    diffusion_times = np.linspace(T - 1, 0, DDIM_STEPS).astype(int)[save_steps]

    figure, axes = plt.subplots(
        3,
        len(save_steps),
        figsize=(1.35 * len(save_steps), 5.0),
        gridspec_kw={"wspace": 0.05, "hspace": 0.12},
    )
    heatmap_handle = None
    for column, (image_a, image_b, difference, diffusion_t) in enumerate(
        zip(rgb_a, rgb_b, differences, diffusion_times)
    ):
        axes[0, column].imshow(image_a, interpolation="nearest")
        axes[1, column].imshow(image_b, interpolation="nearest")
        heatmap_handle = axes[2, column].imshow(
            difference,
            cmap="magma",
            vmin=0.0,
            vmax=difference_max,
            interpolation="nearest",
        )
        axes[0, column].set_title(f"t={diffusion_t}", fontsize=8)
        axes[2, column].set_xlabel(f"{per_step_mse[column]:.1e}", fontsize=7)
        for row in range(3):
            style_axis(axes[row, column])

    axes[0, 0].set_ylabel(
        f"subset {subset_a:03d}\nloss={loss_a:.8e}", fontsize=9, labelpad=12
    )
    axes[1, 0].set_ylabel(
        f"subset {subset_b:03d}\nloss={loss_b:.8e}", fontsize=9, labelpad=12
    )
    axes[2, 0].set_ylabel("|A-B|\nlabel=MSE", fontsize=9, labelpad=12)
    endpoint_mse = float(per_step_mse[-1])
    trajectory_mse = float(np.mean(per_step_mse))
    rank_text = f" · loss-pair rank={loss_rank}" if loss_rank is not None else ""
    figure.suptitle(
        f"q{query_id:02d} · {source.upper()}{rank_text} · "
        f"loss gap={abs(loss_a-loss_b):.3e} · "
        f"15-point trajectory MSE={trajectory_mse:.3e} · endpoint MSE={endpoint_mse:.3e}",
        fontsize=12,
        y=1.01,
    )
    if heatmap_handle is not None:
        colorbar = figure.colorbar(
            heatmap_handle, ax=axes[2, :].tolist(), fraction=0.012, pad=0.01
        )
        colorbar.set_label("mean absolute RGB difference", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    rank_tag = f"rank{loss_rank:02d}_" if loss_rank is not None else ""
    overview_path = output_dir / f"q{query_id:02d}_{source}_{rank_tag}pair_{subset_a:03d}_{subset_b:03d}_overview.png"
    figure.savefig(overview_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    difference_figure, difference_axes = plt.subplots(
        1, len(save_steps), figsize=(1.35 * len(save_steps), 1.8),
        gridspec_kw={"wspace": 0.05},
    )
    difference_handle = None
    for column, (difference, diffusion_t) in enumerate(zip(differences, diffusion_times)):
        difference_handle = difference_axes[column].imshow(
            difference,
            cmap="magma",
            vmin=0.0,
            vmax=difference_max,
            interpolation="nearest",
        )
        difference_axes[column].set_title(f"t={diffusion_t}", fontsize=8)
        difference_axes[column].set_xlabel(f"{per_step_mse[column]:.1e}", fontsize=7)
        style_axis(difference_axes[column])
    difference_figure.suptitle(
        f"q{query_id:02d} subset {subset_a:03d} vs {subset_b:03d}: "
        "dark = similar, bright = different",
        fontsize=11,
        y=1.05,
    )
    if difference_handle is not None:
        colorbar = difference_figure.colorbar(
            difference_handle,
            ax=difference_axes.tolist(),
            fraction=0.018,
            pad=0.01,
        )
        colorbar.set_label("mean |RGB difference|", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    difference_path = output_dir / f"q{query_id:02d}_{source}_{rank_tag}pair_{subset_a:03d}_{subset_b:03d}_difference.png"
    difference_figure.savefig(
        difference_path, dpi=220, bbox_inches="tight", facecolor="white"
    )
    plt.close(difference_figure)
    return {
        "loss_pair_rank": loss_rank,
        "loss_a": loss_a,
        "loss_b": loss_b,
        "loss_gap": abs(loss_a - loss_b),
        "endpoint_mse": endpoint_mse,
        "trajectory_15point_mse": trajectory_mse,
        "overview": str(overview_path),
        "difference": str(difference_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--source", choices=("ema", "raw"), default="ema")
    parser.add_argument(
        "--pairs",
        help="comma-separated query:subset_a:subset_b; duplicate query IDs are allowed",
    )
    parser.add_argument(
        "--query-ids",
        help="queries used with --top-loss-pairs; defaults to 7,30,67,71,72,82",
    )
    parser.add_argument(
        "--top-loss-pairs",
        type=int,
        default=0,
        help="plot this many closest-loss pairs per query instead of --pairs",
    )
    parser.add_argument("--snapshots", type=int, default=15)
    parser.add_argument(
        "--difference-vmax",
        type=float,
        default=0.04,
        help=(
            "fixed upper bound shared by every absolute-difference heatmap; "
            "values above it are saturated"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(LDS_DIR / "nearest_simple_loss_pair_trajectory_overviews"),
    )
    args = parser.parse_args()
    if args.snapshots < 2:
        raise ValueError("--snapshots must be at least 2")
    if args.difference_vmax <= 0:
        raise ValueError("--difference-vmax must be positive")
    if args.top_loss_pairs < 0:
        raise ValueError("--top-loss-pairs cannot be negative")
    if args.top_loss_pairs and args.pairs:
        raise ValueError("use either --top-loss-pairs or --pairs, not both")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    schedule = base.make_linear_schedule(T, device=device)
    masks = json.loads((MASK_DIR / "manifest.json").read_text())
    queries = json.loads((QUERY_DIR / "manifest.json").read_text())
    loss_path = LDS_DIR / f"observed_simple_loss_{args.source}_100t_mc4.npy"
    if not loss_path.is_file():
        loss_path = LDS_DIR / f"observed_simple_loss_{args.source}.npy"
    losses = np.load(loss_path).astype(np.float64)
    if args.top_loss_pairs:
        candidates = {
            query_id: closest_loss_pairs(losses[query_id], args.top_loss_pairs)
            for query_id in parse_query_ids(args.query_ids)
        }
    else:
        candidates = parse_pairs(args.pairs)
    save_steps = np.linspace(0, DDIM_STEPS - 1, args.snapshots, dtype=int).tolist()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    pending_plots = []

    for query_id, query_pairs in candidates.items():
        query = queries[query_id]
        query_dir = Path(query["dir"])
        reference_trajectory = np.load(query_dir / "trajectory_xt.npy")
        x_t = torch.from_numpy(reference_trajectory[0]).to(
            device=device, dtype=torch.float32
        )
        condition = condition_for(query, dataset, device)
        trajectory_cache = {}
        subset_ids = sorted({subset for pair in query_pairs for subset in pair})
        for subset_id in subset_ids:
            path = model_path(masks[subset_id], query["family"])
            if not path.is_file():
                raise FileNotFoundError(path)
            model = build_model(path, args.source, device, dataset)
            trajectory_cache[subset_id] = generate(
                model, schedule, condition, x_t, device, save_steps
            )
            del model
        for loss_rank, (subset_a, subset_b) in enumerate(query_pairs, start=1):
            pending_plots.append(
                {
                    "query_id": query_id,
                    "loss_rank": loss_rank if args.top_loss_pairs else None,
                    "subset_a": subset_a,
                    "subset_b": subset_b,
                    "loss_a": float(losses[query_id, subset_a]),
                    "loss_b": float(losses[query_id, subset_b]),
                    "trajectory_a": trajectory_cache[subset_a],
                    "trajectory_b": trajectory_cache[subset_b],
                }
            )
            print(
                f"[generated] q{query_id:02d} loss-rank={loss_rank} "
                f"pair={subset_a:03d}/{subset_b:03d}",
                flush=True,
            )

    # Use an absolute, user-specified scale rather than adapting to the data.
    # This makes colors comparable across queries and across separate runs.
    global_difference_vmax = float(args.difference_vmax)
    print(
        f"[fixed shared difference scale] vmin=0 "
        f"vmax={global_difference_vmax:.6e}",
        flush=True,
    )

    summary = {f"q{query_id:02d}": [] for query_id in candidates}
    table_rows = []
    for item in pending_plots:
        query_id = item["query_id"]
        subset_a = item["subset_a"]
        subset_b = item["subset_b"]
        result = save_plots(
            output_dir=output_dir,
            query_id=query_id,
            source=args.source,
            subset_a=subset_a,
            subset_b=subset_b,
            loss_a=item["loss_a"],
            loss_b=item["loss_b"],
            save_steps=save_steps,
            trajectory_a=item["trajectory_a"],
            trajectory_b=item["trajectory_b"],
            difference_vmax=global_difference_vmax,
            loss_rank=item["loss_rank"],
        )
        result["subset_a"] = subset_a
        result["subset_b"] = subset_b
        result["shared_difference_vmin"] = 0.0
        result["shared_difference_vmax"] = global_difference_vmax
        summary[f"q{query_id:02d}"].append(result)
        table_rows.append({"query_id": query_id, **result})
        print(
            f"[plotted] q{query_id:02d} pair={subset_a:03d}/{subset_b:03d} "
            f"loss-gap={result['loss_gap']:.3e} "
            f"endpoint={result['endpoint_mse']:.3e} "
            f"trajectory={result['trajectory_15point_mse']:.3e}",
            flush=True,
        )

    summary_path = output_dir / f"summary_{args.source}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    csv_path = output_dir / f"summary_{args.source}.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"[saved] {summary_path}")
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
