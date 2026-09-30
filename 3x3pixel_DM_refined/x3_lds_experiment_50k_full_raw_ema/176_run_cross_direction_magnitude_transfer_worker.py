"""Measure finite update magnitudes on the trained axis and other axes."""

import argparse
import importlib
import json
import time

import numpy as np
import torch

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from null_same_direction_learning_config import nsdl_checkpoint_path
from cross_direction_magnitude_transfer_config import *


shared = importlib.import_module("148_run_null_same_direction_learning_worker")
tblock = importlib.import_module("167_run_null_tblock_cross_direction_worker")


@torch.no_grad()
def source_axis_deltas(
    baseline,
    updated,
    source_x0,
    source_condition,
    source_direction,
    schedule,
    device,
):
    directions = torch.stack((source_direction, -source_direction), dim=0)
    delta = tblock.prediction_delta_bank(
        baseline,
        updated,
        source_x0,
        source_condition,
        directions,
        schedule,
        device,
    )
    flat = delta.double().flatten(2)
    return delta, flat.norm(dim=2)


def run_source(source_index, dataset, checkpoint, schedule, device):
    target_index = ntcd_target_index(source_index)
    output_dir = cdmt_source_dir(source_index)
    result_path = output_dir / "result.json"
    if result_path.is_file():
        print(
            f"[gpu {device.index}] skip source={source_index} target={target_index}",
            flush=True,
        )
        return
    source_image, source_condition = dataset[int(source_index)]
    source_x0 = source_image.unsqueeze(0).to(device)
    source_condition = source_condition.unsqueeze(0).to(device)
    source_direction = shared.fixed_direction(source_index, device)
    baseline = shared.make_model(dataset, checkpoint["model_state"], device)
    fresh_dir = ntcd_source_dir(source_index, "fresh_sgd")
    with open(fresh_dir / "result.json") as handle:
        fresh_result = json.load(handle)
    with np.load(
        fresh_dir / "target_direction_prediction_deltas.npz",
        allow_pickle=False,
    ) as target_archive:
        target_l2_by_block = []
        for block_index in range(len(NTCD_TIMESTAMP_BLOCKS)):
            target_delta = target_archive[
                f"block_{block_index}_prediction_delta"
            ].astype(np.float64)
            target_l2_by_block.append(
                np.linalg.norm(target_delta.reshape(*target_delta.shape[:2], -1), axis=2)
            )
    started = time.perf_counter()
    saved = {"timestamps": np.arange(T, dtype=np.int64)}
    blocks = []
    for block_index, timestamps in enumerate(NTCD_TIMESTAMP_BLOCKS):
        updated_payload = torch.load(
            fresh_result["blocks"][block_index]["updated_model"],
            map_location="cpu",
            weights_only=False,
        )
        updated = shared.make_model(dataset, updated_payload["model_state"], device)
        _, source_l2 = source_axis_deltas(
            baseline,
            updated,
            source_x0,
            source_condition,
            source_direction,
            schedule,
            device,
        )
        source_l2 = source_l2.cpu().numpy()
        target_l2 = target_l2_by_block[block_index]
        target_mean_l2 = target_l2.mean(axis=0)
        original_mean = float(source_l2[0].mean())
        opposite_mean = float(source_l2[1].mean())
        target_direction_means = target_l2.mean(axis=1)
        target_mean = float(target_direction_means.mean())
        saved[f"block_{block_index}_source_plus_l2"] = source_l2[0]
        saved[f"block_{block_index}_source_minus_l2"] = source_l2[1]
        saved[f"block_{block_index}_target_l2"] = target_l2
        saved[f"block_{block_index}_target_mean_l2"] = target_mean_l2
        blocks.append(
            {
                "block_index": block_index,
                "timestamp_start": timestamps[0],
                "timestamp_end": timestamps[-1],
                "source_plus_l2_mean": original_mean,
                "source_minus_l2_mean": opposite_mean,
                "source_minus_over_plus": opposite_mean / max(original_mean, NSDL_EPS),
                "target_direction_l2_means": target_direction_means.tolist(),
                "target_l2_mean": target_mean,
                "target_over_source_plus": target_mean / max(original_mean, NSDL_EPS),
            }
        )
        print(
            f"[gpu {device.index}] source={source_index} block={block_index} "
            f"plus={original_mean:.6e} minus/plus={opposite_mean / max(original_mean, NSDL_EPS):.4f} "
            f"target/plus={target_mean / max(original_mean, NSDL_EPS):.4f}",
            flush=True,
        )
        del updated
        torch.cuda.empty_cache()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "magnitude_by_timestamp.npz", **saved)
    shared.atomic_json(
        result_path,
        {
            "source_datapoint_index": int(source_index),
            "target_datapoint_index": int(target_index),
            "optimizer_mode": "fresh_sgd",
            "original_axis": "source datapoint trained +epsilon axis",
            "opposite_axis": "same source datapoint at -epsilon",
            "other_axes": "different target datapoint, own prompt, five independent directions",
            "blocks": blocks,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    checkpoint = torch.load(
        nsdl_checkpoint_path(), map_location="cpu", weights_only=False
    )
    schedule = base.make_linear_schedule(T, device=device)
    for source_index in nsdl_datapoint_indices()[
        args.shard_index :: args.shard_count
    ]:
        run_source(source_index, dataset, checkpoint, schedule, device)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
