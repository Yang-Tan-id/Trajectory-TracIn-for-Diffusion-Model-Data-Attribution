#!/usr/bin/env python3
"""Define the 250 CelebA full-10k top-1000 jobs and staging manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CELEBA_ROOT = Path(__file__).resolve().parents[1]
if str(CELEBA_ROOT) not in sys.path:
    sys.path.insert(0, str(CELEBA_ROOT))

from dataset_config_attr40 import _prompt_path_tag


BOTH_L2_METHODS = (
    "endpoint_pollute_adamw_timestamp_both_l2_full10k_pos",
    "endpoint_tracin_adamw_linear_both_l2_full10k_neg",
    "retrac_adamw_query_train_l2_full10k_neg",
)
RAW_METHOD = "endpoint_tracin_adamw_linear_raw_full10k_neg"
OUTPUT_GROUP = "endpoint_top1000_removal_q0_69"

POLLUTE_NAMESPACE = (
    "celeba_endpoint_pollute_adamw_full_train1x50_query1x50_"
    "timestamp_sum_squared_shape3d_semantics_full10000"
)
TRACIN_NAMESPACE = (
    "celeba_endpoint_tracin_adamw_full_train_mean50_query_mean50_"
    "linear_shape3d_semantics_full10000"
)
RETRAC_NAMESPACE = (
    "celeba_paper_retrac_adamw_full_train_replay4_batch1_"
    "query_mean50_of_full_l2_linear_shape3d_semantics_full10000"
)


def build_tasks() -> list[tuple[str, int]]:
    tasks = [
        (method, query_id)
        for method in BOTH_L2_METHODS
        for query_id in range(10, 70)
    ]
    tasks.extend((RAW_METHOD, query_id) for query_id in range(70))
    if len(tasks) != 250 or len(set(tasks)) != 250:
        raise AssertionError("CelebA task plan must contain 250 unique jobs")
    return tasks


def load_queries(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text())
    records = payload["queries"] if isinstance(payload, dict) else payload
    return {
        int(record.get("query_id", record.get("id", position))): record
        for position, record in enumerate(records)
    }


def query_seed(record: dict) -> int:
    return int(record.get("initial_seed", record.get("seed")))


def score_relative_paths(query_file: Path) -> set[Path]:
    queries = load_queries(query_file)
    paths: set[Path] = set()
    score_root = Path(
        "diffusion_jax_refined/celeba/result/attr40_experiment1/"
        "attribution_score/prompted_solo/train_seed_42"
    )
    for query_id in range(70):
        record = queries[query_id]
        query_root = (
            score_root
            / f"query_{_prompt_path_tag(str(record['prompt']))}"
            / f"initial_seed_{query_seed(record)}"
        )
        directories = [(TRACIN_NAMESPACE, "score")]
        if query_id >= 10:
            directories.extend(
                (
                    (POLLUTE_NAMESPACE, "score_query_train_l2_normalized"),
                    (TRACIN_NAMESPACE, "score_query_train_l2_normalized"),
                    (RETRAC_NAMESPACE, "score_query_train_l2_normalized"),
                )
            )
        for namespace, score_directory in directories:
            directory = query_root / f"traj_tracin_{namespace}" / score_directory
            paths.add(directory / "scores.npy")
            paths.add(directory / "score_indices.npy")
            paths.add(directory / "manifest.json")
    return paths


def transfer_manifest(query_file: Path) -> list[Path]:
    queries = load_queries(query_file)
    paths = score_relative_paths(query_file)
    paths.update(
        {
            Path("diffusion_jax_refined/celeba/queries_attr40_val25_x4.json"),
            Path("diffusion_jax_refined/dataset/celeba/10000_attr40/dataset.npz"),
            Path(
                "diffusion_jax_refined/dataset/celeba/10000_attr40/"
                "attribution_5k_indices.npy"
            ),
            Path(
                "diffusion_jax_refined/dataset/celeba/10000_attr40/"
                "attribution_5k_mask.npy"
            ),
            Path(
                "diffusion_jax_refined/celeba/result/attr40_experiment1/model/"
                "prompted_jax/seed_42_epoch_0200.ckpt"
            ),
            Path(
                "diffusion_jax_refined/celeba/result/attr40_experiment1/"
                "query72_lds_csv_ddim50/celeba_attr40_query100_lds_true_f_long.csv"
            ),
        }
    )
    sample_root = Path(
        "diffusion_jax_refined/celeba/result/attr40_experiment1/"
        "sample_ddim_eta0_50/cifar"
    )
    for query_id in range(70):
        record = queries[query_id]
        seed_root = (
            sample_root
            / f"prompt_{_prompt_path_tag(str(record['prompt']))}"
            / "model_prompted_solo__ckpt_seed_42_epoch_0200"
            / f"seed_{query_seed(record):06d}"
        )
        paths.add(seed_root / "trajectory_xt.npy")
        paths.add(seed_root / "trajectory_t.npy")
    return sorted(paths, key=str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tasks", "transfer"), required=True)
    parser.add_argument(
        "--query-file",
        type=Path,
        default=CELEBA_ROOT / "queries_attr40_val25_x4.json",
    )
    args = parser.parse_args()
    if args.mode == "tasks":
        for method, query_id in build_tasks():
            print(f"{method}\t{query_id}")
    else:
        for path in transfer_manifest(args.query_file):
            print(path)


if __name__ == "__main__":
    main()
