from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path


def _parse_score_index_ranges(default: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    text = os.environ.get("SCORE_INDEX_RANGES") or os.environ.get("ATTRIBUTION_RANGES")
    if not text:
        return default
    ranges = []
    for part in text.replace(",", " ").split():
        if not part.strip():
            continue
        start_end = part.replace(":", "-").split("-")
        if len(start_end) != 2:
            raise ValueError(
                "SCORE_INDEX_RANGES/ATTRIBUTION_RANGES must look like '1-2500,2501-5000'."
            )
        ranges.append((int(start_end[0]), int(start_end[1])))
    return tuple(ranges)


DATASET_NAME = "cifar2"
DATASET_DISPLAY_NAME = "CIFAR2 horse/automobile"
EXPERIMENTS = ("experiment1", "experiment2", "experiment3")
EXPERIMENT_TAG = os.environ.get("EXPERIMENT_TAG", "experiment1")

DATASET_DIR = Path(__file__).resolve().parent
REFINE_ROOT = DATASET_DIR.parent
REPO_ROOT = REFINE_ROOT.parent
LEGACY_JAX_ROOT = REFINE_ROOT / "legacy_jax"
DATASET_STORAGE_ROOT = REFINE_ROOT / "dataset" / DATASET_NAME
RESULT_ROOT = DATASET_DIR / "result" / EXPERIMENT_TAG

MODEL_ROOT = RESULT_ROOT / "model"
ATTRIBUTION_ROOT = RESULT_ROOT / "attribution_score"
EVAL_ROOT = RESULT_ROOT / "eval"
PROMPTED_JAX_MODEL_ROOT = MODEL_ROOT / "prompted_jax"
SAMPLING_ROOT = EVAL_ROOT / "sampling"

CLASS_NAMES = ("horse", "automobile")
QUERY = "horse,automobile"
DATA_ROOT = str(DATASET_STORAGE_ROOT / "cifar-10-batches-py")
HF_DATASET_ROOT = str(DATASET_STORAGE_ROOT / "hf_cifar10")
LDS_INDEX_ROOT = DATASET_STORAGE_ROOT / "indices" / "lds-val"
CHECKPOINT_DIR = str(PROMPTED_JAX_MODEL_ROOT)
REFERENCE_CKPT = str(PROMPTED_JAX_MODEL_ROOT / "seed_0_epoch_0200.ckpt")
ATTRIBUTION_SAMPLE_DIR = (
    str(SAMPLING_ROOT / "cifar" / "prompt_horse__automobile" / "model_prompted_jax__ckpt_seed_0_epoch_0200")
)
SCORE_INDEX_RANGES = _parse_score_index_ranges(((1, 10000),))


COMMON_CIFAR = {
    "task_type": "cifar10",
    "module_name": "DM__training_CIFAR10_pixel",
    "query": QUERY,
    "seed": 0,
    "data_root": DATA_ROOT,
    "class_names": CLASS_NAMES,
    "model_type": "unet",
    "image_size": 32,
    "in_channels": 3,
    "cond_mode": "multi_hot",
    "prefer_device": "gpu",
    "use_bfloat16": False,
    "score_index_ranges": SCORE_INDEX_RANGES,
    "score_index_base": 1,
    "max_train_points": 10000,
    "random_subset": False,
    "topk": 10000,
    "use_tqdm": True,
}


ATTRIBUTION_CONFIGS = {
    "das": {
        **COMMON_CIFAR,
        "baseline_dir": CHECKPOINT_DIR,
        "reference_ckpt": REFERENCE_CKPT,
        "attribution_sample_dir": ATTRIBUTION_SAMPLE_DIR,
        "attribution_sample_seed": 0,
        "attribution_sample_index": 0,
        "attribution_use_trajectory_endpoint": True,
        "timesteps_total": 1000,
        "ddim_steps": 1000,
        "timesteps": (0, 200, 400, 600, 800, 999),
        "num_mc_noise": 8,
        "proj_dim": 32768,
        "damping": 1e-3,
        "batch_size": 64,
        "max_num_ckpts": 1,
    },
    "traj_tracin": {
        **COMMON_CIFAR,
        "checkpoint_dir": CHECKPOINT_DIR,
        "attribution_sample_dir": ATTRIBUTION_SAMPLE_DIR,
        "attribution_sample_seed": 0,
        "attribution_sample_index": 0,
        "use_saved_trajectory": True,
        "sync_config_from_checkpoint": True,
        "ddim_steps": 1000,
        "num_traj_snapshots": 100,
        "train_mc_samples": 2,
        "m_proj": 2,
        "score_batch_size": 2,
        "progress_every": 512,
    },
    "dtrak": {
        **COMMON_CIFAR,
        "baseline_dir": CHECKPOINT_DIR,
        "reference_ckpt": REFERENCE_CKPT,
        "attribution_sample_dir": ATTRIBUTION_SAMPLE_DIR,
        "attribution_sample_seed": 0,
        "attribution_sample_index": 0,
        "attribution_use_trajectory_endpoint": True,
        "timesteps": 1000,
        "ddim_steps": 1000,
        "proj_dim": 4096,
        "damping": 1e-3,
        "num_samples": 1,
        "batch_size": 64,
        "train_expectation_samples": 8,
        "query_expectation_samples": 8,
    },
    "end_tracin": {
        **COMMON_CIFAR,
        "baseline_dir": CHECKPOINT_DIR,
        "reference_ckpt": REFERENCE_CKPT,
        "use_baseline_ckpts": True,
        "checkpoint_limit": -1,
        "attribution_sample_dir": ATTRIBUTION_SAMPLE_DIR,
        "attribution_sample_seed": 0,
        "attribution_sample_index": 0,
        "attribution_use_trajectory_endpoint": True,
        "timesteps": 1000,
        "ddim_steps": 1000,
        "endpoint_mc_samples": 8,
        "train_mc_samples": 8,
        "score_batch_size": 32,
    },
    "journey_trak": {
        **COMMON_CIFAR,
        "baseline_dir": CHECKPOINT_DIR,
        "reference_ckpt": REFERENCE_CKPT,
        "timesteps_total": 1000,
        "ddim_steps": 1000,
        "proj_dim": 4096,
        "damping": 1e-3,
        "num_samples": 1,
        "batch_size": 64,
        "train_expectation_samples": 8,
        "query_expectation_samples": 8,
        "num_query_traj_steps": 50,
        "max_num_ckpts": 1,
    },
}


def attribution_config(algorithm: str) -> dict:
    return deepcopy(ATTRIBUTION_CONFIGS[algorithm])


def diffusers_training_command(algorithm: str) -> list[str]:
    seed = os.environ.get("TRAIN_SEED", "0")
    subset_index = os.environ.get("SUBSET_INDEX", "0")
    num_processes = os.environ.get("NUM_PROCESSES", "1")
    gpu_ids = os.environ.get("GPU_IDS", "0")
    port = os.environ.get("MAIN_PROCESS_PORT", "29500")
    run_name = f"CIFAR2-unprompted-{algorithm}-sub{subset_index}-seed{seed}"
    return [
        "accelerate",
        "launch",
        f"--num_processes={num_processes}",
        f"--gpu_ids={gpu_ids}",
        f"--main_process_port={port}",
        str(REFINE_ROOT / "training" / "train_diffusers_unconditional.py"),
        f"--seed={seed}",
        "--logger=wandb",
        f"--wandb_name={run_name}",
        "--model_config_name_or_path=config.json",
        f"--dataset_name_or_path={HF_DATASET_ROOT}",
        f"--index_path={LDS_INDEX_ROOT / f'sub-idx-{subset_index}.pkl'}",
        "--dataloader_num_workers=8",
        "--resolution=32",
        "--center_crop",
        "--random_flip",
        "--train_batch_size=128",
        "--num_epochs=200",
        "--checkpointing_steps=100000",
        "--gradient_accumulation_steps=1",
        "--learning_rate=1e-4",
        "--adam_weight_decay=1e-6",
        "--save_images_epochs=100000",
        f"--save_path={MODEL_ROOT / algorithm / 'unprompted' / f'ddpm-sub-{subset_index}-{seed}'}",
    ]


def commands_for_algorithm(algorithm: str) -> dict[str, list[str]]:
    return {
        "training": diffusers_training_command(algorithm),
        "attribution": ["python", "run_attribution.py"],
        "eval": ["python", "run_eval.py"],
    }
