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
                "SCORE_INDEX_RANGES/ATTRIBUTION_RANGES must look like '1-10000,10001-20000'."
            )
        ranges.append((int(start_end[0]), int(start_end[1])))
    return tuple(ranges)


DATASET_NAME = "cifar10"
DATASET_DISPLAY_NAME = "CIFAR10"
EXPERIMENTS = ("experiment1", "experiment2", "experiment3")
EXPERIMENT_TAG = os.environ.get("EXPERIMENT_TAG", "experiment1")

DATASET_DIR = Path(__file__).resolve().parent
REFINE_ROOT = DATASET_DIR.parent
REPO_ROOT = REFINE_ROOT.parent
LEGACY_JAX_ROOT = REFINE_ROOT / "legacy_jax"
DATASET_STORAGE_ROOT = REFINE_ROOT / "dataset" / DATASET_NAME
RESULT_ROOT = DATASET_DIR / "result" / EXPERIMENT_TAG

MODEL_ROOT = RESULT_ROOT / "model"
LDS_MODEL_ROOT = RESULT_ROOT / "lds_model"
ATTRIBUTION_ROOT = RESULT_ROOT / "attribution_score"
EVAL_ROOT = RESULT_ROOT / "eval"
PROMPTED_JAX_MODEL_ROOT = MODEL_ROOT / "prompted_jax"
UNPROMPTED_JAX_MODEL_ROOT = MODEL_ROOT / "unprompted_jax"
SAMPLING_ROOT = EVAL_ROOT / "sampling"

QUERY = "truck"
DATA_ROOT = str(DATASET_STORAGE_ROOT / "cifar-10-batches-py")
HF_DATASET_ROOT = str(DATASET_STORAGE_ROOT / "hf_cifar10")
LDS_INDEX_ROOT = DATASET_STORAGE_ROOT / "indices" / "lds-val"
CHECKPOINT_DIR = str(PROMPTED_JAX_MODEL_ROOT)
REFERENCE_CKPT = str(PROMPTED_JAX_MODEL_ROOT / "seed_42_epoch_0200.ckpt")
ATTRIBUTION_SAMPLE_DIR = str(SAMPLING_ROOT / "cifar" / "prompt_truck" / "model_prompted_jax__ckpt_seed_42_epoch_0200")
UNPROMPTED_TRAIN_SEED = int(os.environ.get("TRAIN_SEED", "42"))
UNPROMPTED_EPOCHS = int(os.environ.get("JAX_EPOCHS", "200"))
UNPROMPTED_CKPT_STEM = f"seed_{UNPROMPTED_TRAIN_SEED}_epoch_{UNPROMPTED_EPOCHS:04d}"
UNPROMPTED_JAX_REFERENCE_CKPT = str(UNPROMPTED_JAX_MODEL_ROOT / f"{UNPROMPTED_CKPT_STEM}.ckpt")
UNPROMPTED_ATTRIBUTION_SAMPLE_DIR = str(
    SAMPLING_ROOT / "cifar" / "prompt_unconditional"
    / f"model_unprompted_jax__ckpt_{UNPROMPTED_CKPT_STEM}"
)
SCORE_INDEX_RANGES = _parse_score_index_ranges(((1, 50000),))


COMMON_CIFAR = {
    "task_type": "cifar10",
    "module_name": "DM__training_CIFAR10_pixel",
    "query": QUERY,
    "seed": 42,
    "data_root": DATA_ROOT,
    "class_names": None,
    "model_type": "unet",
    "image_size": 32,
    "in_channels": 3,
    "cond_mode": "multi_hot",
    "prefer_device": "gpu",
    "use_bfloat16": False,
    "score_index_ranges": SCORE_INDEX_RANGES,
    "score_index_base": 1,
    "max_train_points": 50000,
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
        "timesteps_total": 1000,
        "ddim_steps": 1000,
        "timesteps": (0, 200, 400, 600, 800, 999),
        "num_mc_noise": 8,
        "proj_dim": 4096,
        "damping": 1e-3,
        "batch_size": 64,
        "max_num_ckpts": 1,
    },
    "traj_tracin": {
        **COMMON_CIFAR,
        "checkpoint_dir": CHECKPOINT_DIR,
        "reference_ckpt": REFERENCE_CKPT,
        "attribution_sample_dir": ATTRIBUTION_SAMPLE_DIR,
        "attribution_sample_seed": 0,
        "attribution_sample_index": 0,
        "use_saved_trajectory": True,
        "ddim_steps": 1000,
        "num_traj_snapshots": 100,
        "train_mc_samples": 2,
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

def unprompted_attribution_config(algorithm: str) -> dict:
    config = attribution_config(algorithm)
    config.update(class_cond=False, query="unconditional")
    for key in ("baseline_dir", "checkpoint_dir"):
        if key in config:
            config[key] = str(UNPROMPTED_JAX_MODEL_ROOT)
    if "reference_ckpt" in config:
        config["reference_ckpt"] = UNPROMPTED_JAX_REFERENCE_CKPT
    if "attribution_sample_dir" in config:
        config["attribution_sample_dir"] = UNPROMPTED_ATTRIBUTION_SAMPLE_DIR
    return config


def unprompted_training_command() -> list[str]:
    return [
        "python",
        str(REFINE_ROOT / "common" / "prompted_jax_training.py"),
        str(DATASET_DIR / "dataset_config.py"),
        "--algorithm=shared",
        "--unconditional",
    ]


def commands_for_algorithm(algorithm: str) -> dict[str, list[str]]:
    return {
        "training": unprompted_training_command(),
        "attribution": ["python", "run_attribution.py"],
        "eval": ["python", "run_eval.py"],
    }
