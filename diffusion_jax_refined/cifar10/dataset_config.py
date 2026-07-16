from __future__ import annotations

import os
import re
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


def _prompt_path_tag(prompt: str) -> str:
    text = str(prompt).strip().replace(",", "__")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")[:80] or "empty_prompt"


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
SAMPLE_ROOT = RESULT_ROOT / "sample"

TRAIN_SEED = int(os.environ.get("TRAIN_SEED", "42"))
JAX_EPOCHS = int(os.environ.get("JAX_EPOCHS", "200"))
QUERY = os.environ.get("QUERY", "truck")
INITIAL_SEED = int(os.environ.get("INITIAL_SEED", os.environ.get("SAMPLE_SEED", "0")))
ATTRIBUTION_SCORE_MODEL_MODE = os.environ.get(
    "ATTRIBUTION_SCORE_MODEL_MODE",
    os.environ.get("ATTRIBUTION_SAMPLE_MODEL_MODE", os.environ.get("SAMPLE_MODEL_MODE", "prompted_solo")),
)
UNPROMPTED_SCORE_MODEL_MODE = os.environ.get(
    "UNPROMPTED_SCORE_MODEL_MODE",
    os.environ.get("UNPROMPTED_SAMPLE_MODEL_MODE", "unprompted_solo"),
)
ATTRIBUTION_RUN_ROOT = (
    ATTRIBUTION_ROOT
    / ATTRIBUTION_SCORE_MODEL_MODE
    / f"train_seed_{TRAIN_SEED}"
    / f"query_{_prompt_path_tag(QUERY)}"
    / f"initial_seed_{INITIAL_SEED}"
)
EVAL_RUN_ROOT = (
    EVAL_ROOT
    / ATTRIBUTION_SCORE_MODEL_MODE
    / f"query_{_prompt_path_tag(QUERY)}"
    / f"initial_seed_{INITIAL_SEED}"
)
UNPROMPTED_ATTRIBUTION_RUN_ROOT = (
    ATTRIBUTION_ROOT
    / UNPROMPTED_SCORE_MODEL_MODE
    / f"train_seed_{TRAIN_SEED}"
    / "unprompted"
    / f"initial_seed_{INITIAL_SEED}"
)
UNPROMPTED_EVAL_RUN_ROOT = (
    EVAL_ROOT
    / UNPROMPTED_SCORE_MODEL_MODE
    / "unprompted"
    / f"initial_seed_{INITIAL_SEED}"
)
DATA_ROOT = str(DATASET_STORAGE_ROOT / "cifar-10-batches-py")
HF_DATASET_ROOT = str(DATASET_STORAGE_ROOT / "hf_cifar10")
LDS_INDEX_ROOT = DATASET_STORAGE_ROOT / "indices" / "lds-val"
CHECKPOINT_DIR = str(PROMPTED_JAX_MODEL_ROOT)
PROMPTED_CKPT_STEM = f"seed_{TRAIN_SEED}_epoch_{JAX_EPOCHS:04d}"
REFERENCE_CKPT = str(PROMPTED_JAX_MODEL_ROOT / f"{PROMPTED_CKPT_STEM}.ckpt")
ATTRIBUTION_SAMPLE_DIR = os.environ.get(
    "ATTRIBUTION_SAMPLE_DIR",
    str(
        SAMPLE_ROOT / "cifar" / f"prompt_{_prompt_path_tag(QUERY)}"
        / f"model_{os.environ.get('ATTRIBUTION_SAMPLE_MODEL_MODE', os.environ.get('SAMPLE_MODEL_MODE', 'prompted_solo'))}__ckpt_{Path(REFERENCE_CKPT).stem}"
    ),
)
UNPROMPTED_CKPT_STEM = PROMPTED_CKPT_STEM
UNPROMPTED_JAX_REFERENCE_CKPT = str(UNPROMPTED_JAX_MODEL_ROOT / f"{UNPROMPTED_CKPT_STEM}.ckpt")
UNPROMPTED_ATTRIBUTION_SAMPLE_DIR = str(
    SAMPLE_ROOT / "cifar" / "prompt_unconditional"
    / f"model_{os.environ.get('UNPROMPTED_SAMPLE_MODEL_MODE', 'unprompted_solo')}__ckpt_{UNPROMPTED_CKPT_STEM}"
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
        "attribution_sample_seed": INITIAL_SEED,
        "attribution_sample_index": 0,
        "timesteps_total": 1000,
        "ddim_steps": 1000,
        "timesteps": (0, 200, 400, 600, 800, 999),
        "num_mc_noise": 10,
        "proj_dim": 4096,
        "damping": 1e-3,
        "batch_size": 64,
        "use_batched_per_example_grads": os.environ.get("DAS_BATCHED", "1") not in ("0", "false", "False"),
        "per_example_grad_batch_size": int(os.environ.get("DAS_GRAD_BATCH_SIZE", "8")),
        "max_num_ckpts": 1,
    },
    "traj_tracin": {
        **COMMON_CIFAR,
        "checkpoint_dir": CHECKPOINT_DIR,
        "reference_ckpt": REFERENCE_CKPT,
        "attribution_sample_dir": ATTRIBUTION_SAMPLE_DIR,
        "attribution_sample_seed": INITIAL_SEED,
        "attribution_sample_index": 0,
        "use_saved_trajectory": True,
        "ddim_steps": 1000,
        "num_traj_snapshots": 100,
        "snapshot_chunk_size": int(os.environ.get("TRAJ_SNAPSHOT_CHUNK_SIZE", "8")),
        "train_mc_samples": 10,
        "score_batch_size": int(os.environ.get("TRAJ_SCORE_BATCH_SIZE", "16")),
        "progress_every": 512,
    },
    "dtrak": {
        **COMMON_CIFAR,
        "baseline_dir": CHECKPOINT_DIR,
        "reference_ckpt": REFERENCE_CKPT,
        "attribution_sample_dir": ATTRIBUTION_SAMPLE_DIR,
        "attribution_sample_seed": INITIAL_SEED,
        "attribution_sample_index": 0,
        "timesteps": 1000,
        "ddim_steps": 1000,
        "proj_dim": 4096,
        "damping": 1e-3,
        "num_samples": 1,
        "batch_size": int(os.environ.get("DTRAK_BATCH_SIZE", "64")),
        "train_expectation_samples": 10,
        "query_expectation_samples": 10,
    },
    "end_tracin": {
        **COMMON_CIFAR,
        "baseline_dir": CHECKPOINT_DIR,
        "reference_ckpt": REFERENCE_CKPT,
        "use_baseline_ckpts": True,
        "checkpoint_limit": -1,
        "attribution_sample_dir": ATTRIBUTION_SAMPLE_DIR,
        "attribution_sample_seed": INITIAL_SEED,
        "attribution_sample_index": 0,
        "timesteps": 1000,
        "ddim_steps": 1000,
        "endpoint_mc_samples": 10,
        "train_mc_samples": 10,
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
        "batch_size": int(os.environ.get("JOURNEY_BATCH_SIZE", "64")),
        "train_expectation_samples": 10,
        "query_expectation_samples": 10,
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
        "train_datapoint_gradient": ["python", "01_train_datapoint_gradient.py"],
        "query_gradient": ["python", "02_query_gradient.py"],
        "score": ["python", "03_score.py"],
    }
