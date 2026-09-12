from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path


def _prompt_tag(prompt: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(prompt).replace(",", "__"))
    return re.sub(r"_+", "_", text).strip("_")[:160] or "empty"


def _float_list(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = os.environ.get(name)
    return default if not value else tuple(float(x) for x in value.replace(",", " ").split())


def _int_list(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.environ.get(name)
    return default if not value else tuple(int(x) for x in value.replace(",", " ").split())


def _uniform_timesteps(count: int, total: int = 1000) -> tuple[int, ...]:
    if count < 2:
        raise ValueError("count must be at least 2")
    return tuple(round(i * (total - 1) / (count - 1)) for i in range(count))


DATASET_NAME = "3dshapes"
DATASET_DISPLAY_NAME = "3D Shapes 64x64 balanced 20k"
TRAINING_MODULE_NAME = "DM__training_3DSHAPES_pixel"
IMAGE_SIZE = 64
SHAPE_NAMES = ("cube", "cylinder", "sphere", "capsule")
SHAPE_LABELS = tuple(f"shape_{name}" for name in SHAPE_NAMES)
FLOOR_HUE_LABELS = tuple(f"floor_hue_{i}" for i in range(10))
WALL_HUE_LABELS = tuple(f"wall_hue_{i}" for i in range(10))
OBJECT_HUE_LABELS = tuple(f"object_hue_{i}" for i in range(10))
CLASS_NAMES = SHAPE_LABELS + OBJECT_HUE_LABELS + WALL_HUE_LABELS + FLOOR_HUE_LABELS
NUM_CLASSES = len(CLASS_NAMES)

DATASET_DIR = Path(__file__).resolve().parent
REFINE_ROOT = DATASET_DIR.parent
REPO_ROOT = REFINE_ROOT.parent
LEGACY_JAX_ROOT = REFINE_ROOT / "legacy_jax"
DATASET_STORAGE_ROOT = REFINE_ROOT / "dataset" / DATASET_NAME
DATA_ROOT = str(Path(os.environ.get("THREEDSHAPES_DATA_ROOT", DATASET_STORAGE_ROOT / "20000")))
ATTRIBUTION_INDICES_PATH = Path(
    os.environ.get("ATTRIBUTION_INDICES_PATH", str(Path(DATA_ROOT) / "attribution_5k_indices.npy"))
)

EXPERIMENT_TAG = os.environ.get("EXPERIMENT_TAG", "experiment1")
RESULT_ROOT = DATASET_DIR / "result" / EXPERIMENT_TAG
MODEL_ROOT = RESULT_ROOT / "model"
LDS_MODEL_ROOT = RESULT_ROOT / "lds_model"
ATTRIBUTION_ROOT = RESULT_ROOT / "attribution_score"
EVAL_ROOT = RESULT_ROOT / "eval"
SAMPLE_ROOT = Path(os.environ.get("SAMPLE_ROOT", str(RESULT_ROOT / "sample")))
PROMPTED_JAX_MODEL_ROOT = MODEL_ROOT / "prompted_jax"
CHECKPOINT_DIR = str(PROMPTED_JAX_MODEL_ROOT)

TRAIN_SEED = int(os.environ.get("TRAIN_SEED", "42"))
JAX_EPOCHS = int(os.environ.get("JAX_EPOCHS", "200"))
PROMPTED_CKPT_STEM = f"seed_{TRAIN_SEED}_epoch_{JAX_EPOCHS:04d}"
REFERENCE_CKPT = str(PROMPTED_JAX_MODEL_ROOT / f"{PROMPTED_CKPT_STEM}.ckpt")
QUERY = os.environ.get("QUERY", "shape_cube,object_hue_0,wall_hue_0,floor_hue_0")
INITIAL_SEED = int(os.environ.get("INITIAL_SEED", os.environ.get("SAMPLE_SEED", "0")))
MODEL_MODE = os.environ.get("SAMPLE_MODEL_MODE", "prompted_solo")
ATTRIBUTION_SAMPLE_DIR = os.environ.get(
    "ATTRIBUTION_SAMPLE_DIR",
    str(
        SAMPLE_ROOT
        / "cifar"
        / f"prompt_{_prompt_tag(QUERY)}"
        / f"model_{MODEL_MODE}__ckpt_{Path(REFERENCE_CKPT).stem}"
    ),
)
ATTRIBUTION_RUN_ROOT = (
    ATTRIBUTION_ROOT
    / MODEL_MODE
    / f"train_seed_{TRAIN_SEED}"
    / f"query_{_prompt_tag(QUERY)}"
    / f"initial_seed_{INITIAL_SEED}"
)
EVAL_RUN_ROOT = (
    EVAL_ROOT / MODEL_MODE / f"query_{_prompt_tag(QUERY)}" / f"initial_seed_{INITIAL_SEED}"
)

DAS_DAMPING_SWEEP_VALUES = _float_list(
    "DAS_DAMPING_SWEEP_VALUES",
    (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000),
)

COMMON_CIFAR = {
    "task_type": "cifar10",  # selects the generic pixel-dataset adapter
    "module_name": TRAINING_MODULE_NAME,
    "query": QUERY,
    "seed": 42,
    "data_root": DATA_ROOT,
    "class_names": CLASS_NAMES,
    "model_type": "unet",
    "image_size": IMAGE_SIZE,
    "in_channels": 3,
    "num_classes": NUM_CLASSES,
    "cond_mode": "multi_hot",
    "prefer_device": "gpu",
    "use_bfloat16": False,
    "score_index_ranges": None,
    "score_index_base": 0,
    "max_train_points": 5000,
    "random_subset": True,
    "topk": 5000,
    "use_tqdm": True,
}

ATTRIBUTION_CONFIGS = {
    "das": {
        **COMMON_CIFAR,
        "baseline_dir": CHECKPOINT_DIR,
        "reference_ckpt": REFERENCE_CKPT,
        "parameter_source": "raw",
        "attribution_sample_dir": ATTRIBUTION_SAMPLE_DIR,
        "attribution_sample_seed": INITIAL_SEED,
        "attribution_sample_index": 0,
        "attribution_use_trajectory_endpoint": True,
        "timesteps_total": 1000,
        "ddim_steps": 1000,
        "timesteps": _int_list("DAS_TIMESTEPS", _uniform_timesteps(100)),
        "num_mc_noise": int(os.environ.get("DAS_NUM_MC_NOISE", "1")),
        "proj_dim": int(os.environ.get("DAS_PROJ_DIM", "4096")),
        "damping": float(os.environ.get("DAS_DAMPING", "2")),
        "damping_sweep_values": DAS_DAMPING_SWEEP_VALUES,
        "batch_size": int(os.environ.get("DAS_BATCH_SIZE", "64")),
        "use_batched_per_example_grads": True,
        "per_example_grad_batch_size": int(os.environ.get("DAS_GRAD_BATCH_SIZE", "8")),
        "use_sherman_morrison_denominator": True,
        "max_num_ckpts": 1,
    },
    "traj_tracin": {
        **COMMON_CIFAR,
        "checkpoint_dir": CHECKPOINT_DIR,
        "reference_ckpt": REFERENCE_CKPT,
        "query_objective": os.environ.get("TRAJ_QUERY_OBJECTIVE", "trajectory_next_checkpoint_noise_mse"),
        "parameter_source": os.environ.get("TRAJ_PARAMETER_SOURCE", "raw"),
        "attribution_sample_dir": ATTRIBUTION_SAMPLE_DIR,
        "attribution_sample_seed": INITIAL_SEED,
        "attribution_sample_index": 0,
        "use_saved_trajectory": True,
        "sync_config_from_checkpoint": True,
        "ddim_steps": 1000,
        "num_traj_snapshots": int(os.environ.get("TRAJ_NUM_SNAPSHOTS", "10")),
        "snapshot_chunk_size": int(os.environ.get("TRAJ_SNAPSHOT_CHUNK_SIZE", "8")),
        "train_mc_samples": int(os.environ.get("TRAJ_TRAIN_MC_SAMPLES", "10")),
        "tracin_use_learning_rate_weights": True,
        "tracin_lr_schedule": "cosine_warmup",
        "tracin_warmup_ratio": 0.1,
        "save_query_normalized_scores": True,
        "query_normalize_eps": 1e-8,
        "score_batch_size": int(os.environ.get("TRAJ_SCORE_BATCH_SIZE", "32")),
        "proj_dim": int(os.environ.get("TRAJ_TRACIN_PROJ_DIM", "4096")),
        "progress_every": 256,
    },
}


def attribution_config(algorithm: str) -> dict:
    return deepcopy(ATTRIBUTION_CONFIGS[algorithm])


def commands_for_algorithm(algorithm: str) -> dict[str, list[str]]:
    return {
        "train_datapoint_gradient": ["python", "01_train_datapoint_gradient.py"],
        "query_gradient": ["python", "02_query_gradient.py"],
        "score": ["python", "03_score.py"],
    }
