from __future__ import annotations

import os
from pathlib import Path


DATASET_NAME = "celeba"
DATASET_DISPLAY_NAME = "CelebA 64x64 seed-42 training subset 10k"
TRAINING_MODULE_NAME = "DM__training_CELEBA_pixel"
IMAGE_SIZE = 64
CLASS_NAMES = ("face",)
NUM_CLASSES = 1

DATASET_DIR = Path(__file__).resolve().parent
REFINE_ROOT = DATASET_DIR.parent
REPO_ROOT = REFINE_ROOT.parent
LEGACY_JAX_ROOT = REFINE_ROOT / "legacy_jax"
DATASET_STORAGE_ROOT = REFINE_ROOT / "dataset" / DATASET_NAME
DATA_ROOT = str(Path(os.environ.get("CELEBA_DATA_ROOT", DATASET_STORAGE_ROOT / "10000")))
ATTRIBUTION_INDICES_PATH = Path(
    os.environ.get("ATTRIBUTION_INDICES_PATH", str(Path(DATA_ROOT) / "attribution_5k_indices.npy"))
)
ATTRIBUTION_MASK_PATH = Path(
    os.environ.get("ATTRIBUTION_MASK_PATH", str(Path(DATA_ROOT) / "attribution_5k_mask.npy"))
)

EXPERIMENT_TAG = os.environ.get("EXPERIMENT_TAG", "experiment1")
RESULT_ROOT = DATASET_DIR / "result" / EXPERIMENT_TAG
MODEL_ROOT = RESULT_ROOT / "model"
LDS_MODEL_ROOT = RESULT_ROOT / "lds_model"
ATTRIBUTION_ROOT = RESULT_ROOT / "attribution_score"
EVAL_ROOT = RESULT_ROOT / "eval"
UNPROMPTED_JAX_MODEL_ROOT = MODEL_ROOT / "unprompted_jax"
CHECKPOINT_DIR = str(UNPROMPTED_JAX_MODEL_ROOT)

TRAIN_SEED = int(os.environ.get("TRAIN_SEED", "42"))
JAX_EPOCHS = int(os.environ.get("JAX_EPOCHS", "200"))
UNPROMPTED_CKPT_STEM = f"seed_{TRAIN_SEED}_epoch_{JAX_EPOCHS:04d}"
UNPROMPTED_JAX_REFERENCE_CKPT = str(
    UNPROMPTED_JAX_MODEL_ROOT / f"{UNPROMPTED_CKPT_STEM}.ckpt"
)
REFERENCE_CKPT = UNPROMPTED_JAX_REFERENCE_CKPT

# This is deliberately the same pixel-model contract used by 3D Shapes.  The
# one-dimensional label stored in dataset.npz is only a format placeholder;
# training/run_training.py always passes --unconditional.
COMMON_CIFAR = {
    "task_type": "cifar10",
    "module_name": TRAINING_MODULE_NAME,
    "query": "unconditional",
    "seed": TRAIN_SEED,
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
    "random_subset": False,
    "topk": 5000,
    "use_tqdm": True,
}
