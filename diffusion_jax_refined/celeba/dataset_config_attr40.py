from __future__ import annotations

import os
import re
from pathlib import Path


def _prompt_path_tag(prompt: str) -> str:
    text = str(prompt).strip().replace(",", "__")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")[:80] or "empty_prompt"


ATTRIBUTES = (
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young",
)

DATASET_NAME = "celeba"
DATASET_DISPLAY_NAME = "CelebA 64x64 seed-42 training subset 10k, 40 attributes"
TRAINING_MODULE_NAME = "DM__training_CELEBA_pixel"
IMAGE_SIZE = 64
CLASS_NAMES = ATTRIBUTES
NUM_CLASSES = len(CLASS_NAMES)

DATASET_DIR = Path(__file__).resolve().parent
REFINE_ROOT = DATASET_DIR.parent
REPO_ROOT = REFINE_ROOT.parent
LEGACY_JAX_ROOT = REFINE_ROOT / "legacy_jax"
DATASET_STORAGE_ROOT = REFINE_ROOT / "dataset" / DATASET_NAME
DATA_ROOT = str(
    Path(os.environ.get("CELEBA_ATTR40_DATA_ROOT", DATASET_STORAGE_ROOT / "10000_attr40"))
)
ATTRIBUTION_INDICES_PATH = Path(
    os.environ.get("ATTRIBUTION_INDICES_PATH", str(Path(DATA_ROOT) / "attribution_5k_indices.npy"))
)
ATTRIBUTION_MASK_PATH = Path(
    os.environ.get("ATTRIBUTION_MASK_PATH", str(Path(DATA_ROOT) / "attribution_5k_mask.npy"))
)

EXPERIMENT_TAG = os.environ.get("EXPERIMENT_TAG", "attr40_experiment1")
RESULT_ROOT = DATASET_DIR / "result" / EXPERIMENT_TAG
MODEL_ROOT = RESULT_ROOT / "model"
LDS_MODEL_ROOT = RESULT_ROOT / "lds_model"
ATTRIBUTION_ROOT = RESULT_ROOT / "attribution_score"
EVAL_ROOT = RESULT_ROOT / "eval"
SAMPLE_ROOT = Path(os.environ.get("SAMPLE_ROOT", str(RESULT_ROOT / "sample_ddim_eta0_50")))
PROMPTED_JAX_MODEL_ROOT = MODEL_ROOT / "prompted_jax"
CHECKPOINT_DIR = str(PROMPTED_JAX_MODEL_ROOT)

TRAIN_SEED = int(os.environ.get("TRAIN_SEED", "42"))
JAX_EPOCHS = int(os.environ.get("JAX_EPOCHS", "200"))
PROMPTED_CKPT_STEM = f"seed_{TRAIN_SEED}_epoch_{JAX_EPOCHS:04d}"
REFERENCE_CKPT = str(PROMPTED_JAX_MODEL_ROOT / f"{PROMPTED_CKPT_STEM}.ckpt")
QUERY = os.environ.get("QUERY", "Smiling,Young")
INITIAL_SEED = int(os.environ.get("INITIAL_SEED", os.environ.get("SAMPLE_SEED", "0")))
SAMPLE_MODEL_MODE = os.environ.get("SAMPLE_MODEL_MODE", "prompted_solo")
ATTRIBUTION_SCORE_MODEL_MODE = os.environ.get(
    "ATTRIBUTION_SCORE_MODEL_MODE",
    os.environ.get("ATTRIBUTION_SAMPLE_MODEL_MODE", SAMPLE_MODEL_MODE),
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
ATTRIBUTION_SAMPLE_DIR = os.environ.get(
    "ATTRIBUTION_SAMPLE_DIR",
    str(
        SAMPLE_ROOT
        / "cifar"
        / f"prompt_{_prompt_path_tag(QUERY)}"
        / f"model_{os.environ.get('ATTRIBUTION_SAMPLE_MODEL_MODE', SAMPLE_MODEL_MODE)}__ckpt_{Path(REFERENCE_CKPT).stem}"
    ),
)

COMMON_CIFAR = {
    "task_type": "cifar10",
    "module_name": TRAINING_MODULE_NAME,
    "query": QUERY,
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
