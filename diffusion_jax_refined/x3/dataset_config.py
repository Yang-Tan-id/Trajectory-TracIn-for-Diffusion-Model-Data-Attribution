from __future__ import annotations

import csv
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
            raise ValueError("SCORE_INDEX_RANGES/ATTRIBUTION_RANGES must look like '1-2500,2501-5000'.")
        ranges.append((int(start_end[0]), int(start_end[1])))
    return tuple(ranges)


def _parse_float_list_env(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    text = os.environ.get(name)
    if not text:
        return default
    return tuple(float(part) for part in text.replace(",", " ").split() if part.strip())


def _parse_int_list_env(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    text = os.environ.get(name)
    if not text:
        return default
    return tuple(int(part) for part in text.replace(",", " ").split() if part.strip())


def _prompt_path_tag(prompt: str) -> str:
    text = str(prompt).strip().replace(",", "__")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")[:80] or "empty_prompt"


DATASET_NAME = "x3"
DATASET_DISPLAY_NAME = "3x3 color grid"
EXPERIMENTS = ("experiment1", "experiment2", "experiment3")
EXPERIMENT_TAG = os.environ.get("EXPERIMENT_TAG", "experiment1_67")
TRAJ_QUERY_OBJECTIVE = os.environ.get(
    "TRAJ_QUERY_OBJECTIVE",
    os.environ.get("QUERY_OBJECTIVE", "trajectory_noise_squared_deviation"),
)

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

CSV_PATH = str(Path(os.environ.get("X3_CSV_PATH", str(DATASET_STORAGE_ROOT / "databases" / "3x3_4342_100000.csv"))))
GRID_SIZE = int(os.environ.get("X3_GRID_SIZE", "3"))
FIXED_S = float(os.environ.get("X3_FIXED_S", "0.9"))
FIXED_V = float(os.environ.get("X3_FIXED_V", "0.9"))
LABEL_START = int(os.environ.get("X3_LABEL_START", str(1 + GRID_SIZE * GRID_SIZE)))


def _load_class_names() -> tuple[str, ...]:
    labels: set[str] = set()
    path = Path(CSV_PATH)
    if path.is_file():
        with path.open(newline="") as fh:
            for row in csv.reader(fh):
                if not row or row[0].lower() == "id":
                    continue
                for label in row[LABEL_START:]:
                    if label:
                        labels.add(label)
    return tuple(sorted(labels))


CLASS_NAMES = _load_class_names()
DAS_DAMPING_SWEEP_VALUES = _parse_float_list_env(
    "DAS_DAMPING_SWEEP_VALUES",
    (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0),
)

TRAIN_SEED = int(os.environ.get("TRAIN_SEED", "67"))
JAX_EPOCHS = int(os.environ.get("JAX_EPOCHS", "200"))
QUERY = os.environ.get("QUERY", CLASS_NAMES[0] if CLASS_NAMES else "label_0")
INITIAL_SEED = int(os.environ.get("INITIAL_SEED", os.environ.get("SAMPLE_SEED", "0")))
ATTRIBUTION_SAMPLE_INDEX = int(os.environ.get("ATTRIBUTION_SAMPLE_INDEX", os.environ.get("SAMPLE_INDEX", "0")))
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
EVAL_RUN_ROOT = EVAL_ROOT / ATTRIBUTION_SCORE_MODEL_MODE / f"query_{_prompt_path_tag(QUERY)}" / f"initial_seed_{INITIAL_SEED}"
UNPROMPTED_ATTRIBUTION_RUN_ROOT = (
    ATTRIBUTION_ROOT / UNPROMPTED_SCORE_MODEL_MODE / f"train_seed_{TRAIN_SEED}" / "unprompted" / f"initial_seed_{INITIAL_SEED}"
)
UNPROMPTED_EVAL_RUN_ROOT = EVAL_ROOT / UNPROMPTED_SCORE_MODEL_MODE / "unprompted" / f"initial_seed_{INITIAL_SEED}"

DATA_ROOT = CSV_PATH
LDS_INDEX_ROOT = DATASET_STORAGE_ROOT / "indices" / "lds-val"
CHECKPOINT_DIR = str(PROMPTED_JAX_MODEL_ROOT)
PROMPTED_CKPT_STEM = f"seed_{TRAIN_SEED}_epoch_{JAX_EPOCHS:04d}"
REFERENCE_CKPT = str(PROMPTED_JAX_MODEL_ROOT / f"{PROMPTED_CKPT_STEM}.ckpt")
UNPROMPTED_CKPT_STEM = PROMPTED_CKPT_STEM
UNPROMPTED_JAX_REFERENCE_CKPT = str(UNPROMPTED_JAX_MODEL_ROOT / f"{UNPROMPTED_CKPT_STEM}.ckpt")

ATTRIBUTION_SAMPLE_DIR = os.environ.get(
    "ATTRIBUTION_SAMPLE_DIR",
    str(
        SAMPLE_ROOT
        / "x3"
        / f"prompt_{_prompt_path_tag(QUERY)}"
        / f"model_{os.environ.get('ATTRIBUTION_SAMPLE_MODEL_MODE', os.environ.get('SAMPLE_MODEL_MODE', 'prompted_solo'))}__ckpt_{Path(REFERENCE_CKPT).stem}"
    ),
)
UNPROMPTED_ATTRIBUTION_SAMPLE_DIR = str(
    SAMPLE_ROOT
    / "x3"
    / "prompt_unconditional"
    / f"model_{os.environ.get('UNPROMPTED_SAMPLE_MODEL_MODE', 'unprompted_solo')}__ckpt_{UNPROMPTED_CKPT_STEM}"
)

SCORE_INDEX_RANGES = _parse_score_index_ranges(((1, 10000),))

COMMON_X3 = {
    "task_type": "x3",
    "module_name": "DM__training_x3_pixel",
    "query": QUERY,
    "seed": 42,
    "csv_path": CSV_PATH,
    "grid_size": GRID_SIZE,
    "fixed_s": FIXED_S,
    "fixed_v": FIXED_V,
    "label_start": LABEL_START,
    "row_indices": None,
    "subset_ranges": None,
    "model_type": "cnn",
    "image_size": GRID_SIZE,
    "in_channels": 3,
    "base_channels": int(os.environ.get("JAX_BASE_CHANNELS", "160")),
    "time_emb_dim": 128,
    "num_classes": len(CLASS_NAMES),
    "class_cond": True,
    "cond_mode": "multi_hot",
    "prefer_device": "gpu",
    "use_bfloat16": os.environ.get("JAX_BFLOAT16", "0") in ("1", "true", "True", "yes"),
    "score_index_ranges": SCORE_INDEX_RANGES,
    "score_index_base": 1,
    "max_train_points": 10000,
    "random_subset": False,
    "topk": 10000,
    "use_tqdm": True,
    "epochs": JAX_EPOCHS,
    "batch_size": int(os.environ.get("JAX_BATCH_SIZE", "128")),
    "learning_rate": float(os.environ.get("JAX_LEARNING_RATE", "2e-4")),
    "weight_decay": float(os.environ.get("JAX_WEIGHT_DECAY", "1e-4")),
    "grad_clip_norm": 1.0,
    "ema_decay": 0.999,
    "log_every": int(os.environ.get("JAX_LOG_EVERY", "20")),
    "dropout": float(os.environ.get("JAX_DROPOUT", "0.1")),
    "predict_x0": False,
}

ATTRIBUTION_CONFIGS = {
    "das": {
        **COMMON_X3,
        "baseline_dir": CHECKPOINT_DIR,
        "reference_ckpt": REFERENCE_CKPT,
        "attribution_sample_dir": ATTRIBUTION_SAMPLE_DIR,
        "attribution_sample_seed": INITIAL_SEED,
        "attribution_sample_index": ATTRIBUTION_SAMPLE_INDEX,
        "attribution_use_trajectory_endpoint": True,
        "timesteps_total": 1000,
        "ddim_steps": 1000,
        "timesteps": _parse_int_list_env("DAS_TIMESTEPS", (0, 111, 222, 333, 444, 555, 666, 777, 888, 999)),
        "num_mc_noise": int(os.environ.get("DAS_NUM_MC_NOISE", "10")),
        "proj_dim": int(os.environ.get("DAS_PROJ_DIM", "4096")),
        "damping": float(os.environ.get("DAS_DAMPING", "2")),
        "damping_sweep_values": DAS_DAMPING_SWEEP_VALUES,
        "batch_size": 64,
        "use_batched_per_example_grads": os.environ.get("DAS_BATCHED", "1") not in ("0", "false", "False"),
        "per_example_grad_batch_size": int(os.environ.get("DAS_GRAD_BATCH_SIZE", "8")),
        "use_sherman_morrison_denominator": os.environ.get("DAS_SHERMAN_MORRISON_DENOMINATOR", "1")
        not in ("0", "false", "False"),
        "max_num_ckpts": 1,
    },
    "traj_tracin": {
        **COMMON_X3,
        "checkpoint_dir": CHECKPOINT_DIR,
        "reference_ckpt": REFERENCE_CKPT,
        "query_objective": TRAJ_QUERY_OBJECTIVE,
        "attribution_sample_dir": ATTRIBUTION_SAMPLE_DIR,
        "attribution_sample_seed": INITIAL_SEED,
        "attribution_sample_index": ATTRIBUTION_SAMPLE_INDEX,
        "use_saved_trajectory": os.environ.get("TRAJ_USE_SAVED_TRAJECTORY", "1") not in ("0", "false", "False"),
        "sync_config_from_checkpoint": True,
        "ddim_steps": 1000,
        "num_traj_snapshots": 100,
        "snapshot_chunk_size": int(os.environ.get("TRAJ_SNAPSHOT_CHUNK_SIZE", "8")),
        "train_mc_samples": 10,
        "tracin_use_learning_rate_weights": os.environ.get("TRACIN_USE_LR_WEIGHTS", "1") not in ("0", "false", "False"),
        "tracin_lr_schedule": os.environ.get("TRACIN_LR_SCHEDULE", "cosine_warmup"),
        "tracin_warmup_ratio": float(os.environ.get("TRACIN_WARMUP_RATIO", "0.1")),
        "save_query_normalized_scores": False,
        "query_normalize_eps": float(os.environ.get("TRAJ_QUERY_NORMALIZE_EPS", "1e-8")),
        "score_batch_size": int(os.environ.get("TRAJ_SCORE_BATCH_SIZE", "32")),
        "progress_every": 512,
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
