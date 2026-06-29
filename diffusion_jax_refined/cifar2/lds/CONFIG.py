from pathlib import Path
import os
import sys

DATASET_DIR = Path(__file__).resolve().parents[1]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from dataset_config import ATTRIBUTION_ROOT, CLASS_NAMES, DATASET_NAME, DATA_ROOT, EVAL_ROOT, EXPERIMENT_TAG, QUERY, REFERENCE_CKPT


def _range_suffix(part: str) -> str:
    start_end = part.strip().replace(":", "-").split("-")
    if len(start_end) != 2:
        raise ValueError("ATTRIBUTION_RANGES must look like '1-2500,2501-5000'.")
    return f"range_{int(start_end[0])}_{int(start_end[1])}"


def _split_list(text: str) -> list[str]:
    return [part for part in text.replace(",", " ").split() if part]


def _split_paths(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _attribution_result_dirs(algorithm: str) -> list[str]:
    explicit = os.environ.get("ATTRIBUTION_RESULT_DIRS")
    if explicit:
        return _split_paths(explicit)
    ranges = os.environ.get("ATTRIBUTION_RANGES") or os.environ.get("SCORE_INDEX_RANGES")
    if algorithm == "traj_tracin" and ranges:
        base = str(ATTRIBUTION_ROOT / algorithm)
        return [f"{base}_{_range_suffix(part)}" for part in _split_list(ranges)]
    return [str(ATTRIBUTION_ROOT / algorithm)]


ALGORITHM = os.environ.get("ALGORITHM", "das")
RESULT_DIRS = _attribution_result_dirs(ALGORITHM)
DEFAULT_TRAJECTORY_REDUCTION = "snapshot_mean" if ALGORITHM == "traj_tracin" else "sum"
COMMAND_CWD = "legacy_jax"
COMMANDS = {
    "lds": [
        "python",
        "LDS/DM_cifar_lds.py",
        "--score-file",
        ",".join(RESULT_DIRS),
        "--base-checkpoint",
        REFERENCE_CKPT,
        "--data-root",
        DATA_ROOT,
        "--class-names",
        ",".join(CLASS_NAMES),
        "--subset-size",
        os.environ.get("LDS_SUBSET_SIZE", "5000"),
        "--m",
        os.environ.get("LDS_M", "100"),
        "--subset-seed",
        os.environ.get("LDS_SUBSET_SEED", "0"),
        "--prompt",
        QUERY,
        "--target-function",
        os.environ.get("LDS_TARGET_FUNCTION", "noise_trajectory"),
        "--trajectory-reduction",
        os.environ.get("LDS_TRAJECTORY_REDUCTION", DEFAULT_TRAJECTORY_REDUCTION),
        "--prediction-sign",
        os.environ.get("LDS_PREDICTION_SIGN", "-1"),
        "--out-root",
        str(EVAL_ROOT / "lds" / ALGORITHM),
        "--run-name",
        f"{EXPERIMENT_TAG}_{DATASET_NAME}_{ALGORITHM}",
        "--epochs",
        os.environ.get("LDS_EPOCHS", "200"),
        "--prefer-device",
        os.environ.get("LDS_DEVICE", "gpu"),
        "--num-devices",
        os.environ.get("LDS_NUM_DEVICES", "1"),
        "--no-use-data-parallel",
        "--save-every-epochs",
        os.environ.get("LDS_SAVE_EVERY_EPOCHS", "200"),
        "--keep-last-k",
        os.environ.get("LDS_KEEP_LAST_K", "1"),
    ]
}
