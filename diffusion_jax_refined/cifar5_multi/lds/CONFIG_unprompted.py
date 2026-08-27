from pathlib import Path
import os
import sys

DATASET_DIR = Path(__file__).resolve().parents[1]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from dataset_config import (
    UNPROMPTED_ATTRIBUTION_RUN_ROOT,
    CLASS_NAMES,
    DATASET_NAME,
    DATA_ROOT,
    EXPERIMENT_TAG,
    SCORE_INDEX_RANGES,
    UNPROMPTED_ATTRIBUTION_SAMPLE_DIR,
    UNPROMPTED_JAX_REFERENCE_CKPT,
    UNPROMPTED_EVAL_RUN_ROOT,
    TRAINING_MODULE_NAME,
)


def _range_suffix(start: int, end: int) -> str:
    return f"range_{int(start)}_{int(end)}"


def _parse_ranges(text: str) -> list[tuple[int, int]]:
    ranges = []
    for part in text.replace(",", " ").split():
        start_end = part.strip().replace(":", "-").split("-")
        if len(start_end) != 2:
            raise ValueError(
                "ATTRIBUTION_RANGES must look like '1-2500,2501-5000'."
            )
        ranges.append((int(start_end[0]), int(start_end[1])))
    return ranges


def _split_paths(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _attribution_result_dirs(algorithm: str) -> list[str]:
    explicit = os.environ.get("ATTRIBUTION_RESULT_DIRS")
    if explicit:
        return _split_paths(explicit)
    ranges_text = os.environ.get("ATTRIBUTION_RANGES") or os.environ.get(
        "SCORE_INDEX_RANGES"
    )
    ranges = _parse_ranges(ranges_text) if ranges_text else list(SCORE_INDEX_RANGES)
    base = UNPROMPTED_ATTRIBUTION_RUN_ROOT / f"{algorithm}_unprompted"
    if not ranges:
        return [str(base)]
    return [
        str(base.with_name(f"{base.name}_{_range_suffix(start, end)}"))
        for start, end in ranges
    ]


ALGORITHM = os.environ.get("ALGORITHM", "das")
RESULT_DIRS = _attribution_result_dirs(ALGORITHM)
DEFAULT_TRAJECTORY_REDUCTION = "snapshot_mean"
COMMAND_CWD = "legacy_jax"
COMMANDS = {
    "lds": [
        "python",
        "LDS/DM_cifar_lds.py",
        "--score-file",
        ",".join(RESULT_DIRS),
        "--base-checkpoint",
        UNPROMPTED_JAX_REFERENCE_CKPT,
        "--code-file",
        f"{TRAINING_MODULE_NAME}.py",
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
        "unconditional",
        "--attribution-sample-dir",
        UNPROMPTED_ATTRIBUTION_SAMPLE_DIR,
        "--attribution-sample-seed",
        os.environ.get("LDS_SAMPLE_SEED", "1"),
        "--attribution-sample-index",
        os.environ.get("LDS_SAMPLE_INDEX", "0"),
        "--target-function",
        os.environ.get("LDS_TARGET_FUNCTION", "noise_trajectory"),
        "--trajectory-reduction",
        os.environ.get("LDS_TRAJECTORY_REDUCTION", DEFAULT_TRAJECTORY_REDUCTION),
        "--prediction-sign",
        os.environ.get("LDS_PREDICTION_SIGN", "-1"),
        "--out-root",
        str(UNPROMPTED_EVAL_RUN_ROOT / "lds_unprompted" / ALGORITHM),
        "--run-name",
        f"{EXPERIMENT_TAG}_{DATASET_NAME}_{ALGORITHM}_unprompted",
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
