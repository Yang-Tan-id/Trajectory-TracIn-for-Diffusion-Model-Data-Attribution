from pathlib import Path
import os
import sys

DATASET_DIR = Path(__file__).resolve().parents[1]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from dataset_config import (
    ATTRIBUTION_RUN_ROOT,
    CLASS_NAMES,
    DATASET_NAME,
    DATA_ROOT,
    EVAL_RUN_ROOT,
    EXPERIMENT_TAG,
    QUERY,
    REFERENCE_CKPT,
)


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
        base = str(ATTRIBUTION_RUN_ROOT / algorithm)
        return [f"{base}_{_range_suffix(part)}" for part in _split_list(ranges)]
    matches = sorted(path for path in ATTRIBUTION_RUN_ROOT.glob(f"{algorithm}*") if path.is_dir())
    return [str(path) for path in matches] or [str(ATTRIBUTION_RUN_ROOT / algorithm)]


ALGORITHM = os.environ.get("ALGORITHM", "das")
TOPK = os.environ.get("TOPK", "5000")
RESULT_DIRS = _attribution_result_dirs(ALGORITHM)
COMMAND_CWD = "legacy_jax"
COMMANDS = {
    "counterfactual": [
        "python",
        "DM_counterfactual_retrain_from_attribution.py",
        "--result-dirs",
        *RESULT_DIRS,
        "--topk",
        TOPK,
        "--base-checkpoint",
        REFERENCE_CKPT,
        "--data-root",
        DATA_ROOT,
        "--class-names",
        ",".join(CLASS_NAMES),
        "--dataset-tag",
        DATASET_NAME,
        "--model-tag",
        f"{EXPERIMENT_TAG}_{ALGORITHM}",
        "--query",
        QUERY,
        "--score-tag",
        ALGORITHM,
        "--out-root",
        str(EVAL_RUN_ROOT / "counterfactual" / ALGORITHM),
    ]
}
