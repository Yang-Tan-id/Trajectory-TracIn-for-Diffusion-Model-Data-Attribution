"""Configuration for cross-direction loss-gradient prediction at null."""

from exp_config import *
from null_same_direction_learning_config import (
    NSDL_DATAPOINT_COUNT,
    NSDL_DIRECTION_SEED_BASE,
    NSDL_TIMESTAMPS,
    NSDL_UPDATE_BATCH_SIZE,
    nsdl_datapoint_indices,
)


NGCD_NULL_EPOCH = BASE_SAVE_EVERY_EPOCHS
NGCD_NEXT_EPOCH = 2 * BASE_SAVE_EVERY_EPOCHS
NGCD_ROOT = ROOT / "null_gradient_cross_direction_next_checkpoint_10points"
NGCD_POINT_DIR = NGCD_ROOT / "points"
NGCD_LOG_DIR = NGCD_ROOT / "logs"
NGCD_SUMMARY_PATH = NGCD_ROOT / "summary.json"
NGCD_RANDOM_PROMPT_ROOT = (
    ROOT / "null_gradient_cross_direction_random_prompt_next_checkpoint_10points"
)
NGCD_RANDOM_PROMPT_SEED_BASE = 979100
NGCD_EPS = 1e-12


def ngcd_checkpoint_path(epoch):
    return (
        MODEL_DIR
        / "base"
        / "prompted"
        / f"epoch_{int(epoch):04d}.pt"
    )


def ngcd_output_paths(evaluation_prompt):
    if evaluation_prompt == "original":
        root = NGCD_ROOT
    elif evaluation_prompt == "random":
        root = NGCD_RANDOM_PROMPT_ROOT
    else:
        raise ValueError(evaluation_prompt)
    return root, root / "points", root / "logs", root / "summary.json"
