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
NGCD_EPS = 1e-12


def ngcd_checkpoint_path(epoch):
    return (
        MODEL_DIR
        / "base"
        / "prompted"
        / f"epoch_{int(epoch):04d}.pt"
    )
