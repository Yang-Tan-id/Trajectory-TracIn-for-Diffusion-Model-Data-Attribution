"""Configuration for four-step restored-AdamW fixed-point validation."""

from path_integrated_jvp_config import *
from null_same_direction_learning_config import (
    NSDL_POINT_DIR,
    NSDL_TIMESTAMPS,
    NSDL_UPDATE_BATCH_SIZE,
    NSDL_UPDATE_COUNT,
)


SEQ4_METHODS = (
    "saved_total_start_jvp_control",
    "replayed_total_start_jvp",
    "replayed_stepwise_start_jvp",
    "replayed_stepwise_trapezoid_jvp",
    "replayed_stepwise_gauss2_jvp",
)
SEQ4_DEFAULT_BATCH_SIZE = 256
SEQ4_ROOT = ROOT / "sequential4_restored_adamw_fixed_point_20dir_100t_float64"
SEQ4_SOURCE_DIR = SEQ4_ROOT / "sources"
SEQ4_LOG_DIR = SEQ4_ROOT / "logs"
SEQ4_SUMMARY_PATH = SEQ4_ROOT / "summary.json"


def seq4_source_dir(source_index):
    target_index = ntcd_target_index(source_index)
    return SEQ4_SOURCE_DIR / f"source_{source_index:05d}_target_{target_index:05d}"
