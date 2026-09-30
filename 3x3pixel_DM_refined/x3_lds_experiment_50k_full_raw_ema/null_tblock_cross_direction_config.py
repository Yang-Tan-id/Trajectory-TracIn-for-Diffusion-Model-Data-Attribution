"""Configuration for independent null updates over four timestamp blocks."""

import numpy as np

from exp_config import *
from null_same_direction_learning_config import (
    NSDL_DATAPOINT_COUNT,
    NSDL_DIRECTION_SEED_BASE,
    NSDL_EPS,
    NSDL_NULL_EPOCH,
    NSDL_UPDATE_BATCH_SIZE,
    nsdl_datapoint_indices,
)


NTCD_TARGET_SEED_BASE = 981100
NTCD_TARGET_DIRECTION_SEED_BASE = 981200
NTCD_TARGET_DIRECTION_COUNT = 5
NTCD_EVAL_TIMESTAMP_BATCH = 250
NTCD_TIMESTAMP_BLOCKS = tuple(
    tuple(range(start, start + NSDL_UPDATE_BATCH_SIZE))
    for start in range(0, T, NSDL_UPDATE_BATCH_SIZE)
)
NTCD_ROOT = ROOT / "null_tblock_cross_direction_10source_40models_5targetdirs"
NTCD_SOURCE_DIR = NTCD_ROOT / "sources"
NTCD_LOG_DIR = NTCD_ROOT / "logs"
NTCD_SUMMARY_PATH = NTCD_ROOT / "summary.json"
NTCD_OPTIMIZER_MODES = (
    "restored_adamw",
    "fresh_sgd",
    "fresh_adamw",
    "zero_grad_restored_adamw",
)
NTCD_CONTROL_ROOT = ROOT / "null_tblock_cross_direction_optimizer_controls"


def ntcd_target_index(source_index):
    generator = np.random.default_rng(NTCD_TARGET_SEED_BASE + int(source_index))
    target = int(generator.integers(0, N_TRAIN - 1))
    return target + int(target >= int(source_index))


def ntcd_mode_root(optimizer_mode="restored_adamw"):
    if optimizer_mode not in NTCD_OPTIMIZER_MODES:
        raise ValueError(optimizer_mode)
    return (
        NTCD_ROOT
        if optimizer_mode == "restored_adamw"
        else NTCD_CONTROL_ROOT / optimizer_mode
    )


def ntcd_mode_log_dir(optimizer_mode="restored_adamw"):
    return ntcd_mode_root(optimizer_mode) / "logs"


def ntcd_mode_summary_path(optimizer_mode="restored_adamw"):
    return ntcd_mode_root(optimizer_mode) / "summary.json"


def ntcd_source_dir(source_index, optimizer_mode="restored_adamw"):
    target_index = ntcd_target_index(source_index)
    return (
        ntcd_mode_root(optimizer_mode)
        / "sources"
        / f"source_{source_index:05d}_target_{target_index:05d}"
    )
