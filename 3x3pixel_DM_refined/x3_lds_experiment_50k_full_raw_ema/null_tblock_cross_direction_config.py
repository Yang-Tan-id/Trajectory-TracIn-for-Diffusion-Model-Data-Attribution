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


def ntcd_target_index(source_index):
    generator = np.random.default_rng(NTCD_TARGET_SEED_BASE + int(source_index))
    target = int(generator.integers(0, N_TRAIN - 1))
    return target + int(target >= int(source_index))


def ntcd_source_dir(source_index):
    target_index = ntcd_target_index(source_index)
    return NTCD_SOURCE_DIR / f"source_{source_index:05d}_target_{target_index:05d}"
