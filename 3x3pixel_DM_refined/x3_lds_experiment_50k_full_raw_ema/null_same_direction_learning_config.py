"""Configuration for the null-model same-noise-direction learning probe."""

import numpy as np

from exp_config import *


NSDL_NULL_EPOCH = BASE_SAVE_EVERY_EPOCHS
NSDL_DATAPOINT_COUNT = 10
NSDL_DATAPOINT_SEED = 967100
NSDL_DIRECTION_SEED_BASE = 967200
NSDL_RANDOM_DIRECTION_SEED_BASE = 967300
NSDL_TIMESTAMPS = tuple(range(T - 1, -1, -1))
NSDL_UPDATE_COUNT = 4
NSDL_UPDATE_BATCH_SIZE = T // NSDL_UPDATE_COUNT
NSDL_RANDOM_DIRECTION_COUNT = 100
NSDL_EVAL_DIRECTION_BATCH = 8
NSDL_EPS = 1e-12
NSDL_ROOT = ROOT / "null_same_direction_learning_10points"
NSDL_POINT_DIR = NSDL_ROOT / "points"
NSDL_LOG_DIR = NSDL_ROOT / "logs"
NSDL_SUMMARY_PATH = NSDL_ROOT / "summary.json"


def nsdl_checkpoint_path():
    return (
        MODEL_DIR
        / "base"
        / "prompted"
        / f"epoch_{int(NSDL_NULL_EPOCH):04d}.pt"
    )


def nsdl_datapoint_indices():
    generator = np.random.default_rng(NSDL_DATAPOINT_SEED)
    return tuple(
        int(value)
        for value in generator.choice(
            N_TRAIN, size=NSDL_DATAPOINT_COUNT, replace=False
        )
    )
