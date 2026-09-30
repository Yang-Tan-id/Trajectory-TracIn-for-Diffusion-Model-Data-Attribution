"""Configuration for fixed-point parameter-path JVP validation."""

import numpy as np

from endpoint_direction_mc_config import *


PIJVP_DIRECTION_COUNT = 20
PIJVP_TIMESTAMP_COUNT = 100
PIJVP_DIRECTION_INDICES = tuple(
    int(value)
    for value in np.linspace(
        0, EDMC_DIRECTION_COUNT - 1, PIJVP_DIRECTION_COUNT, dtype=np.int64
    )
)
PIJVP_TIMESTAMPS = tuple(
    int(value)
    for value in np.linspace(0, T - 1, PIJVP_TIMESTAMP_COUNT, dtype=np.int64)
)
PIJVP_DEFAULT_BATCH_SIZE = 256
PIJVP_PRECISION_MODE = "float64_no_tf32"
PIJVP_METHODS = (
    "start_jvp",
    "second_order_taylor",
    "endpoint_trapezoid",
    "gauss2_path_jvp",
    "gauss4_path_jvp",
)
PIJVP_ROOT = ROOT / "path_integrated_jvp_fresh_sgd_20dir_100t_float64"
PIJVP_SOURCE_DIR = PIJVP_ROOT / "sources"
PIJVP_LOG_DIR = PIJVP_ROOT / "logs"
PIJVP_SUMMARY_PATH = PIJVP_ROOT / "summary.json"


def pijvp_source_dir(source_index):
    target_index = ntcd_target_index(source_index)
    return PIJVP_SOURCE_DIR / f"source_{source_index:05d}_target_{target_index:05d}"


if len(set(PIJVP_DIRECTION_INDICES)) != PIJVP_DIRECTION_COUNT:
    raise ValueError("path-JVP direction selection contains duplicates")
if len(set(PIJVP_TIMESTAMPS)) != PIJVP_TIMESTAMP_COUNT:
    raise ValueError("path-JVP timestamp selection contains duplicates")
