"""Configuration for four-source sequential fixed-point validation."""

import numpy as np

from sequential4_fixed_point_config import *


MS4_SEQUENCE_COUNT = len(nsdl_datapoint_indices())
MS4_TARGET_SEED_BASE = 1004100
MS4_METHODS = (
    "total_start_jvp",
    "stepwise_start_vector_sum",
    "stepwise_gauss2_vector_sum",
)
MS4_DEFAULT_BATCH_SIZE = 256
MS4_ROOT = ROOT / "multisource4_restored_adamw_fixed_point_20dir_100t_float64"
MS4_SEQUENCE_DIR = MS4_ROOT / "sequences"
MS4_LOG_DIR = MS4_ROOT / "logs"
MS4_SUMMARY_PATH = MS4_ROOT / "summary.json"


def ms4_source_indices(sequence_index):
    pool = nsdl_datapoint_indices()
    return tuple(pool[(int(sequence_index) + offset) % len(pool)] for offset in range(4))


def ms4_target_index(sequence_index):
    excluded = set(ms4_source_indices(sequence_index))
    generator = np.random.default_rng(MS4_TARGET_SEED_BASE + int(sequence_index))
    while True:
        candidate = int(generator.integers(0, N_TRAIN))
        if candidate not in excluded:
            return candidate


def ms4_sequence_dir(sequence_index):
    sources = ms4_source_indices(sequence_index)
    target = ms4_target_index(sequence_index)
    source_label = "_".join(f"{value:05d}" for value in sources)
    return MS4_SEQUENCE_DIR / f"sequence_{sequence_index:02d}_sources_{source_label}_target_{target:05d}"
