"""Configuration for one-minibatch per-datapoint gradient decomposition."""

from multisource4_fixed_point_config import *


MB4_SEQUENCE_COUNT = MS4_SEQUENCE_COUNT
MB4_DATAPOINT_COUNT = 4
MB4_TIMESTAMPS_PER_DATAPOINT = NSDL_UPDATE_BATCH_SIZE
MB4_DEFAULT_BATCH_SIZE = 256
MB4_METHODS = (
    "sgd_per_datapoint_vector_sum",
    "adam_data_only_vector_sum",
    "adam_baseline_plus_data_vector_sum",
    "adam_exact_parameter_delta_control",
)
MB4_ROOT = ROOT / "minibatch4_gradient_decomposition_20dir_100t_float64"
MB4_SEQUENCE_DIR = MB4_ROOT / "sequences"
MB4_LOG_DIR = MB4_ROOT / "logs"
MB4_SUMMARY_PATH = MB4_ROOT / "summary.json"


def mb4_source_indices(sequence_index):
    return ms4_source_indices(sequence_index)


def mb4_target_index(sequence_index):
    return ms4_target_index(sequence_index)


def mb4_sequence_dir(sequence_index):
    sources = mb4_source_indices(sequence_index)
    target = mb4_target_index(sequence_index)
    source_label = "_".join(f"{value:05d}" for value in sources)
    return MB4_SEQUENCE_DIR / (
        f"sequence_{sequence_index:02d}_sources_{source_label}_target_{target:05d}"
    )
