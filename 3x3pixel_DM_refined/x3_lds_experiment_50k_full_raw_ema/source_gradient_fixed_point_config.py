"""Configuration for source-gradient-to-fixed-target response validation."""

from path_integrated_jvp_config import *


SGFP_METHODS = (
    "source_gradient_sgd",
    "source_gradient_sgd_float32_quantized",
    "exact_parameter_delta_control",
)
SGFP_DEFAULT_BATCH_SIZE = 512
SGFP_ROOT = ROOT / "source_gradient_fixed_point_jvp_20dir_100t_float64"
SGFP_SOURCE_DIR = SGFP_ROOT / "sources"
SGFP_LOG_DIR = SGFP_ROOT / "logs"
SGFP_SUMMARY_PATH = SGFP_ROOT / "summary.json"


def sgfp_source_dir(source_index):
    target_index = ntcd_target_index(source_index)
    return SGFP_SOURCE_DIR / f"source_{source_index:05d}_target_{target_index:05d}"
