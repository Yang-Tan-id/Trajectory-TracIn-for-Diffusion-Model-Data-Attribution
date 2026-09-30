"""Configuration for opposite-side gradient magnitude prediction."""

from exp_config import ROOT
from null_tblock_cross_direction_config import *


OGM_CANDIDATES = (
    "plus_gradient_sgd",
    "minus_gradient_sgd",
    "even_gradient_sgd",
    "odd_gradient_sgd",
    "actual_parameter_delta_jvp",
)
OGM_ROOT = ROOT / "null_tblock_opposite_gradient_magnitude_fresh_sgd"
OGM_SOURCE_DIR = OGM_ROOT / "sources"
OGM_LOG_DIR = OGM_ROOT / "logs"
OGM_SUMMARY_PATH = OGM_ROOT / "summary.json"


def ogm_source_dir(source_index):
    target_index = ntcd_target_index(source_index)
    return OGM_SOURCE_DIR / f"source_{source_index:05d}_target_{target_index:05d}"
