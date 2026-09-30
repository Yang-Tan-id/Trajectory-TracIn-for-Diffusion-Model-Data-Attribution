"""Configuration for target pollution/loss-noise mismatch ablation."""

from endpoint_direction_mc_config import *


EDNA_ROOT = ROOT / "endpoint_direction_mc_nonaligned_shift100dir_1000t"
EDNA_SOURCE_DIR = EDNA_ROOT / "sources"
EDNA_LOG_DIR = EDNA_ROOT / "logs"
EDNA_SUMMARY_PATH = EDNA_ROOT / "summary.json"


def edna_source_dir(source_index):
    target_index = ntcd_target_index(source_index)
    return EDNA_SOURCE_DIR / f"source_{source_index:05d}_target_{target_index:05d}"
