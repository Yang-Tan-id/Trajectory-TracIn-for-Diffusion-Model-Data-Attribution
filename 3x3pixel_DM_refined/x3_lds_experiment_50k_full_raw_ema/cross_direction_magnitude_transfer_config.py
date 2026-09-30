"""Configuration for finite-difference magnitude-transfer diagnostics."""

from exp_config import ROOT
from null_tblock_cross_direction_config import *


CDMT_ROOT = ROOT / "null_tblock_cross_direction_magnitude_transfer_fresh_sgd"
CDMT_SOURCE_DIR = CDMT_ROOT / "sources"
CDMT_LOG_DIR = CDMT_ROOT / "logs"
CDMT_SUMMARY_PATH = CDMT_ROOT / "summary.json"


def cdmt_source_dir(source_index):
    target_index = ntcd_target_index(source_index)
    return CDMT_SOURCE_DIR / f"source_{source_index:05d}_target_{target_index:05d}"
