"""Configuration for endpoint multi-direction response estimation."""

from exp_config import ROOT
from null_tblock_cross_direction_config import *


EDMC_DIRECTION_COUNT = 100
EDMC_DIRECTION_SEED_BASE = 995100
EDMC_DEFAULT_BATCH_SIZE = 5120
EDMC_SUBSET_COUNTS = (1, 2, 4, 8, 10, 20, 50, 100)
EDMC_SUBSET_REPEATS = 20
EDMC_ROOT = ROOT / "endpoint_direction_mc_fresh_sgd_100dir_1000t"
EDMC_SOURCE_DIR = EDMC_ROOT / "sources"
EDMC_LOG_DIR = EDMC_ROOT / "logs"
EDMC_SUMMARY_PATH = EDMC_ROOT / "summary.json"


def edmc_source_dir(source_index):
    target_index = ntcd_target_index(source_index)
    return EDMC_SOURCE_DIR / f"source_{source_index:05d}_target_{target_index:05d}"
