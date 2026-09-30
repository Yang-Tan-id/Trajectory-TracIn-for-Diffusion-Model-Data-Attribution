"""Pairwise frozen-start AdamW tangent Bundle experiment."""

from bundle_tracin_config import *


ADAM_BUNDLE_METHOD = (
    "bundle_tracin_frozen_start_adamw_tangent_scalar_calibrated_"
    "projected4096_49pair_timestamp_mean"
)
ADAM_BUNDLE_ROOT = ATTR_DIR / f"_{ADAM_BUNDLE_METHOD}_pairs"
ADAM_BUNDLE_MASK_CHUNK = 192

