#!/usr/bin/env python3
"""Run cached LDS for the Q0-Q99 reference timestamp-wise experiments."""

from __future__ import annotations

import run_traj_tracin_lds_cached as cached


EXTRA_SCHEMES = {
    "raw_mc10_aligned10x10_linear_stored_lr_ref100q":
        "traj_tracin_raw_mc10_aligned10x10_linear_stored_lr_ref100q",
    "raw_mc10_aligned10x10_timestamp_sum_squared_previous_lr_ref100q":
        "traj_tracin_raw_mc10_aligned10x10_timestamp_sum_squared_previous_lr_ref100q",
    "adamw_residual_aligned10x10_timestamp_sum_squared_previous_lr_ref100q":
        "traj_tracin_adamw_residual_aligned10x10_timestamp_sum_squared_previous_lr_ref100q",
    "adamw_full_aligned10x10_timestamp_sum_squared_previous_lr_ref100q":
        "traj_tracin_adamw_full_aligned10x10_timestamp_sum_squared_previous_lr_ref100q",
}


if __name__ == "__main__":
    cached.SCORE_SCHEMES.update(EXTRA_SCHEMES)
    cached.main()
