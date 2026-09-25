#!/usr/bin/env python3
"""Print Q0-Q99 aligned AdamW timestamp-wise score families."""

from __future__ import annotations

import sys

import print_aligned10x10_lds as printer


SCHEMES = (
    (
        "adamw_residual_timestamp_square_previous_lr",
        "traj_tracin_adamw_residual_aligned10x10_timestamp_sum_squared_previous_lr_ref100q",
    ),
    (
        "adamw_full_timestamp_square_previous_lr",
        "traj_tracin_adamw_full_aligned10x10_timestamp_sum_squared_previous_lr_ref100q",
    ),
)


if __name__ == "__main__":
    # Reuse the established table reader/formatter without modifying its
    # existing scheme registry, which may contain concurrent local work.
    printer.SCHEME_GROUPS["raw_loss"] = SCHEMES
    if "--scheme-group" not in sys.argv:
        sys.argv.extend(("--scheme-group", "raw_loss"))
    printer.main()
