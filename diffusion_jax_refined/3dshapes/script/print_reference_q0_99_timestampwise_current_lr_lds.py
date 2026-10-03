#!/usr/bin/env python3
"""Print Q0-Q99 full-AdamW timestamp-wise scores with current checkpoint LR."""

from __future__ import annotations

import sys

import print_aligned10x10_lds as printer


SCHEMES = (
    (
        "adamw_full_timestamp_square_current_lr",
        "traj_tracin_adamw_full_aligned10x10_timestamp_sum_squared_current_lr_ref100q",
    ),
)


if __name__ == "__main__":
    printer.SCHEME_GROUPS["timestamp_sum_squared"] = SCHEMES
    if "--scheme-group" not in sys.argv:
        sys.argv.extend(("--scheme-group", "timestamp_sum_squared"))
    printer.main()
