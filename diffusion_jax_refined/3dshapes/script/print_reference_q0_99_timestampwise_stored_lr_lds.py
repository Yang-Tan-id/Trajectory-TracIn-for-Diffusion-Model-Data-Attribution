#!/usr/bin/env python3
"""Print Q0-Q99 aligned AdamW timestamp-wise scores with stored/current LR."""

from __future__ import annotations

import sys

import print_aligned10x10_lds as printer


SCHEMES = (
    (
        "adamw_residual_timestamp_square_stored_lr",
        "traj_tracin_adamw_residual_aligned10x10_timestamp_sum_squared_stored_lr_ref100q",
    ),
    (
        "adamw_full_timestamp_square_stored_lr",
        "traj_tracin_adamw_full_aligned10x10_timestamp_sum_squared_stored_lr_ref100q",
    ),
)


if __name__ == "__main__":
    # Reuse an accepted command-line group name while replacing its registry.
    printer.SCHEME_GROUPS["timestamp_sum_squared"] = SCHEMES
    if "--scheme-group" not in sys.argv:
        sys.argv.extend(("--scheme-group", "timestamp_sum_squared"))
    printer.main()
