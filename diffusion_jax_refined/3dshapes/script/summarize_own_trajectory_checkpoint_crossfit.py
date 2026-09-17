#!/usr/bin/env python3
"""Print no-flip and five-bin CV LDS from checkpoint crossfit output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    with args.summary.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["method"] == "five_bins"
        ]

    print("OWN-TRAJECTORY ORIGINAL-F — NO FLIP vs FIVE-BIN CROSSFIT")
    print("Q VARIANT             NO-FLIP FULL   FIVE-BIN CV     STD    CV>0")
    print("-" * 76)
    for row in rows:
        print(
            f"{int(row['query']):1d} "
            f"{row['variant']:<19s} "
            f"{float(row['all_plus_cf_joint_percent']):+11.3f}% "
            f"{float(row['crossfit_cf_joint_mean_percent']):+11.3f}% "
            f"{float(row['crossfit_cf_joint_std_percent']):7.3f}% "
            f"{float(row['crossfit_positive_repeat_fraction']):6.2f}"
        )


if __name__ == "__main__":
    main()
