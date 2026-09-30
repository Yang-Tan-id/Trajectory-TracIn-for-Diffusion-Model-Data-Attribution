"""Run all optimizer controls and print their combined comparison."""

import argparse
import subprocess
import sys

from null_tblock_cross_direction_config import NTCD_OPTIMIZER_MODES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    # Fresh optimizers do not load mutable checkpoint state. Their existing
    # outputs remain valid; launchers skip them when result.json is present.
    for optimizer_mode in NTCD_OPTIMIZER_MODES:
        print(f"\n[control launcher] optimizer={optimizer_mode}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-u",
                "168_launch_null_tblock_cross_direction_4gpu.py",
                "--gpus",
                args.gpus,
                "--optimizer-mode",
                optimizer_mode,
            ],
            check=True,
        )
    subprocess.run(
        [sys.executable, "-u", "171_compare_null_tblock_optimizer_controls.py"],
        check=True,
    )


if __name__ == "__main__":
    main()
