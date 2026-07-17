from __future__ import annotations

import argparse
from pathlib import Path
import sys


REFINE_ROOT = Path(__file__).resolve().parents[1]
if str(REFINE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFINE_ROOT))

from common.algorithm_runner import run_algorithm_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the original monolithic legacy attribution engine for one CONFIG.py."
    )
    parser.add_argument("config_path", type=Path)
    args = parser.parse_args()
    run_algorithm_config(args.config_path)


if __name__ == "__main__":
    main()
