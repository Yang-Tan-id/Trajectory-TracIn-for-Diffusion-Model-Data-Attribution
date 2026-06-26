from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from .algorithm_runner import run_unprompted_algorithm_config
except ImportError:
    import sys

    refine_root = Path(__file__).resolve().parents[1]
    if str(refine_root) not in sys.path:
        sys.path.insert(0, str(refine_root))
    from common.algorithm_runner import run_unprompted_algorithm_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unconditional JAX attribution.")
    parser.add_argument("config", type=str)
    parser.add_argument("--algorithm", default=os.environ.get("ALGORITHM", "das"))
    args = parser.parse_args()
    run_unprompted_algorithm_config(args.config, args.algorithm)


if __name__ == "__main__":
    main()
