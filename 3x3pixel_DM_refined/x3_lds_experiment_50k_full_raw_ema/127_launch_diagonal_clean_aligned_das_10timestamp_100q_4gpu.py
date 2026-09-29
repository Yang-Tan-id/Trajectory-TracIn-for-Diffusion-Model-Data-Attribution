"""Launch the ten-timestamp, one-endpoint-per-level DAS experiment."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def main():
    if "--timestamp-count" not in sys.argv:
        sys.argv.extend(["--timestamp-count", "10"])
    path = Path(__file__).with_name(
        "123_launch_diagonal_clean_aligned_das_10q_4gpu.py"
    )
    spec = spec_from_file_location("diagonal_clean_das_launcher_10t", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
