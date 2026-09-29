"""Clearly named entry point for the full q00-q99 diagonal-clean DAS run."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def main():
    path = Path(__file__).with_name(
        "123_launch_diagonal_clean_aligned_das_10q_4gpu.py"
    )
    spec = spec_from_file_location("diagonal_clean_das_launcher", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
