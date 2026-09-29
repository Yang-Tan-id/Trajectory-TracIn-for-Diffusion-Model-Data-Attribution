"""Clearly named entry point for the q00-q99 multiclean DAS LDS sweep."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def main():
    path = Path(__file__).with_name("110_eval_multiclean_aligned_das_10q_lds.py")
    spec = spec_from_file_location("multiclean_das_eval", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
