"""Clearly named entry point for the full q00-q99 experiment."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def main():
    path = Path(__file__).with_name("115_launch_diffusion_retrac_10q_4gpu.py")
    spec = spec_from_file_location("diffusion_retrac_launcher", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
