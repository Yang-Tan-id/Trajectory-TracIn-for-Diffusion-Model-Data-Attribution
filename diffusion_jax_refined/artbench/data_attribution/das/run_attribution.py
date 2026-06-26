from pathlib import Path
import sys

REFINE_ROOT = Path(__file__).resolve().parents[3]
if str(REFINE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFINE_ROOT))

from common.algorithm_runner import run_algorithm_config


if __name__ == "__main__":
    run_algorithm_config(Path(__file__).with_name("CONFIG.py"))

