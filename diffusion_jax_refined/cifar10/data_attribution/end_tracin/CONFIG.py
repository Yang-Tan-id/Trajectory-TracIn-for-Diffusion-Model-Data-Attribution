from pathlib import Path
import sys

DATASET_DIR = Path(__file__).resolve().parents[2]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from dataset_config import DATASET_NAME, EXPERIMENT_TAG, attribution_config, commands_for_algorithm

ALGORITHM = "end_tracin"
ATTRIBUTION_CONFIG = attribution_config(ALGORITHM)
COMMANDS = commands_for_algorithm(ALGORITHM)
COMMAND_CWD = "legacy_jax"

