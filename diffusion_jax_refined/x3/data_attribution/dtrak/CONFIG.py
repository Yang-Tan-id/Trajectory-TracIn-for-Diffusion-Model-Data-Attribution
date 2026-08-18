from pathlib import Path
import os
import sys

DATASET_DIR = Path(__file__).resolve().parents[2]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from dataset_config import DATASET_NAME, EXPERIMENT_TAG, attribution_config, commands_for_algorithm, unprompted_attribution_config

ALGORITHM = "dtrak"
ATTRIBUTION_CONFIG = unprompted_attribution_config(ALGORITHM) if os.environ.get("UNPROMPTED", "0") in ("1", "true", "True", "yes") else attribution_config(ALGORITHM)
COMMANDS = commands_for_algorithm(ALGORITHM)
COMMAND_CWD = "legacy_jax"

