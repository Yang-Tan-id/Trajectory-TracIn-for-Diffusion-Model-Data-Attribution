from pathlib import Path
import os
import sys

DATASET_DIR = Path(__file__).resolve().parents[1]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from dataset_config import DATA_ROOT, QUERY, REFERENCE_CKPT, RESULT_ROOT, TRAINING_MODULE_NAME

SAMPLE_SEEDS = os.environ.get("SAMPLE_SEEDS", os.environ.get("INITIAL_SEED", "0"))
SAMPLE_ROOT = Path(os.environ.get("SAMPLE_ROOT", str(RESULT_ROOT / "sample")))
COMMAND_CWD = "legacy_jax"
COMMANDS = {
    "sampling": [
        os.environ.get("PYTHON_BIN", "python3"),
        "DM___data_attribution_sampler.py",
        "--adapter=cifar",
        f"--code-file={TRAINING_MODULE_NAME}.py",
        f"--checkpoint={REFERENCE_CKPT}",
        f"--cifar-data-root={DATA_ROOT}",
        "--model-tag=prompted_solo",
        f"--prompt={QUERY}",
        f"--seeds={SAMPLE_SEEDS}",
        "--batch-size=1",
        "--prefer-device=gpu",
        f"--outdir={SAMPLE_ROOT}",
        "--num-trajectory-steps=1000",
        "--trajectory-sampler=ddim_eta0",
    ]
}

