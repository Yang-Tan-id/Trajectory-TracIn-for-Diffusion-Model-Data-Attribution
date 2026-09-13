from pathlib import Path
import os
import sys

DATASET_DIR = Path(__file__).resolve().parents[1]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from dataset_config import (
    DATA_ROOT,
    DATASET_NAME,
    EXPERIMENT_TAG,
    QUERY,
    REFERENCE_CKPT,
    RESULT_ROOT,
    TRAINING_MODULE_NAME,
)

SAMPLE_SEEDS = os.environ.get("SAMPLE_SEEDS", os.environ.get("INITIAL_SEED", "0"))
SAMPLE_BATCH_SIZE = os.environ.get("SAMPLE_BATCH_SIZE", "1")
SAMPLE_TRAJECTORY_STEPS = os.environ.get("SAMPLE_TRAJECTORY_STEPS", "1000")
SAMPLE_PREFER_DEVICE = os.environ.get("SAMPLE_PREFER_DEVICE", "gpu")
TRAJECTORY_SAMPLER = os.environ.get("DIFFUSION_TRAJECTORY_SAMPLER", "ddim_eta0")
SAVE_TRAJECTORY_PNGS = os.environ.get("SAVE_TRAJECTORY_PNGS", "0").lower() in (
    "1",
    "true",
    "yes",
)
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
        f"--batch-size={SAMPLE_BATCH_SIZE}",
        f"--prefer-device={SAMPLE_PREFER_DEVICE}",
        f"--outdir={SAMPLE_ROOT}",
        f"--num-trajectory-steps={SAMPLE_TRAJECTORY_STEPS}",
        f"--trajectory-sampler={TRAJECTORY_SAMPLER}",
    ]
    + (["--save-trajectory-pngs"] if SAVE_TRAJECTORY_PNGS else [])
}
