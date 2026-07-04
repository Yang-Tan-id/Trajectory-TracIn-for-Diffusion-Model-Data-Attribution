from pathlib import Path
import os
import sys

DATASET_DIR = Path(__file__).resolve().parents[1]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from dataset_config import DATA_ROOT, DATASET_NAME, EVAL_ROOT, EXPERIMENT_TAG, QUERY, REFERENCE_CKPT, UNPROMPTED_JAX_REFERENCE_CKPT

SAMPLE_SEEDS = os.environ.get("SAMPLE_SEEDS", "0")
SAMPLE_BATCH_SIZE = os.environ.get("SAMPLE_BATCH_SIZE", "1")
SAMPLE_TRAJECTORY_STEPS = os.environ.get("SAMPLE_TRAJECTORY_STEPS", "100")
UNPROMPTED = os.environ.get("UNPROMPTED", "0") in ("1", "true", "True", "yes")
SAMPLE_CHECKPOINT = UNPROMPTED_JAX_REFERENCE_CKPT if UNPROMPTED else REFERENCE_CKPT
SAMPLE_PROMPT = "unconditional" if UNPROMPTED else QUERY
MODEL_TAG = "unprompted_jax" if UNPROMPTED else "prompted_jax"

COMMAND_CWD = "legacy_jax"
COMMANDS = {
    "sampling": [
        "python",
        "DM___data_attribution_sampler.py",
        "--adapter=cifar",
        "--code-file=DM__training_CIFAR10_pixel.py",
        f"--checkpoint={SAMPLE_CHECKPOINT}",
        f"--cifar-data-root={DATA_ROOT}",
        f"--model-tag={MODEL_TAG}",
        f"--prompt={SAMPLE_PROMPT}",
        f"--seeds={SAMPLE_SEEDS}",
        f"--batch-size={SAMPLE_BATCH_SIZE}",
        "--prefer-device=gpu",
        f"--outdir={EVAL_ROOT / 'sampling'}",
        f"--num-trajectory-steps={SAMPLE_TRAJECTORY_STEPS}",
    ]
}
