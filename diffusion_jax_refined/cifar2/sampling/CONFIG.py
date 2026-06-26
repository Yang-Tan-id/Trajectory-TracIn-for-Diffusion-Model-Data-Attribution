from pathlib import Path
import sys

DATASET_DIR = Path(__file__).resolve().parents[1]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from dataset_config import DATASET_NAME, EXPERIMENT_TAG, EVAL_ROOT, QUERY, REFERENCE_CKPT

COMMAND_CWD = "legacy_jax"
COMMANDS = {
    "sampling": [
        "python",
        "DM___data_attribution_sampler.py",
        "--adapter=cifar",
        "--code-file=DM__training_CIFAR10_pixel.py",
        f"--checkpoint={REFERENCE_CKPT}",
        "--model-tag=prompted_jax",
        f"--prompt={QUERY}",
        "--seeds=0",
        "--batch-size=1",
        "--prefer-device=gpu",
        f"--outdir={EVAL_ROOT / 'sampling'}",
        "--num-trajectory-steps=100",
    ]
}
