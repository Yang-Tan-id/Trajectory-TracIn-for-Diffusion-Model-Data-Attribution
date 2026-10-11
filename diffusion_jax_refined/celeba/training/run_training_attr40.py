from pathlib import Path
import os
import sys

REFINE_ROOT = Path(__file__).resolve().parents[2]
if str(REFINE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFINE_ROOT))

from common.prompted_jax_training import main


if __name__ == "__main__":
    os.environ.setdefault("TRAIN_SEED", "42")
    os.environ.setdefault("JAX_EPOCHS", "200")
    os.environ.setdefault("JAX_LR_SCHEDULE", "cosine_warmup")
    os.environ.setdefault("JAX_LR_WARMUP_RATIO", "0.1")
    os.environ.setdefault("JAX_WEIGHT_DECAY", "1e-4")
    os.environ.setdefault("JAX_ADAM_B1", "0.9")
    os.environ.setdefault("JAX_BFLOAT16", "1")
    sys.argv.insert(1, str(Path(__file__).resolve().parents[1] / "dataset_config_attr40.py"))
    main()
