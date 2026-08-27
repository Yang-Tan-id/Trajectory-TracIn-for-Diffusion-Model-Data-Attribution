from pathlib import Path
import os
import sys

DATASET_DIR = Path(__file__).resolve().parents[1]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from dataset_config import DATA_ROOT, DATASET_NAME, EXPERIMENT_TAG, QUERY, REFERENCE_CKPT, RESULT_ROOT, TRAINING_MODULE_NAME, UNPROMPTED_JAX_REFERENCE_CKPT


def _normalize_model_mode(value: str, *, unprompted: bool) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "prompt": "prompted_solo",
        "prompted": "prompted_solo",
        "prompted_jax": "prompted_solo",
        "prompted_solo": "prompted_solo",
        "multi": "prompted_multi",
        "prompted_multi": "prompted_multi",
        "unprompted": "unprompted_solo",
        "unprompted_jax": "unprompted_solo",
        "unprompted_solo": "unprompted_solo",
        "unprompted_multi": "unprompted_multi",
    }
    if not text:
        return "unprompted_solo" if unprompted else "prompted_solo"
    if text not in aliases:
        raise ValueError(
            "SAMPLE_MODEL_MODE must be one of prompted_solo, prompted_multi, "
            "unprompted_solo, unprompted_multi, prompted, unprompted, or multi."
        )
    return aliases[text]

SAMPLE_SEEDS = os.environ.get("SAMPLE_SEEDS", "0")
SAMPLE_BATCH_SIZE = os.environ.get("SAMPLE_BATCH_SIZE", "1")
SAMPLE_TRAJECTORY_STEPS = os.environ.get("SAMPLE_TRAJECTORY_STEPS", "100")
UNPROMPTED = os.environ.get("UNPROMPTED", "0") in ("1", "true", "True", "yes")
SAMPLE_MODEL_MODE = _normalize_model_mode(os.environ.get("SAMPLE_MODEL_MODE", ""), unprompted=UNPROMPTED)
UNPROMPTED = UNPROMPTED or SAMPLE_MODEL_MODE.startswith("unprompted")
SAMPLE_CHECKPOINT = UNPROMPTED_JAX_REFERENCE_CKPT if UNPROMPTED else REFERENCE_CKPT
SAMPLE_PROMPT = "unconditional" if UNPROMPTED else QUERY
MODEL_TAG = SAMPLE_MODEL_MODE
SAMPLE_ROOT = Path(os.environ.get("SAMPLE_ROOT", str(RESULT_ROOT / "sample")))

COMMAND_CWD = "legacy_jax"
COMMANDS = {
    "sampling": [
        os.environ.get("PYTHON_BIN", "python3"),
        "DM___data_attribution_sampler.py",
        "--adapter=cifar",
        f"--code-file={TRAINING_MODULE_NAME}.py",
        f"--checkpoint={SAMPLE_CHECKPOINT}",
        f"--cifar-data-root={DATA_ROOT}",
        f"--model-tag={MODEL_TAG}",
        f"--prompt={SAMPLE_PROMPT}",
        f"--seeds={SAMPLE_SEEDS}",
        f"--batch-size={SAMPLE_BATCH_SIZE}",
        "--prefer-device=gpu",
        f"--outdir={SAMPLE_ROOT}",
        f"--num-trajectory-steps={SAMPLE_TRAJECTORY_STEPS}",
    ]
}
