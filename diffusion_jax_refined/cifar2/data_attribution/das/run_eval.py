from pathlib import Path
import sys

REFINE_ROOT = Path(__file__).resolve().parents[3]
if str(REFINE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFINE_ROOT))

from common.eval_placeholders import write_eval_note
from common.paths import eval_root
from common.config_loader import load_config


if __name__ == "__main__":
    cfg = load_config(Path(__file__).with_name("CONFIG.py"))
    out = eval_root(cfg.DATASET_NAME, cfg.EXPERIMENT_TAG, cfg.ALGORITHM) / "eval_note.txt"
    write_eval_note(
        out,
        "DAS eval placeholder. Use ../../scripts/02_metric_counterfactual.sh or "
        "../../scripts/03_metric_lds.sh for metric evaluation.\n",
    )
    print(out)

