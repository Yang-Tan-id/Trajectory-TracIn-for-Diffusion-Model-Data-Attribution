from pathlib import Path
import os
import sys

DATASET_DIR = Path(__file__).resolve().parents[1]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from dataset_config import ATTRIBUTION_ROOT, EVAL_ROOT, EXPERIMENT_TAG

ALGORITHM = os.environ.get("ALGORITHM", "das")
NOTE_PATH = EVAL_ROOT / "counterfactual" / ALGORITHM / "counterfactual_note.txt"
COMMAND_CWD = "repo"
COMMANDS = {
    "counterfactual": [
        "python",
        "-c",
        (
            "from pathlib import Path; "
            f"p=Path(r'{NOTE_PATH}'); "
            "p.parent.mkdir(parents=True, exist_ok=True); "
            "p.write_text('ArtBench counterfactual retrain is scaffolded here, "
            "but the current legacy counterfactual runner is CIFAR-specific.\\n'); "
            "print(p)"
        ),
    ]
}

