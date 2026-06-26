from pathlib import Path
import os
import sys

DATASET_DIR = Path(__file__).resolve().parents[1]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from dataset_config import EVAL_ROOT

ALGORITHM = os.environ.get("ALGORITHM", "das")
NOTE_PATH = EVAL_ROOT / "lds" / ALGORITHM / "lds_note.txt"
COMMAND_CWD = "repo"
COMMANDS = {
    "lds": [
        "python",
        "-c",
        (
            "from pathlib import Path; "
            f"p=Path(r'{NOTE_PATH}'); "
            "p.parent.mkdir(parents=True, exist_ok=True); "
            "p.write_text('ArtBench LDS folder is scaffolded here. "
            "The current legacy LDS runner is CIFAR-specific, so add an "
            "ArtBench LDS engine before launching this metric.\\n'); "
            "print(p)"
        ),
    ]
}

