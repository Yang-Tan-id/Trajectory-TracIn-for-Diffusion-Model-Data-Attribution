from pathlib import Path
import sys

REFINE_ROOT = Path(__file__).resolve().parents[2]
if str(REFINE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFINE_ROOT))

from common.lds_model_train import main

if __name__ == "__main__":
    sys.argv.insert(1, str(Path(__file__).resolve().parents[1] / "dataset_config.py"))
    main()
