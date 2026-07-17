from pathlib import Path
import sys

REFINE_ROOT = Path(__file__).resolve().parents[3]
if str(REFINE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFINE_ROOT))

from common.stage_artifact_runner import run_score_combination_stage


if __name__ == "__main__":
    run_score_combination_stage(Path(__file__).with_name("CONFIG.py"))
