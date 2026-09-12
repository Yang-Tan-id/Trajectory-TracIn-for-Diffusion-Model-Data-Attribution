from pathlib import Path
import os
import sys

REFINE_ROOT = Path(__file__).resolve().parents[3]
if str(REFINE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFINE_ROOT))
from common.stage_artifact_runner import run_score_combination_stage

if __name__ == "__main__":
    # One cached train/query pass emits raw, query-L2, train-L2, and both-L2.
    os.environ.setdefault("TRACIN_SCORE_QUERY_NORMALIZE", "1")
    os.environ.setdefault("TRACIN_SCORE_TRAIN_NORMALIZE", "1")
    run_score_combination_stage(Path(__file__).with_name("CONFIG.py"))
