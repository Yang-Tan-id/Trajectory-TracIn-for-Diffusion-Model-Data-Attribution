from pathlib import Path
import sys

REFINE_ROOT = Path(__file__).resolve().parents[3]
if str(REFINE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFINE_ROOT))

from common.stage_artifact_producer import run_query_gradient_artifact


if __name__ == "__main__":
    run_query_gradient_artifact(Path(__file__).with_name("CONFIG.py"))
