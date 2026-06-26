from pathlib import Path
import sys

REFINE_ROOT = Path(__file__).resolve().parents[2]
if str(REFINE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFINE_ROOT))

from common.command_runner import run_named_command


if __name__ == "__main__":
    raise SystemExit(run_named_command(Path(__file__).with_name("CONFIG.py"), "lds"))

