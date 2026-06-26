from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .command_runner import run_command
    from .config_loader import load_config, require_attr
    from .paths import ensure_experiment_dirs
except ImportError:
    import sys

    refine_root = Path(__file__).resolve().parents[1]
    if str(refine_root) not in sys.path:
        sys.path.insert(0, str(refine_root))
    from common.command_runner import run_command
    from common.config_loader import load_config, require_attr
    from common.paths import ensure_experiment_dirs


def run_unprompted_training(config_path: str | Path, *, dry_run: bool = False) -> int:
    cfg = load_config(config_path)
    dataset_name = require_attr(cfg, "DATASET_NAME")
    experiment_tag = require_attr(cfg, "EXPERIMENT_TAG")
    command_fn = require_attr(cfg, "unprompted_training_command")
    ensure_experiment_dirs(dataset_name, experiment_tag)
    return run_command(command_fn(), cwd=Path.cwd(), dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run shared unprompted JAX training.")
    parser.add_argument("config", type=str)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run_unprompted_training(args.config, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
