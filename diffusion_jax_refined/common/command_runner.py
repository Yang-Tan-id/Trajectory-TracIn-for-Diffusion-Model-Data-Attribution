from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Iterable

try:
    from .config_loader import load_config, require_attr
    from .paths import (
        LEGACY_JAX_ROOT,
        REPO_ROOT,
        add_legacy_jax_to_path,
        ensure_experiment_dirs,
    )
except ImportError:
    import sys

    refine_root = Path(__file__).resolve().parents[1]
    if str(refine_root) not in sys.path:
        sys.path.insert(0, str(refine_root))
    from common.config_loader import load_config, require_attr
    from common.paths import (
        LEGACY_JAX_ROOT,
        REPO_ROOT,
        add_legacy_jax_to_path,
        ensure_experiment_dirs,
    )


def run_command(command: Iterable[str], *, cwd: Path | None = None, dry_run: bool = False) -> int:
    cmd = [str(part) for part in command if str(part) != ""]
    print("+ " + " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(cwd) if cwd is not None else None)


def run_named_command(config_path: str | Path, command_name: str, *, dry_run: bool = False) -> int:
    cfg = load_config(config_path)
    dataset_name = require_attr(cfg, "DATASET_NAME")
    experiment_tag = require_attr(cfg, "EXPERIMENT_TAG")
    commands = require_attr(cfg, "COMMANDS")
    if command_name not in commands:
        raise ValueError(f"{config_path} does not define command {command_name!r}")
    ensure_experiment_dirs(dataset_name, experiment_tag)
    add_legacy_jax_to_path()
    command = commands[command_name]
    cwd_name = getattr(cfg, "COMMAND_CWD", "legacy_jax")
    if cwd_name == "legacy_jax":
        cwd = LEGACY_JAX_ROOT
    elif cwd_name == "repo":
        cwd = REPO_ROOT
    else:
        cwd = Path.cwd()
    return run_command(command, cwd=cwd, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a named command from a refine CONFIG.py")
    parser.add_argument("config", type=str)
    parser.add_argument("command", type=str)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run_named_command(args.config, args.command, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
