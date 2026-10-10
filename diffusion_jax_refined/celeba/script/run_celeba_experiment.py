#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


CELEBA_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = CELEBA_ROOT.parent
DEFAULT_OUT = REFINE_ROOT / "dataset" / "celeba" / "10000"


def run(command: list[str], env: dict[str, str], execute: bool) -> None:
    print("+", " ".join(command), flush=True)
    if execute:
        subprocess.run(command, cwd=CELEBA_ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and train the CelebA 10k experiment.")
    parser.add_argument("--images-zip", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", "--skip-training", dest="skip_training", action="store_true")
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    env = os.environ.copy()
    env.update(
        {
            "CELEBA_DATA_ROOT": str(args.out_root.resolve()),
            "EXPERIMENT_TAG": args.experiment,
            "TRAIN_SEED": str(args.seed),
            "JAX_EPOCHS": str(args.epochs),
            "JAX_LR_SCHEDULE": "cosine_warmup",
            "JAX_LR_WARMUP_RATIO": "0.1",
        }
    )
    python = env.get("PYTHON_BIN", sys.executable)
    if not args.skip_prepare:
        run(
            [
                python,
                "script/prepare_celeba.py",
                "--images-zip",
                str(args.images_zip.resolve()),
                "--partition",
                str(args.partition.resolve()),
                "--out-root",
                str(args.out_root.resolve()),
                "--seed",
                str(args.seed),
                "--train-size",
                "10000",
                "--attribution-size",
                "5000",
                "--crop-size",
                "140",
                "--image-size",
                "64",
            ],
            env,
            args.execute,
        )
    if not args.skip_training:
        run([python, "training/run_training.py"], env, args.execute)


if __name__ == "__main__":
    main()
