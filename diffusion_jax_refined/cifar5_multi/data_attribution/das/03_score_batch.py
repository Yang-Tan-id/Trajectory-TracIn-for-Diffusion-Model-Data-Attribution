#!/usr/bin/env python3
from pathlib import Path

from common.stage_artifact_runner import run_das_score_batch_stage


if __name__ == "__main__":
    run_das_score_batch_stage(Path(__file__).with_name("CONFIG.py"))
