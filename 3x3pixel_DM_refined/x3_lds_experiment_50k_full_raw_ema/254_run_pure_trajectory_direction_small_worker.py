"""Run one query of the tiny pure trajectory-direction TracIn-DAS test."""

import argparse
import importlib.util
import sys
from pathlib import Path

from exp_config import ATTR_DIR


SMALL_TIMESTAMP_INDICES = (0, 6, 12, 18, 24)


def load_worker():
    path = Path(__file__).with_name(
        "248_run_trajectory_bridge_aligned_tracin_das_shard.py"
    )
    spec = importlib.util.spec_from_file_location("trajectory_bridge_worker", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--query-id", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--grad-microbatch-size", type=int, default=4)
    ap.add_argument("--query-term-batch-size", type=int, default=32)
    args = ap.parse_args()
    if not 0 <= args.query_id <= 3:
        raise ValueError("tiny experiment is restricted to q00-q03")

    worker = load_worker()
    worker.TBA_QUERY_IDS = (args.query_id,)
    worker.TBA_DIRECTION_COUNT = 1
    worker.TBA_PURE_IMPLIED_NOISE = True
    worker.NPA_TIMESTAMP_INDICES = SMALL_TIMESTAMP_INDICES
    worker.NPA_TIMESTAMP_GROUPS = {"small": SMALL_TIMESTAMP_INDICES}
    worker.TBA_VERSION = 2

    def root(_shard_index, _shard_count):
        return (
            ATTR_DIR
            / "_tracin_das_pure_trajectory_direction_tiny_shards"
            / f"q{args.query_id:02d}"
        )

    worker.tba_shard_root = root
    sys.argv = [
        str(Path(__file__)),
        "--gpu", str(args.gpu),
        "--timestamp-shard-index", "0",
        "--timestamp-shard-count", "1",
        "--batch-size", str(args.batch_size),
        "--grad-microbatch-size", str(args.grad_microbatch_size),
        "--query-term-batch-size", str(args.query_term_batch_size),
    ]
    worker.main()


if __name__ == "__main__":
    main()
