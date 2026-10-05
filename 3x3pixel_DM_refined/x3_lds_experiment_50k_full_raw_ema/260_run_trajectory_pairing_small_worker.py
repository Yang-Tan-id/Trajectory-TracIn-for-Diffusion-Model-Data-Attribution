"""Pure trajectory direction with aligned or independent train noise."""

import argparse
import importlib.util
import sys
from pathlib import Path

from exp_config import ATTR_DIR


def load_worker():
    path = Path(__file__).with_name(
        "248_run_trajectory_bridge_aligned_tracin_das_shard.py"
    )
    spec = importlib.util.spec_from_file_location("trajectory_pairing_worker", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--mode", choices=("aligned", "independent"), required=True)
    ap.add_argument("--timestamp-shard-index", type=int, required=True)
    ap.add_argument("--timestamp-shard-count", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--grad-microbatch-size", type=int, default=4)
    ap.add_argument("--query-term-batch-size", type=int, default=32)
    args = ap.parse_args()

    worker = load_worker()
    worker.TBA_QUERY_IDS = tuple(range(10))
    worker.TBA_DIRECTION_COUNT = 1
    worker.TBA_PURE_IMPLIED_NOISE = True
    worker.TBA_TRAIN_NOISE_MODE = args.mode
    worker.TBA_VERSION = 4

    def root(shard_index, shard_count):
        return (
            ATTR_DIR
            / "_tracin_das_trajectory_pairing_small_shards"
            / args.mode
            / f"shard_{shard_index:02d}_of_{shard_count:02d}"
        )

    worker.tba_shard_root = root
    sys.argv = [
        str(Path(__file__)),
        "--gpu", str(args.gpu),
        "--timestamp-shard-index", str(args.timestamp_shard_index),
        "--timestamp-shard-count", str(args.timestamp_shard_count),
        "--batch-size", str(args.batch_size),
        "--grad-microbatch-size", str(args.grad_microbatch_size),
        "--query-term-batch-size", str(args.query_term_batch_size),
    ]
    worker.main()


if __name__ == "__main__":
    main()
