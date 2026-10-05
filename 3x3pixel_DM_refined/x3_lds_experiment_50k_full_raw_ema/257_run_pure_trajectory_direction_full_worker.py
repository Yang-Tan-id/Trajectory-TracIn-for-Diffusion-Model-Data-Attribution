"""Query-sharded full pure trajectory-direction TracIn-DAS worker."""

import argparse
import importlib.util
import sys
from pathlib import Path

from exp_config import ATTR_DIR, TRAJ_SNAPSHOTS


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
    ap.add_argument("--query-shard-index", type=int, required=True)
    ap.add_argument("--query-shard-count", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--grad-microbatch-size", type=int, default=4)
    ap.add_argument("--query-term-batch-size", type=int, default=32)
    args = ap.parse_args()
    all_query_ids = tuple(range(10))
    query_ids = all_query_ids[
        args.query_shard_index :: args.query_shard_count
    ]
    if not query_ids:
        raise ValueError("empty query shard")

    worker = load_worker()
    worker.TBA_QUERY_IDS = query_ids
    worker.TBA_DIRECTION_COUNT = 1
    worker.TBA_PURE_IMPLIED_NOISE = True
    worker.NPA_TIMESTAMP_INDICES = tuple(range(TRAJ_SNAPSHOTS))
    worker.NPA_TIMESTAMP_GROUPS = {
        "all": tuple(range(TRAJ_SNAPSHOTS)),
        **{
            f"q{quarter + 1}": tuple(
                range(quarter * 25, (quarter + 1) * 25)
            )
            for quarter in range(4)
        },
    }
    worker.NPA_CHECKPOINT_PAIRS = tuple(range(49))
    worker.TBA_VERSION = 3

    def root(_timestamp_shard_index, _timestamp_shard_count):
        return (
            ATTR_DIR
            / "_tracin_das_pure_trajectory_direction_full_q00_q09_shards"
            / f"query_shard_{args.query_shard_index:02d}_of_"
              f"{args.query_shard_count:02d}"
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
