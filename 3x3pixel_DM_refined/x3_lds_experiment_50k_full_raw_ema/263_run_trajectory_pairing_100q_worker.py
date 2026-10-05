"""Run the 100-query pure-trajectory pairing test with separable t ranges."""

import argparse
import importlib.util
import sys
from pathlib import Path

from exp_config import ATTR_DIR
from noise_pairing_ablation_config import NPA_TIMESTAMP_GROUPS, npa100_query_ids


def load_worker():
    path = Path(__file__).with_name(
        "248_run_trajectory_bridge_aligned_tracin_das_shard.py"
    )
    spec = importlib.util.spec_from_file_location("trajectory_pairing_100q_worker", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--family", choices=("prompted", "unprompted"), required=True)
    parser.add_argument("--mode", choices=("aligned", "independent"), required=True)
    parser.add_argument("--timestamp-shard-index", type=int, required=True)
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-microbatch-size", type=int, default=4)
    parser.add_argument("--query-term-batch-size", type=int, default=128)
    args = parser.parse_args()

    worker = load_worker()
    selected_groups = {
        name: tuple(NPA_TIMESTAMP_GROUPS[name])
        for name in ("q1", "q2", "q3", "q4")
    }
    selected_groups["q1_q2"] = tuple(
        value for name in ("q1", "q2") for value in selected_groups[name]
    )
    selected_groups["q1_q3"] = tuple(
        value for name in ("q1", "q2", "q3") for value in selected_groups[name]
    )
    selected_groups["all"] = tuple(
        value for name in ("q1", "q2", "q3", "q4")
        for value in selected_groups[name]
    )
    worker.NPA_TIMESTAMP_GROUPS = selected_groups
    worker.NPA_TIMESTAMP_INDICES = selected_groups["all"]
    worker.TBA_QUERY_IDS = npa100_query_ids(args.family)
    worker.TBA_DIRECTION_COUNT = 1
    # The shared worker accumulates all three contractions internally.  The
    # 100-query report intentionally evaluates timestamp-sum-square only.
    worker.TBA_CONTRACTIONS = (
        "linear", "termwise_squared", "timestamp_sum_squared"
    )
    worker.TBA_PURE_IMPLIED_NOISE = True
    worker.TBA_TRAIN_NOISE_MODE = args.mode
    worker.TBA_VERSION = 6

    def root(shard_index, shard_count):
        return (
            ATTR_DIR
            / "_tracin_das_trajectory_pairing_100q_shards"
            / args.mode
            / args.family
            / f"shard_{shard_index:02d}_of_{shard_count:02d}"
        )

    worker.tba_shard_root = root
    sys.argv = [
        str(Path(__file__)),
        "--gpu", str(args.gpu),
        "--family", args.family,
        "--timestamp-shard-index", str(args.timestamp_shard_index),
        "--timestamp-shard-count", str(args.timestamp_shard_count),
        "--batch-size", str(args.batch_size),
        "--grad-microbatch-size", str(args.grad_microbatch_size),
        "--query-term-batch-size", str(args.query_term_batch_size),
    ]
    worker.main()


if __name__ == "__main__":
    main()
