"""Verify prerequisites for endpoint multi-direction response estimation."""

from null_same_direction_learning_config import nsdl_checkpoint_path
from endpoint_direction_mc_config import *


def main():
    if not nsdl_checkpoint_path().is_file():
        raise FileNotFoundError(nsdl_checkpoint_path())
    for source_index in nsdl_datapoint_indices():
        source_dir = ntcd_source_dir(source_index, "fresh_sgd")
        result_path = source_dir / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
    print(f"source updates = {len(nsdl_datapoint_indices())}")
    print(f"branches/source = {len(NTCD_TIMESTAMP_BLOCKS)}")
    print(f"endpoint directions = {EDMC_DIRECTION_COUNT}")
    print(f"noise levels = {T}")
    print(f"MC subset counts = {EDMC_SUBSET_COUNTS}")
    print("endpoint = paired target datapoint with its own prompt")
    print("ground truth = finite updated-minus-null predicted-noise response")
    print("[OK] endpoint direction-MC prerequisites verified")


if __name__ == "__main__":
    main()
