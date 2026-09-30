"""Verify artifacts needed for finite-difference magnitude transfer."""

from null_same_direction_learning_config import nsdl_checkpoint_path
from cross_direction_magnitude_transfer_config import *


def main():
    if not nsdl_checkpoint_path().is_file():
        raise FileNotFoundError(nsdl_checkpoint_path())
    for source_index in nsdl_datapoint_indices():
        source_dir = ntcd_source_dir(source_index, "fresh_sgd")
        result_path = source_dir / "result.json"
        archive_path = source_dir / "target_direction_prediction_deltas.npz"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
    print(f"sources = {list(nsdl_datapoint_indices())}")
    print(f"branches/source = {len(NTCD_TIMESTAMP_BLOCKS)}")
    print(f"target directions/branch = {NTCD_TARGET_DIRECTION_COUNT}")
    print("reference magnitude = source datapoint on its trained +epsilon axis")
    print("comparisons = source -epsilon axis and target's five independent axes")
    print("[OK] cross-direction magnitude-transfer prerequisites verified")


if __name__ == "__main__":
    main()
