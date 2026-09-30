"""Verify fresh-SGD artifacts required for magnitude prediction."""

from null_same_direction_learning_config import nsdl_checkpoint_path
from opposite_gradient_magnitude_config import *


def main():
    if not nsdl_checkpoint_path().is_file():
        raise FileNotFoundError(nsdl_checkpoint_path())
    for source_index in nsdl_datapoint_indices():
        source_dir = ntcd_source_dir(source_index, "fresh_sgd")
        for name in ("result.json", "target_direction_prediction_deltas.npz"):
            path = source_dir / name
            if not path.is_file():
                raise FileNotFoundError(path)
    print(f"sources = {list(nsdl_datapoint_indices())}")
    print(f"timestamp blocks = {[(b[0], b[-1]) for b in NTCD_TIMESTAMP_BLOCKS]}")
    print(f"target directions/source = {NTCD_TARGET_DIRECTION_COUNT}")
    print(f"candidates = {OGM_CANDIDATES}")
    print("actual target = finite fresh-SGD updated-minus-null predicted noise")
    print("[OK] opposite-gradient magnitude prerequisites verified")


if __name__ == "__main__":
    main()
