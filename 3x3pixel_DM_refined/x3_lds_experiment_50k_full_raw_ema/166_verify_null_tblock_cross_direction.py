"""Verify prerequisites and print the 40-model experiment definition."""

import torch

from null_tblock_cross_direction_config import *
from null_same_direction_learning_config import nsdl_checkpoint_path


def main():
    if len(NTCD_TIMESTAMP_BLOCKS) != 4:
        raise ValueError("expected four timestamp blocks")
    flattened = [value for block in NTCD_TIMESTAMP_BLOCKS for value in block]
    if flattened != list(range(T)):
        raise ValueError("timestamp blocks must cover 0..999 exactly")
    checkpoint_path = nsdl_checkpoint_path()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing = {"model_state", "optimizer_state"} - set(checkpoint)
    if missing:
        raise KeyError(f"null checkpoint missing {sorted(missing)}")
    sources = nsdl_datapoint_indices()
    pairs = [(source, ntcd_target_index(source)) for source in sources]
    if any(source == target for source, target in pairs):
        raise ValueError("source and target must differ")
    print(f"null checkpoint = {checkpoint_path}")
    print(f"source-target pairs = {pairs}")
    print(f"timestamp blocks = {[(b[0], b[-1]) for b in NTCD_TIMESTAMP_BLOCKS]}")
    print(f"independent branches/source = {len(NTCD_TIMESTAMP_BLOCKS)}")
    print(f"updated models = {len(sources) * len(NTCD_TIMESTAMP_BLOCKS)}")
    print(f"target directions/model = {NTCD_TARGET_DIRECTION_COUNT}")
    print("optimizer = restored checkpoint AdamW state + checkpoint LR + clipping")
    print("[OK] null timestamp-block cross-direction experiment verified")


if __name__ == "__main__":
    main()
