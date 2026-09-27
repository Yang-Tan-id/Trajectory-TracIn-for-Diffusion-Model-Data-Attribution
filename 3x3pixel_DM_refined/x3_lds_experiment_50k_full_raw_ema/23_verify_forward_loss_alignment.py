"""Fail-fast verification for forward-loss alignment prerequisites."""

import json

import numpy as np
import torch

from forward_loss_alignment_config import *


def main():
    missing = [str(checkpoint_path(e)) for e in FLA_CHECKPOINT_EPOCHS if not checkpoint_path(e).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} raw base checkpoints; first={missing[0]}")
    if not (QUERY_DIR / "manifest.json").is_file():
        raise FileNotFoundError(QUERY_DIR / "manifest.json")
    if not (MASK_DIR / "membership.npy").is_file():
        raise FileNotFoundError(MASK_DIR / "membership.npy")
    for metric in FLA_LDS_METRICS:
        if not (LDS_DIR / f"observed_{metric}.npy").is_file():
            raise FileNotFoundError(LDS_DIR / f"observed_{metric}.npy")

    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    records = {int(item["query_id"]): item for item in manifest}
    for qid in FLA_QUERY_IDS:
        if records[qid]["family"] != FLA_FAMILY:
            raise ValueError(f"q{qid:02d} is {records[qid]['family']}, expected {FLA_FAMILY}")
        cached = QUERY_DIR / f"q{qid:02d}" / "trajectory_xt.npy"
        if not cached.is_file():
            raise FileNotFoundError(cached)

    membership = np.load(MASK_DIR / "membership.npy", mmap_mode="r")
    if membership.shape[1] != N_TRAIN:
        raise ValueError(f"membership shape={membership.shape}; expected second dimension {N_TRAIN}")

    first = torch.load(checkpoint_path(FLA_CHECKPOINT_EPOCHS[0]), map_location="cpu")
    last = torch.load(checkpoint_path(FLA_CHECKPOINT_EPOCHS[-1]), map_location="cpu")
    if int(first["epoch"]) != FLA_CHECKPOINT_EPOCHS[0] or int(last["epoch"]) != EPOCHS:
        raise ValueError("Checkpoint epoch metadata do not match filenames")

    print(f"family = {FLA_FAMILY}")
    print(f"queries = q{FLA_QUERY_IDS[0]:02d}..q{FLA_QUERY_IDS[-1]:02d} ({len(FLA_QUERY_IDS)})")
    print(f"checkpoints = {len(FLA_CHECKPOINT_EPOCHS)} ({FLA_CHECKPOINT_EPOCHS[0]}..{FLA_CHECKPOINT_EPOCHS[-1]})")
    print(f"training events/datapoint/checkpoint = {FLA_EVENTS_PER_CHECKPOINT}")
    print(f"reference timestamps = {DDIM_STEPS}")
    print(f"artificial updates = raw SGD and global-gradient-normalized SGD")
    print("score normalizations = absolute, log-relative, loss-conditioned robust")
    print(f"loss-conditioned bins = {FLA_LOSS_CONDITION_BINS}")
    print(f"datapoint batch = {FLA_DATAPOINT_BATCH_SIZE} (forward event batch={FLA_DATAPOINT_BATCH_SIZE * FLA_EVENTS_PER_CHECKPOINT})")
    print(f"reference trajectory = cached EMA DDIM initial state, replayed at all {DDIM_STEPS} states")
    print(f"checkpoint/update/evaluation parameters = raw")
    print("[OK] forward-loss-alignment prerequisites verified")


if __name__ == "__main__":
    main()
