"""Print and validate the Bundle TracIn experiment contract."""

import json

import numpy as np

from attribution_one_query import model_paths
from bundle_tracin_config import *
from forward_loss_alignment_config import replay_noise_path, replay_t_path


def main():
    print(f"method = {BUNDLE_TRACIN_METHOD}")
    print(f"default queries = q{BUNDLE_TRACIN_QUERY_IDS[0]:02d}-q{BUNDLE_TRACIN_QUERY_IDS[-1]:02d}")
    print(f"replayed training events/checkpoint = {BUNDLE_TRACIN_TRAIN_EVENTS}")
    print("training t/noise = exact four realized events from the checkpoint interval")
    print("training events are not forced to align with the query trajectory")
    print(f"parameter source = {BUNDLE_TRACIN_PARAM_SOURCE}")
    print(f"projection dimension = {BUNDLE_TRACIN_PROJ_DIM}")
    print(f"output vector dimension = {BUNDLE_TRACIN_OUTPUT_DIM}")
    print("checkpoint contraction = signed vector sum with checkpoint LR")
    print("subset contraction = mean_t ||sum_i a_i,t||_2^2")

    if BUNDLE_TRACIN_TRAIN_EVENTS != BASE_SAVE_EVERY_EPOCHS:
        raise ValueError("replayed event count must match epochs per checkpoint interval")
    if BUNDLE_TRACIN_PROJ_DIM <= 0:
        raise ValueError("projection dimension must be positive")
    if BUNDLE_TRACIN_PARAM_SOURCE != "raw":
        raise ValueError("this experiment is defined on raw checkpoint parameters")

    if MASK_DIR.joinpath("membership.npy").is_file():
        membership = np.load(MASK_DIR / "membership.npy", mmap_mode="r")
        expected = (SUBSETS_PER_SEED * len(LDS_SEEDS), N_TRAIN)
        if membership.shape != expected:
            raise ValueError(f"membership shape={membership.shape}, expected={expected}")
        print(f"membership = {membership.shape}")
    else:
        print(f"[not local] {MASK_DIR / 'membership.npy'}")

    if QUERY_DIR.joinpath("manifest.json").is_file():
        with open(QUERY_DIR / "manifest.json") as handle:
            records = json.load(handle)
        if len(records) != len(INITIAL_SEEDS):
            raise ValueError(f"query count={len(records)}, expected={len(INITIAL_SEEDS)}")
        print(f"query manifest = {len(records)} queries")
    else:
        print(f"[not local] {QUERY_DIR / 'manifest.json'}")

    replay_paths = (replay_t_path(), replay_noise_path())
    expected_shapes = (
        (50, N_TRAIN, BUNDLE_TRACIN_TRAIN_EVENTS),
        (50, N_TRAIN, BUNDLE_TRACIN_TRAIN_EVENTS, 3, 3, 3),
    )
    for path, expected_shape in zip(replay_paths, expected_shapes):
        if path.is_file():
            value = np.load(path, mmap_mode="r")
            if value.shape != expected_shape:
                raise ValueError(f"{path}: shape={value.shape}, expected={expected_shape}")
            print(f"replay cache = {path} {value.shape}")
        else:
            print(f"[missing replay cache] {path}")
    if not all(path.is_file() for path in replay_paths):
        print(
            "[prepare on server] python -u 24_prepare_forward_loss_alignment.py "
            "--skip-queries --skip-baseline"
        )

    for family in FAMILIES:
        paths = model_paths(family)
        if paths and len(paths) != 50:
            raise ValueError(f"{family}: found {len(paths)} checkpoints, expected 50")
        print(f"{family} checkpoints = {len(paths) if paths else '[not local]'}")
    print("[OK] Bundle TracIn contract verified")


if __name__ == "__main__":
    main()
