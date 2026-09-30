"""Print and validate the Bundle TracIn experiment contract."""

import json

import numpy as np

from attribution_one_query import model_paths
from bundle_tracin_config import *


def main():
    print(f"method = {BUNDLE_TRACIN_METHOD}")
    print(f"default queries = q{BUNDLE_TRACIN_QUERY_IDS[0]:02d}-q{BUNDLE_TRACIN_QUERY_IDS[-1]:02d}")
    print(f"training gradient MC = {BUNDLE_TRACIN_TRAIN_MC}")
    print("training t/noise = independent of query trajectory; averaged before JVP")
    print(f"parameter source = {BUNDLE_TRACIN_PARAM_SOURCE}")
    print(f"projection dimension = {BUNDLE_TRACIN_PROJ_DIM}")
    print(f"output vector dimension = {BUNDLE_TRACIN_OUTPUT_DIM}")
    print("checkpoint contraction = signed vector sum with checkpoint LR")
    print("subset contraction = mean_t ||sum_i a_i,t||_2^2")

    if BUNDLE_TRACIN_TRAIN_MC <= 0 or BUNDLE_TRACIN_PROJ_DIM <= 0:
        raise ValueError("MC count and projection dimension must be positive")
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

    for family in FAMILIES:
        paths = model_paths(family)
        if paths and len(paths) != 50:
            raise ValueError(f"{family}: found {len(paths)} checkpoints, expected 50")
        print(f"{family} checkpoints = {len(paths) if paths else '[not local]'}")
    print("[OK] Bundle TracIn contract verified")


if __name__ == "__main__":
    main()
