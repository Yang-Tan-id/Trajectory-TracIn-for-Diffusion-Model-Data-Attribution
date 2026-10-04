"""Verify prerequisites and print the exact Diffusion-ReTrac contract."""

import json

import numpy as np

from attribution_one_query import model_paths
from diffusion_retrac_config import *
from forward_loss_alignment_config import replay_noise_path, replay_t_path


def main():
    for family in RETRAC_FAMILIES:
        paths = model_paths(family)
        if len(paths) != 50:
            raise ValueError(f"expected 50 {family} checkpoints, found {len(paths)}")
    with open(QUERY_DIR / "manifest.json") as handle:
        records = {int(item["query_id"]): item for item in json.load(handle)}
    for query_id in RETRAC_QUERY_IDS:
        if query_id not in records:
            raise FileNotFoundError(f"missing q{query_id:02d} manifest record")
        expected_family = "prompted" if query_id < 75 else "unprompted"
        if records[query_id]["family"] != expected_family:
            raise ValueError(f"q{query_id:02d} is not {expected_family}")
        endpoint = QUERY_DIR / f"q{query_id:02d}" / "final_state.npy"
        if not endpoint.is_file():
            raise FileNotFoundError(endpoint)
    for path in (replay_t_path(), replay_noise_path()):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing; run: python -u 24_prepare_forward_loss_alignment.py "
                "--skip-queries --skip-baseline"
            )
    t_shape = np.load(replay_t_path(), mmap_mode="r").shape
    noise_shape = np.load(replay_noise_path(), mmap_mode="r").shape
    expected_t = (50, N_TRAIN, RETRAC_EVENTS_PER_CHECKPOINT)
    expected_noise = (*expected_t, 3, 3, 3)
    if t_shape != expected_t or noise_shape != expected_noise:
        raise ValueError(
            f"replay cache shape mismatch: t={t_shape}, noise={noise_shape}, "
            f"expected={expected_t}/{expected_noise}"
        )
    print(f"queries = q00-q99 ({len(RETRAC_QUERY_IDS)})")
    print("families = q00-q74 prompted; q75-q99 unprompted")
    print("checkpoints = 50 raw checkpoints per family")
    print(f"query timesteps = {len(RETRAC_TIMESTEPS)} evenly spaced in [1, 999]")
    print(f"query noise MC = {RETRAC_QUERY_MC}")
    print("train L = four replayed training events/checkpoint/datapoint")
    print("train t/noise = exact cached training t_train and epsilon_train")
    print(f"projection = shared CountSketch{RETRAC_PROJ_DIM} per checkpoint")
    print("Diffusion-TracIn = raw query/train loss gradients")
    print("Diffusion-ReTrac = per-timestep query and per-event train gradients L2-normalized")
    print(
        "AdamW-ReTrac = unchanged normalized query gradient; each exact train "
        "event becomes a saved-state full-AdamW update, then is L2-normalized"
    )
    print("query AdamW transform = false")
    print("checkpoint weight = saved checkpoint learning rate")
    print("[OK] Diffusion-TracIn/ReTrac replay prerequisites verified")


if __name__ == "__main__":
    main()
