from __future__ import annotations

import os
from copy import deepcopy

from dataset_config_attr40 import *  # noqa: F401,F403
from dataset_config_attr40 import (
    ATTRIBUTION_INDICES_PATH,
    ATTRIBUTION_SAMPLE_DIR,
    CHECKPOINT_DIR,
    COMMON_CIFAR,
    INITIAL_SEED,
    QUERY,
    REFERENCE_CKPT,
)


def _float_list(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = os.environ.get(name, "").strip()
    return default if not value else tuple(
        float(item) for item in value.replace(",", " ").split()
    )


def _int_list(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.environ.get(name, "").strip()
    return default if not value else tuple(
        int(item) for item in value.replace(",", " ").split()
    )


# Exact model-evaluation grid used by the repository's 50-step DDIM sampler.
# Stored in ascending order so independently produced timestamp shards merge
# into the same term order as query artifacts.
DDIM_50_TIMESTEPS = tuple(range(19, 1000, 20))
if len(DDIM_50_TIMESTEPS) != 50:
    raise AssertionError("CelebA DAS must use exactly 50 DDIM timesteps")

DAS_DAMPING_SWEEP_VALUES = _float_list(
    "DAS_DAMPING_SWEEP_VALUES",
    (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000),
)

ATTRIBUTION_CONFIGS = {
    "das": {
        **COMMON_CIFAR,
        "baseline_dir": CHECKPOINT_DIR,
        "reference_ckpt": REFERENCE_CKPT,
        "parameter_source": "raw",
        "attribution_sample_dir": ATTRIBUTION_SAMPLE_DIR,
        "attribution_sample_seed": INITIAL_SEED,
        "attribution_sample_index": 0,
        "attribution_use_trajectory_endpoint": True,
        "timesteps_total": 1000,
        "ddim_steps": 50,
        "timesteps": _int_list("DAS_TIMESTEPS", DDIM_50_TIMESTEPS),
        "num_mc_noise": int(os.environ.get("DAS_NUM_MC_NOISE", "1")),
        "proj_dim": int(os.environ.get("DAS_PROJ_DIM", "4096")),
        "damping": float(os.environ.get("DAS_DAMPING", "2")),
        "damping_sweep_values": DAS_DAMPING_SWEEP_VALUES,
        "batch_size": int(os.environ.get("DAS_BATCH_SIZE", "64")),
        "use_batched_per_example_grads": True,
        "per_example_grad_batch_size": int(os.environ.get("DAS_GRAD_BATCH_SIZE", "8")),
        # This is the original projected DAS score.  The optional leave-one-out
        # Sherman--Morrison correction is a separate variant and is disabled.
        "use_sherman_morrison_denominator": False,
        "max_num_ckpts": 1,
        "random_subset": False,
        "max_train_points": 5000,
        "score_indices_path": str(ATTRIBUTION_INDICES_PATH),
        "topk": 5000,
        "query": QUERY,
    },
}


def attribution_config(algorithm: str) -> dict:
    return deepcopy(ATTRIBUTION_CONFIGS[algorithm])


def commands_for_algorithm(algorithm: str) -> dict[str, list[str]]:
    if algorithm != "das":
        raise KeyError(algorithm)
    return {
        "train_datapoint_gradient": ["python", "01_train_datapoint_gradient.py"],
        "query_gradient": ["python", "02_query_gradient.py"],
        "score": ["python", "03_score.py"],
    }

