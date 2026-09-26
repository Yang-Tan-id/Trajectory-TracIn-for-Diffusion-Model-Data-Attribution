"""Merge per-query observed LDS response shards into evaluator arrays."""

import json

import numpy as np

from exp_config import LDS_DIR, MASK_DIR, QUERY_DIR


METRICS = (
    "simple_loss_ema", "simple_loss_raw",
    "traj_ref_ema", "traj_ref_raw",
    "endpoint_deviation_ema", "endpoint_deviation_raw",
    "trajectory_state_mse_ema", "trajectory_state_mse_raw",
)


def main():
    with open(QUERY_DIR / "manifest.json") as handle:
        queries = json.load(handle)
    with open(MASK_DIR / "manifest.json") as handle:
        masks = json.load(handle)
    shard_dir = LDS_DIR / "observed_query_shards"
    arrays = {
        metric: np.empty((len(queries), len(masks)), dtype=np.float64)
        for metric in METRICS
    }
    for qi in range(len(queries)):
        path = shard_dir / f"q{qi:02d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing observed query shard: {path}")
        with np.load(path, allow_pickle=False) as payload:
            actual_qid = int(np.asarray(payload["query_id"]).item())
            if actual_qid != qi:
                raise ValueError(f"{path} contains query_id={actual_qid}, expected {qi}")
            for metric in METRICS:
                value = np.asarray(payload[metric], dtype=np.float64)
                if value.shape != (len(masks),):
                    raise ValueError(f"{path}:{metric} has shape {value.shape}")
                arrays[metric][qi] = value
    LDS_DIR.mkdir(parents=True, exist_ok=True)
    for metric, value in arrays.items():
        path = LDS_DIR / f"observed_{metric}.npy"
        np.save(path, value)
        print(f"[saved] {path} shape={value.shape}", flush=True)


if __name__ == "__main__":
    main()
