"""Collect staged observed outputs on four GPUs and merge query shards."""

import json
import subprocess
import sys
import time

import numpy as np

from staged_lds_config import *

METRICS = (
    "simple_loss_ema", "simple_loss_raw", "traj_ref_ema", "traj_ref_raw",
    "endpoint_deviation_ema", "endpoint_deviation_raw",
    "trajectory_state_mse_ema", "trajectory_state_mse_raw",
)


def main():
    STAGED_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = STAGED_LOG_DIR / "collect_observed_4gpu.log"
    with open(log_path, "a", buffering=1) as stream:
        active = []
        for shard, gpu in enumerate(CUDA_IDS[:4]):
            command = [sys.executable, "-u", "64_collect_staged_observed_worker.py", "--gpu", str(gpu), "--query-shard-index", str(shard)]
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            active.append((shard, process))
            print(f"[launcher] observed shard={shard} gpu={gpu} pid={process.pid}", flush=True)
        while active:
            for item in list(active):
                shard, process = item
                code = process.poll()
                if code is None:
                    continue
                active.remove(item)
                print(f"[launcher] shard={shard} code={code}", flush=True)
                if code:
                    raise SystemExit(code)
            if active:
                time.sleep(2)
    with open(STAGED_QUERY_DIR / "manifest.json") as handle:
        queries = json.load(handle)
    STAGED_LDS_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {metric: np.empty((len(queries), STAGED_LDS_MASK_COUNT), dtype=np.float64) for metric in METRICS}
    for position, query in enumerate(queries):
        qid = int(query["query_id"])
        with np.load(STAGED_LDS_DIR / "observed_query_shards" / f"q{qid:02d}.npz") as payload:
            for metric in METRICS:
                arrays[metric][position] = payload[metric]
    for metric, values in arrays.items():
        np.save(STAGED_LDS_DIR / f"observed_{metric}.npy", values)
        print(f"[saved] observed_{metric}.npy {values.shape}", flush=True)


if __name__ == "__main__":
    main()
