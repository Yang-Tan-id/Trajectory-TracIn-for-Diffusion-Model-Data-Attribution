"""Run staged Traj or final-EMA/all-50k DAS on four GPUs."""

import argparse
import subprocess
import sys
import time

from staged_lds_config import *


def run_four(worker, shard_flag, label):
    STAGED_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = STAGED_LOG_DIR / f"{label}_4gpu.log"
    with open(log_path, "a", buffering=1) as stream:
        active = []
        for shard, gpu in enumerate(CUDA_IDS[:4]):
            command = [sys.executable, "-u", worker, "--gpu", str(gpu), shard_flag, str(shard)]
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            active.append((shard, process))
            print(f"[launcher] {label} shard={shard} gpu={gpu} pid={process.pid}", flush=True)
        print(f"[launcher] log={log_path}", flush=True)
        while active:
            for item in list(active):
                shard, process = item
                code = process.poll()
                if code is None:
                    continue
                active.remove(item)
                print(f"[launcher] {label} shard={shard} code={code}", flush=True)
                if code:
                    for _, other in active:
                        other.terminate()
                    raise SystemExit(code)
            if active:
                time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("traj", "das", "all"), default="all")
    args = parser.parse_args()
    if args.experiment in ("traj", "all"):
        run_four("66_run_staged_traj_shard.py", "--timestamp-shard-index", "staged_traj")
        subprocess.run([sys.executable, "-u", "67_merge_staged_traj.py"], check=True)
    if args.experiment in ("das", "all"):
        run_four("68_run_staged_das_shard.py", "--term-shard-index", "staged_das")
        subprocess.run([sys.executable, "-u", "69_merge_staged_das.py"], check=True)


if __name__ == "__main__":
    main()
