"""Launch four reference-trajectory MC4 timestamp shards."""

import argparse
import subprocess
import sys
import time

from reference_traj_mc4_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=REF_MC4_DEFAULT_EPSILON)
    parser.add_argument("--batch-size", type=int, default=REF_MC4_BATCH_SIZE)
    parser.add_argument("--train-mc", type=int, default=REF_MC4_DEFAULT_TRAIN_MC)
    args = parser.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / (
        f"reference_traj_mc4_eps_{epsilon_tag(args.epsilon)}_"
        f"train_mc{args.train_mc}_4gpu.log"
    )
    active = []
    with open(log_path, "a", buffering=1) as stream:
        for shard_index, gpu in enumerate(CUDA_IDS[:4]):
            command = [
                sys.executable, "-u", "77_run_reference_traj_mc4_shard.py",
                "--gpu", str(gpu),
                "--timestamp-shard-index", str(shard_index),
                "--timestamp-shard-count", "4",
                "--epsilon", str(args.epsilon),
                "--batch-size", str(args.batch_size),
                "--train-mc", str(args.train_mc),
            ]
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            active.append((shard_index, process))
            print(f"[launcher] shard={shard_index} gpu={gpu} pid={process.pid}", flush=True)
        print(f"[launcher] log={log_path}", flush=True)
        while active:
            for item in list(active):
                shard_index, process = item
                code = process.poll()
                if code is None:
                    continue
                active.remove(item)
                print(f"[launcher] shard={shard_index} code={code}", flush=True)
                if code:
                    for _, other in active:
                        other.terminate()
                    raise SystemExit(code)
            if active:
                time.sleep(2)
        subprocess.run(
            [
                sys.executable, "-u", "78_merge_reference_traj_mc4.py",
                "--epsilon", str(args.epsilon),
                "--train-mc", str(args.train_mc),
            ],
            stdout=stream, stderr=subprocess.STDOUT, check=True,
        )
    print("[done] reference-trajectory MC4 scores merged", flush=True)


if __name__ == "__main__":
    main()
