"""Run four timestamp shards of old-experiment q00-q09 TracIn-DAS."""

import argparse
import subprocess
import sys
import time

from tracin_das_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=TRACIN_DAS_BATCH_SIZE)
    parser.add_argument("--noise-mode", choices=TRACIN_DAS_NOISE_MODES, default="checkpoint")
    parser.add_argument(
        "--parameter-projection",
        choices=TRACIN_DAS_PARAMETER_PROJECTIONS,
        default="exact",
    )
    parser.add_argument(
        "--train-noise-mode",
        choices=TRACIN_DAS_TRAIN_NOISE_MODES,
        default="aligned",
    )
    parser.add_argument("--query-mc", type=int, default=1)
    args = parser.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    mode_tag = args.noise_mode.replace("-", "_")
    projection_tag = args.parameter_projection
    train_tag = args.train_noise_mode.replace("-", "_")
    query_tag = "" if args.query_mc == 1 else f"_query_mc{args.query_mc}"
    log_path = LOG_DIR / (
        f"tracin_das_{mode_tag}_{projection_tag}{query_tag}_"
        f"{train_tag}_q00_q09_4gpu.log"
    )
    active = []
    with open(log_path, "a", buffering=1) as stream:
        stream.write("\n[launcher] TracIn-DAS q00-q09 four-GPU run\n")
        for shard_index, gpu in enumerate(CUDA_IDS[:4]):
            command = [
                sys.executable, "-u", "73_run_tracin_das_shard.py",
                "--gpu", str(gpu), "--timestamp-shard-index", str(shard_index),
                "--timestamp-shard-count", "4", "--batch-size", str(args.batch_size),
                "--noise-mode", args.noise_mode,
                "--parameter-projection", args.parameter_projection,
                "--train-noise-mode", args.train_noise_mode,
                "--query-mc", str(args.query_mc),
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
                sys.executable, "-u", "74_merge_tracin_das_shards.py",
                "--noise-mode", args.noise_mode,
                "--parameter-projection", args.parameter_projection,
                "--train-noise-mode", args.train_noise_mode,
                "--query-mc", str(args.query_mc),
            ],
            stdout=stream, stderr=subprocess.STDOUT, check=True,
        )
    print("[done] TracIn-DAS q00-q09 merged", flush=True)


if __name__ == "__main__":
    main()
