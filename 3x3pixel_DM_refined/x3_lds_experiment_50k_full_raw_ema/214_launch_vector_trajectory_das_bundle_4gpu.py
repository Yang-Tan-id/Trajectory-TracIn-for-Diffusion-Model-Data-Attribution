"""Four-GPU launcher for vector trajectory DAS Bundle."""

import argparse
import subprocess
import sys

from exp_config import LOG_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--family", default="prompted")
    parser.add_argument("--query-ids", default="0-9")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lambda", dest="damping", type=float, default=100.0)
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / "vector_trajectory_das_bundle_q00_q09_4gpu.log"
    with open(log, "a", buffering=1) as stream:
        workers = []
        for shard, gpu in enumerate(gpus):
            command = [sys.executable, "-u", "213_run_vector_trajectory_das_bundle.py", "--gpu", str(gpu), "--family", args.family, "--query-ids", args.query_ids, "--batch-size", str(args.batch_size), "--lambda", str(args.damping), "--train-shard-index", str(shard), "--train-shard-count", str(len(gpus)), "--train-only"]
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            workers.append(process)
            print(f"[launcher] shard={shard} gpu={gpu} pid={process.pid}", flush=True)
        for shard, process in enumerate(workers):
            code = process.wait()
            print(f"[launcher] shard={shard} code={code}", flush=True)
            if code:
                raise SystemExit(code)
    command = [sys.executable, "-u", "213_run_vector_trajectory_das_bundle.py", "--gpu", str(gpus[0]), "--family", args.family, "--query-ids", args.query_ids, "--batch-size", str(args.batch_size), "--lambda", str(args.damping), "--train-shard-count", str(len(gpus)), "--use-train-cache"]
    with open(log, "a", buffering=1) as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(f"[done] log={log}", flush=True)


if __name__ == "__main__":
    main()
