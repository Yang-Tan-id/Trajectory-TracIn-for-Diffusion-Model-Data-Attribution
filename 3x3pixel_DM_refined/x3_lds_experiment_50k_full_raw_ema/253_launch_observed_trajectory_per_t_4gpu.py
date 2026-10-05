"""Four-GPU launcher for per-timestamp observed trajectory LDS targets."""

import argparse
import subprocess
import sys
from pathlib import Path

from exp_config import LOG_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--query-ids", default="0-9")
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(",")]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    procs = []
    logs = []
    for shard, gpu in enumerate(gpus):
        log_path = LOG_DIR / f"observed_trajectory_per_t_shard_{shard:02d}.log"
        handle = open(log_path, "w")
        command = [
            sys.executable, "-u", "251_collect_observed_trajectory_per_t.py",
            "--gpu", str(gpu),
            "--query-shard-index", str(shard),
            "--query-shard-count", str(len(gpus)),
            "--query-ids", args.query_ids,
        ]
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        procs.append((process, log_path))
        logs.append(handle)
        print(f"[launcher] gpu={gpu} pid={process.pid} log={log_path}", flush=True)
    failed = False
    for process, log_path in procs:
        code = process.wait()
        print(f"[launcher] {log_path.name} code={code}", flush=True)
        failed |= code != 0
    for handle in logs:
        handle.close()
    if failed:
        raise SystemExit("one or more workers failed")
    subprocess.run(
        [sys.executable, "-u", "252_eval_observed_trajectory_t_contribution.py", "--query-ids", args.query_ids],
        check=True,
    )


if __name__ == "__main__":
    main()
