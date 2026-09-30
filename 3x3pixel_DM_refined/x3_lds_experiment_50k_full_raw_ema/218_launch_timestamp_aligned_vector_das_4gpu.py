"""Dynamically schedule ten aligned vector-DAS timestamp tasks on four GPUs."""

import argparse
import subprocess
import sys
import time

from exp_config import LOG_DIR
from timestamp_aligned_vector_das_config import TAVD_POSITIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--family", default="prompted")
    parser.add_argument("--query-ids", default="0-9")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    gpus = [int(v) for v in args.gpus.split(",") if v.strip()]
    pending = list(range(len(TAVD_POSITIONS)))
    active = {}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / "timestamp_aligned_vector_das_10t_q00_q09_4gpu.log"
    with open(log, "a", buffering=1) as stream:
        def launch(task, gpu):
            command = [sys.executable, "-u", "216_run_timestamp_aligned_vector_das_task.py", "--gpu", str(gpu), "--task-index", str(task), "--family", args.family, "--query-ids", args.query_ids, "--batch-size", str(args.batch_size)]
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            active[gpu] = (task, process)
            print(f"[launcher] task={task} gpu={gpu} pid={process.pid}", flush=True)
        for gpu in gpus:
            if pending: launch(pending.pop(0), gpu)
        print(f"[launcher] log={log}", flush=True)
        while active:
            for gpu, (task, process) in list(active.items()):
                code = process.poll()
                if code is None: continue
                del active[gpu]
                print(f"[launcher] task={task} gpu={gpu} code={code}", flush=True)
                if code:
                    for _, other in active.values(): other.terminate()
                    raise SystemExit(code)
                if pending: launch(pending.pop(0), gpu)
            if active: time.sleep(2)
    subprocess.run([sys.executable, "-u", "217_merge_eval_timestamp_aligned_vector_das.py", "--family", args.family, "--query-ids", args.query_ids], check=True)


if __name__ == "__main__": main()
