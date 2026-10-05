"""Launch the separable q1-q4 pure-trajectory pairing experiment for q00-q99."""

import argparse
import subprocess
import sys
import time

from exp_config import LOG_DIR


def parse_gpus(text):
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if len(values) != 4 or len(set(values)) != 4:
        raise ValueError("--gpus must contain four distinct CUDA IDs")
    return values


def run(family, mode, gpus, args, stream):
    active = {}
    for shard, gpu in enumerate(gpus):
        command = [
            sys.executable, "-u", "263_run_trajectory_pairing_100q_worker.py",
            "--gpu", str(gpu), "--family", family, "--mode", mode,
            "--timestamp-shard-index", str(shard),
            "--timestamp-shard-count", str(len(gpus)),
            "--batch-size", str(args.batch_size),
            "--grad-microbatch-size", str(args.grad_microbatch_size),
            "--query-term-batch-size", str(args.query_term_batch_size),
        ]
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        label = f"{mode}-{family}-timestamp-shard-{shard}"
        active[label] = process
        print(f"[launcher] {label} gpu={gpu} pid={process.pid}", flush=True)
    while active:
        for label, process in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            del active[label]
            print(f"[launcher] {label} code={code}", flush=True)
            if code:
                for other in active.values():
                    other.terminate()
                for other in active.values():
                    other.wait()
                raise SystemExit(code)
        if active:
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-microbatch-size", type=int, default=4)
    parser.add_argument("--query-term-batch-size", type=int, default=128)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()
    gpus = parse_gpus(args.gpus)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / "trajectory_pairing_100q_4gpu.log"
    with open(log, "a", buffering=1) as stream:
        stream.write(
            "\n[launcher] q00-q99; q1-q4 saved separately; 10 checkpoint pairs; "
            "pure trajectory direction; aligned vs independent train noise; "
            "full AdamW; projected4096; timestamp-sum-square\n"
        )
        if not args.skip_run:
            for family in ("prompted", "unprompted"):
                for mode in ("aligned", "independent"):
                    run(family, mode, gpus, args, stream)
        subprocess.run(
            [sys.executable, "-u", "264_merge_eval_trajectory_pairing_100q.py"],
            check=True, stdout=stream, stderr=subprocess.STDOUT,
        )
    print(f"[done] results and paired p-values written to {log}", flush=True)


if __name__ == "__main__":
    main()
