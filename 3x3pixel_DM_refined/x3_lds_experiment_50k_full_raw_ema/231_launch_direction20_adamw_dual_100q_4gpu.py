"""Launch X3 direction20/mean100t full-AdamW dual-query attribution."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--query-term-batch-size", type=int, default=128)
    args = parser.parse_args()
    gpus = parse_gpus(args.gpus)
    assignments = (
        ("prompted", 0, 2, gpus[0]),
        ("prompted", 1, 2, gpus[1]),
        ("unprompted", 0, 2, gpus[2]),
        ("unprompted", 1, 2, gpus[3]),
    )
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "direction20_mean100t_adamw_full_dual_100q_4gpu.log"
    active = {}
    with open(log_path, "a", buffering=1) as stream:
        stream.write(
            "\n[launcher] direction20 x mean100t; full AdamW; normalized next delta; "
            "trajectory nonaligned + endpoint-polluted aligned; norm4; q00-q99\n"
        )
        for family, shard_index, shard_count, gpu in assignments:
            label = f"{family}-direction-shard-{shard_index}"
            command = [
                sys.executable,
                "-u",
                "229_run_direction20_adamw_dual_shard.py",
                "--gpu",
                str(gpu),
                "--family",
                family,
                "--direction-shard-index",
                str(shard_index),
                "--direction-shard-count",
                str(shard_count),
                "--batch-size",
                str(args.batch_size),
                "--query-term-batch-size",
                str(args.query_term_batch_size),
            ]
            stream.write(f"[launcher] {label}: {' '.join(command)}\n")
            process = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT
            )
            active[label] = process
            print(f"[launcher] {label} gpu={gpu} pid={process.pid}", flush=True)
        print(f"[launcher] log={log_path}", flush=True)
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
        subprocess.run(
            [
                sys.executable,
                "-u",
                "230_merge_eval_direction20_adamw_dual.py",
                "--direction-shard-count",
                "2",
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("[done] direction20 full-AdamW dual-query run", flush=True)


if __name__ == "__main__":
    main()
