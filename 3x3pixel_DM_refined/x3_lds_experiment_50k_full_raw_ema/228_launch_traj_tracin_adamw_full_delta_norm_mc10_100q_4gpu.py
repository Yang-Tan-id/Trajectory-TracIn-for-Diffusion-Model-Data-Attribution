"""Launch 100-query non-aligned MC10 full-AdamW trajectory attribution."""

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
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    gpus = parse_gpus(args.gpus)
    assignments = (
        ("prompted", 0, gpus[0]),
        ("prompted", 1, gpus[1]),
        ("unprompted", 0, gpus[2]),
        ("unprompted", 1, gpus[3]),
    )
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / (
        "traj_tracin_adamw_full_delta_norm_independent_mc10_100q_4gpu.log"
    )
    active = {}
    with open(log_path, "a", buffering=1) as stream:
        stream.write(
            "\n[launcher] reference trajectory; normalized next delta; "
            "independent train MC10; full AdamW; norm4; "
            "timestamp-sum-square; q00-q99\n"
        )
        for family, shard_index, gpu in assignments:
            label = f"{family}-timestamp-shard-{shard_index}"
            command = [
                sys.executable,
                "-u",
                "226_run_traj_tracin_adamw_full_delta_norm_mc10_shard.py",
                "--gpu",
                str(gpu),
                "--family",
                family,
                "--timestamp-shard-index",
                str(shard_index),
                "--timestamp-shard-count",
                "2",
                "--batch-size",
                str(args.batch_size),
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
                "227_merge_eval_traj_tracin_adamw_full_delta_norm_mc10.py",
                "--timestamp-shard-count",
                "2",
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("[done] non-aligned MC10 full-AdamW trajectory run", flush=True)


if __name__ == "__main__":
    main()
