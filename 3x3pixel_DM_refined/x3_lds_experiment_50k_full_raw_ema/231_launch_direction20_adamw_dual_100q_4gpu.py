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
    parser.add_argument("--grad-microbatch-size", type=int, default=8)
    parser.add_argument("--query-term-batch-size", type=int, default=128)
    parser.add_argument("--direction-count", type=int, default=20)
    parser.add_argument(
        "--noise-pairing",
        choices=(
            "aligned-shared-direction",
            "aligned-per-t",
            "independent-per-t",
        ),
        default="aligned-shared-direction",
    )
    parser.add_argument(
        "--query-mode",
        choices=("dual", "trajectory", "endpoint"),
        default="dual",
    )
    args = parser.parse_args()
    gpus = parse_gpus(args.gpus)
    if args.direction_count == 1:
        assignments = (
            ("prompted", 0, 1, 0, 2, gpus[0]),
            ("prompted", 0, 1, 1, 2, gpus[1]),
            ("unprompted", 0, 1, 0, 2, gpus[2]),
            ("unprompted", 0, 1, 1, 2, gpus[3]),
        )
        direction_shard_count = 1
        train_shard_count = 2
    else:
        assignments = (
            ("prompted", 0, 2, 0, 1, gpus[0]),
            ("prompted", 1, 2, 0, 1, gpus[1]),
            ("unprompted", 0, 2, 0, 1, gpus[2]),
            ("unprompted", 1, 2, 0, 1, gpus[3]),
        )
        direction_shard_count = 2
        train_shard_count = 1
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_tag = (
        "direction20_mean100t_dual_aligned"
        if args.noise_pairing == "aligned-shared-direction"
        and args.direction_count == 20
        and args.query_mode == "dual"
        else f"bank{args.direction_count}_100x1_{args.query_mode}_{args.noise_pairing}"
    )
    log_path = LOG_DIR / f"{run_tag}_adamw_full_100q_4gpu.log"
    active = {}
    with open(log_path, "a", buffering=1) as stream:
        stream.write(
            f"\n[launcher] direction_count={args.direction_count}; mean100t; "
            f"noise_pairing={args.noise_pairing}; query_mode={args.query_mode}; "
            "full AdamW; normalized next delta; norm4; q00-q99\n"
        )
        for (
            family,
            shard_index,
            shard_count,
            train_shard_index,
            worker_train_shard_count,
            gpu,
        ) in assignments:
            label = (
                f"{family}-direction-shard-{shard_index}-"
                f"train-shard-{train_shard_index}"
            )
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
                "--train-shard-index",
                str(train_shard_index),
                "--train-shard-count",
                str(worker_train_shard_count),
                "--batch-size",
                str(args.batch_size),
                "--grad-microbatch-size",
                str(args.grad_microbatch_size),
                "--query-term-batch-size",
                str(args.query_term_batch_size),
                "--direction-count",
                str(args.direction_count),
                "--noise-pairing",
                args.noise_pairing,
                "--query-mode",
                args.query_mode,
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
                str(direction_shard_count),
                "--direction-count",
                str(args.direction_count),
                "--train-shard-count",
                str(train_shard_count),
                "--noise-pairing",
                args.noise_pairing,
                "--query-mode",
                args.query_mode,
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("[done] direction20 full-AdamW dual-query run", flush=True)


if __name__ == "__main__":
    main()
