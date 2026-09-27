"""Run true no-projection next Traj with three contractions on four GPUs."""

import argparse
import subprocess
import sys
import time

from exp_config import CUDA_IDS, LOG_DIR, TRACIN_PROJECTED_BATCH_SIZE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size", type=int, default=TRACIN_PROJECTED_BATCH_SIZE
    )
    parser.add_argument(
        "--prompted-only",
        action="store_true",
        help="run and evaluate only prompted q00-q74",
    )
    parser.add_argument(
        "--prompted-shards",
        type=int,
        choices=(3, 4),
        default=3,
        help="timestamp shards for prompted-only mode; 4 uses all four GPUs",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not args.prompted_only and args.prompted_shards != 3:
        parser.error("--prompted-shards is only configurable with --prompted-only")
    required_gpus = args.prompted_shards if args.prompted_only else 4
    if len(CUDA_IDS) < required_gpus:
        raise ValueError(
            f"exact Traj launcher requires {required_gpus} CUDA_IDS"
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_name = (
        "exact_traj_next_three_50ckpt_prompted_q00_q74_"
        f"{args.prompted_shards}gpu.log"
        if args.prompted_only
        else "exact_traj_next_three_50ckpt_100q_4gpu.log"
    )
    log_path = LOG_DIR / log_name
    all_assignments = (
        ("prompted", 0, 3, CUDA_IDS[0]),
        ("prompted", 1, 3, CUDA_IDS[1]),
        ("prompted", 2, 3, CUDA_IDS[2]),
        ("unprompted", 0, 1, CUDA_IDS[3]),
    )
    assignments = (
        tuple(
            ("prompted", shard, args.prompted_shards, CUDA_IDS[shard])
            for shard in range(args.prompted_shards)
        )
        if args.prompted_only
        else all_assignments
    )
    with open(log_path, "a", buffering=1) as stream:
        stream.write(
            "\n[launcher] exact/no-projection first-order next Traj; "
            "50 checkpoints, 49 transitions, three contractions\n"
        )
        active = {}
        for family, shard, shard_count, gpu in assignments:
            label = f"exact-three-{family}-shard-{shard}"
            command = [
                sys.executable,
                "-u",
                "run_exact_traj_next_bank.py",
                "--family",
                family,
                "--gpu",
                str(gpu),
                "--timestamp-shard-index",
                str(shard),
                "--timestamp-shard-count",
                str(shard_count),
                "--batch-size",
                str(args.batch_size),
                "--all-contractions",
            ]
            stream.write(f"[launcher] {label}: {' '.join(command)}\n")
            process = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT
            )
            active[label] = process
            print(f"[launcher] started {label} pid={process.pid}", flush=True)
        print(f"[launcher] detailed log: {log_path}", flush=True)

        while active:
            for label, process in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[label]
                print(f"[launcher] {label} exited code={code}", flush=True)
                if code != 0:
                    for other in active.values():
                        other.terminate()
                    for other in active.values():
                        other.wait()
                    raise SystemExit(code)
            if active:
                time.sleep(1)

        merge_families = (
            (("prompted", args.prompted_shards),)
            if args.prompted_only
            else (("prompted", 3), ("unprompted", 1))
        )
        for family, shard_count in merge_families:
            subprocess.run(
                [
                    sys.executable,
                    "merge_exact_traj_next_shards.py",
                    "--family",
                    family,
                    "--timestamp-shard-count",
                    str(shard_count),
                    "--all-contractions",
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    eval_command = [sys.executable, "-u", "50_eval_exact_traj_next_lds.py"]
    if args.prompted_only:
        eval_command.append("--prompted-only")
    subprocess.run(eval_command, check=True)
    print("[done] exact next Traj three-contraction LDS", flush=True)


if __name__ == "__main__":
    main()
