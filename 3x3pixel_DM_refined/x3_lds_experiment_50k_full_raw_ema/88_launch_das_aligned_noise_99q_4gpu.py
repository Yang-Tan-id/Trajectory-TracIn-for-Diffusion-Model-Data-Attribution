"""Launch q00-q98 final-EMA aligned-noise DAS on four GPUs."""

import argparse
import subprocess
import sys
import time

from exp_config import CUDA_IDS, LOG_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if len(CUDA_IDS) < 4:
        raise ValueError("four CUDA_IDS are required")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "das_ema_aligned_noise_99q_4gpu.log"
    assignments = (
        ("prompted", 0, CUDA_IDS[0]),
        ("prompted", 1, CUDA_IDS[1]),
        ("unprompted", 0, CUDA_IDS[2]),
        ("unprompted", 1, CUDA_IDS[3]),
    )
    active = {}
    with open(log_path, "a", buffering=1) as stream:
        stream.write(
            "\n[launcher] q00-q98 final-EMA DAS; query/train noise aligned; "
            "100 timestamps x 10 MC; projected4096\n"
        )
        for family, shard, gpu in assignments:
            label = f"{family}-timestamp-shard-{shard}"
            command = [
                sys.executable,
                "-u",
                "86_run_das_aligned_noise_shard.py",
                "--family",
                family,
                "--gpu",
                str(gpu),
                "--timestamp-shard-index",
                str(shard),
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
            print(f"[launcher] started {label} pid={process.pid}", flush=True)
        print(f"[launcher] log={log_path}", flush=True)

        while active:
            for label, process in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[label]
                print(f"[launcher] {label} exited code={code}", flush=True)
                if code:
                    for other in active.values():
                        other.terminate()
                    for other in active.values():
                        other.wait()
                    raise SystemExit(code)
            if active:
                time.sleep(2)

        for family in ("prompted", "unprompted"):
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "87_merge_das_aligned_noise_99q.py",
                    "--family",
                    family,
                    "--timestamp-shard-count",
                    "2",
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
    subprocess.run(
        [sys.executable, "-u", "89_eval_das_aligned_noise_99q_lds.py"],
        check=True,
    )
    print("[done] q00-q98 aligned-noise DAS + LDS", flush=True)


if __name__ == "__main__":
    main()
