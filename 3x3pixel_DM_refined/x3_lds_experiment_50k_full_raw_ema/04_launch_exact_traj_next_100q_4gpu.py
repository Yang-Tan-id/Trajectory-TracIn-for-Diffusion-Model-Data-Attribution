"""Run exact/no-projection raw first-order next Traj for 100 queries on 4 GPUs."""

import subprocess
import sys
import time

from exp_config import CUDA_IDS, LOG_DIR


def main():
    if len(CUDA_IDS) < 4:
        raise ValueError("exact Traj launcher requires four CUDA_IDS")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "exact_traj_next_raw_100q_4gpu.log"
    assignments = (
        ("prompted", 0, CUDA_IDS[0]),
        ("prompted", 1, CUDA_IDS[1]),
        ("unprompted", 0, CUDA_IDS[2]),
        ("unprompted", 1, CUDA_IDS[3]),
    )
    with open(log_path, "a", buffering=1) as stream:
        stream.write("\n[launcher] exact/no-projection Traj-next raw start\n")
        active = {}
        for family, shard, gpu in assignments:
            label = f"exact-traj-{family}-shard-{shard}"
            command = [
                sys.executable, "-u", "run_exact_traj_next_bank.py",
                "--family", family,
                "--gpu", str(gpu),
                "--timestamp-shard-index", str(shard),
                "--timestamp-shard-count", "2",
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

        for family in ("prompted", "unprompted"):
            subprocess.run(
                [
                    sys.executable, "merge_exact_traj_next_shards.py",
                    "--family", family,
                    "--timestamp-shard-count", "2",
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
    print("[done] exact/no-projection Traj-next raw 100-query bank", flush=True)


if __name__ == "__main__":
    main()
