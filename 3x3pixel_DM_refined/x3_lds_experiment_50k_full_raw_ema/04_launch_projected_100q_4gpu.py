"""Launch restartable family-bank attribution without retraining models."""

import json
import subprocess
import sys
import time

from exp_config import CUDA_IDS, LOG_DIR, QUERY_DIR


def run_workers(commands, log_path, phase_label):
    with open(log_path, "a", buffering=1) as stream:
        stream.write(f"\n[launcher] starting {phase_label}\n")
        workers = []
        for label, command in commands:
            stream.write(f"[launcher] {label}: {' '.join(command)}\n")
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            workers.append((label, process))
            print(f"[launcher] started {label} pid={process.pid}", flush=True)
        print(f"[launcher] {phase_label} log: {log_path}", flush=True)

        active = dict(workers)
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


def run_all():
    if len(CUDA_IDS) < 4:
        raise ValueError("the bank launcher requires four CUDA_IDS")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    traj_log_path = LOG_DIR / "projected_traj_100q.log"
    das_log_path = LOG_DIR / "das_100q.log"

    das_commands = []
    for family, gpu in zip(("prompted", "unprompted"), CUDA_IDS[:2]):
        das_commands.append((
            f"das-{family}",
            [sys.executable, "run_das_bank.py", "--family", family, "--gpu", str(gpu)],
        ))
    run_workers(das_commands, das_log_path, "DAS prompted/unprompted")

    traj_commands = []
    assignments = (
        ("prompted", 0, CUDA_IDS[0]),
        ("prompted", 1, CUDA_IDS[1]),
        ("unprompted", 0, CUDA_IDS[2]),
        ("unprompted", 1, CUDA_IDS[3]),
    )
    for family, shard, gpu in assignments:
        traj_commands.append((
            f"traj-{family}-shard-{shard}",
            [
                sys.executable,
                "run_projected_traj_bank.py",
                "--family", family,
                "--gpu", str(gpu),
                "--timestamp-shard-index", str(shard),
                "--timestamp-shard-count", "2",
            ],
        ))
    run_workers(traj_commands, traj_log_path, "four-GPU projected Traj")

    with open(traj_log_path, "a", buffering=1) as stream:
        for family in ("prompted", "unprompted"):
            command = [
                sys.executable,
                "merge_projected_traj_shards.py",
                "--family", family,
                "--timestamp-shard-count", "2",
            ]
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    with open(QUERY_DIR / "manifest.json") as handle:
        queries = json.load(handle)
    if len(queries) != 100:
        raise ValueError(f"expected 100 queries, found {len(queries)}")

    # DAS runs first (and skips if already complete), then Traj uses all four
    # GPUs as two timestamp shards per family and merges exact partial sums.
    run_all()
    print("[done] projected Traj + DAS family-bank attribution", flush=True)


if __name__ == "__main__":
    main()
