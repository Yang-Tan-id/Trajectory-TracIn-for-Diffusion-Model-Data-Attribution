"""Launch restartable family-bank attribution without retraining models."""

import json
import subprocess
import sys
import time

from exp_config import CUDA_IDS, LOG_DIR, QUERY_DIR


def run_all():
    if len(CUDA_IDS) < 4:
        raise ValueError("the bank launcher requires four CUDA_IDS")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    traj_log_path = LOG_DIR / "projected_traj_100q.log"
    das_log_path = LOG_DIR / "das_100q.log"
    traj_log = open(traj_log_path, "a", buffering=1)
    das_log = open(das_log_path, "a", buffering=1)
    traj_log.write("\n[launcher] starting projected Traj prompted/unprompted workers\n")
    das_log.write("\n[launcher] starting DAS prompted/unprompted workers\n")

    commands = []
    for family, gpu in zip(("prompted", "unprompted"), CUDA_IDS[:2]):
        commands.append((
            f"traj-{family}",
            [sys.executable, "run_projected_traj_bank.py", "--family", family, "--gpu", str(gpu)],
            traj_log,
        ))
    for family, gpu in zip(("prompted", "unprompted"), CUDA_IDS[2:4]):
        commands.append((
            f"das-{family}",
            [sys.executable, "run_das_bank.py", "--family", family, "--gpu", str(gpu)],
            das_log,
        ))

    workers = []
    for label, command, stream in commands:
        stream.write(f"[launcher] {label}: {' '.join(command)}\n")
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        workers.append((label, process))
        print(f"[launcher] started {label} pid={process.pid}", flush=True)
    print(f"[launcher] Traj log: {traj_log_path}", flush=True)
    print(f"[launcher] DAS log:  {das_log_path}", flush=True)

    active = dict(workers)
    try:
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
    finally:
        traj_log.close()
        das_log.close()


def main():
    with open(QUERY_DIR / "manifest.json") as handle:
        queries = json.load(handle)
    if len(queries) != 100:
        raise ValueError(f"expected 100 queries, found {len(queries)}")

    # Four concurrent workers: Traj prompted/unprompted and DAS
    # prompted/unprompted. Each shares train work across its query family.
    run_all()
    print("[done] projected Traj + DAS family-bank attribution", flush=True)


if __name__ == "__main__":
    main()
