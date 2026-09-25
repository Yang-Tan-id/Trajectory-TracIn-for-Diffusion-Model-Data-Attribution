"""Run backward first-order projected Traj over four timestamp-sharded GPUs."""

import json
import subprocess
import sys
import time

from exp_config import CUDA_IDS, LOG_DIR, QUERY_DIR


def main():
    if len(CUDA_IDS) < 4:
        raise ValueError("the backward Traj launcher requires four CUDA_IDS")
    with open(QUERY_DIR / "manifest.json") as handle:
        queries = json.load(handle)
    if len(queries) != 100:
        raise ValueError(f"expected 100 queries, found {len(queries)}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "projected_traj_backward_100q.log"
    assignments = (
        ("prompted", 0, CUDA_IDS[0]),
        ("prompted", 1, CUDA_IDS[1]),
        ("unprompted", 0, CUDA_IDS[2]),
        ("unprompted", 1, CUDA_IDS[3]),
    )
    with open(log_path, "a", buffering=1) as stream:
        stream.write("\n[launcher] starting four-GPU backward first-order projected Traj\n")
        active = {}
        for family, shard, gpu in assignments:
            label = f"backward-{family}-shard-{shard}"
            command = [
                sys.executable,
                "run_projected_traj_bank.py",
                "--family", family,
                "--gpu", str(gpu),
                "--timestamp-shard-index", str(shard),
                "--timestamp-shard-count", "2",
                "--checkpoint-direction", "backward",
            ]
            stream.write(f"[launcher] {label}: {' '.join(command)}\n")
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            active[label] = process
            print(f"[launcher] started {label} pid={process.pid}", flush=True)
        print(f"[launcher] backward Traj log: {log_path}", flush=True)

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
            command = [
                sys.executable,
                "merge_projected_traj_shards.py",
                "--family", family,
                "--timestamp-shard-count", "2",
                "--checkpoint-direction", "backward",
            ]
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print("[done] backward first-order projected Traj merged", flush=True)


if __name__ == "__main__":
    main()
