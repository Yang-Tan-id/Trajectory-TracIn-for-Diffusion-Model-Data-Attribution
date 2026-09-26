"""Collect 100-query observed LDS responses on four query shards."""

import subprocess
import sys
import time

from exp_config import CUDA_IDS, LOG_DIR


def main():
    if len(CUDA_IDS) < 4:
        raise ValueError("the observed-output launcher requires four CUDA_IDS")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "collect_100q_lds_4gpu.log"
    with open(log_path, "a", buffering=1) as stream:
        stream.write("\n[launcher] starting four-GPU observed LDS collection\n")
        active = {}
        for shard, gpu in enumerate(CUDA_IDS[:4]):
            label = f"observed-shard-{shard}"
            command = [
                sys.executable,
                "05_collect_subset_outputs.py",
                "--gpu", str(gpu),
                "--query-shard-index", str(shard),
                "--query-shard-count", "4",
            ]
            stream.write(f"[launcher] {label}: {' '.join(command)}\n")
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            active[label] = process
            print(f"[launcher] started {label} pid={process.pid}", flush=True)
        print(f"[launcher] log: {log_path}", flush=True)

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

        subprocess.run(
            [sys.executable, "05_merge_subset_outputs.py"],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("[done] four-GPU observed LDS collection merged", flush=True)


if __name__ == "__main__":
    main()
