"""Run a 10-interval first-order next/previous Traj comparison on four GPUs."""

import json
import subprocess
import sys
import time

from exp_config import CUDA_IDS, LOG_DIR, QUERY_DIR


PAIR_INDICES = (0, 5, 11, 16, 21, 27, 32, 37, 43, 48)
PAIR_ARGUMENT = ",".join(str(value) for value in PAIR_INDICES)
METHOD_SUFFIX = "10interval_even"


def run_direction(direction, stream):
    assignments = (
        ("prompted", 0, CUDA_IDS[0]),
        ("prompted", 1, CUDA_IDS[1]),
        ("unprompted", 0, CUDA_IDS[2]),
        ("unprompted", 1, CUDA_IDS[3]),
    )
    active = {}
    for family, shard, gpu in assignments:
        label = f"{direction}-{family}-shard-{shard}"
        command = [
            sys.executable,
            "run_projected_traj_bank.py",
            "--family",
            family,
            "--gpu",
            str(gpu),
            "--timestamp-shard-index",
            str(shard),
            "--timestamp-shard-count",
            "2",
            "--checkpoint-direction",
            direction,
            "--checkpoint-pair-indices",
            PAIR_ARGUMENT,
            "--first-order-only",
            "--output-suffix",
            METHOD_SUFFIX,
        ]
        stream.write(f"[launcher] {label}: {' '.join(command)}\n")
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        active[label] = process
        print(f"[launcher] started {label} pid={process.pid}", flush=True)

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
            "--family",
            family,
            "--timestamp-shard-count",
            "2",
            "--checkpoint-direction",
            direction,
            "--checkpoint-pair-indices",
            PAIR_ARGUMENT,
            "--first-order-only",
            "--output-suffix",
            METHOD_SUFFIX,
        ]
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    if len(CUDA_IDS) < 4:
        raise ValueError("the 10-interval launcher requires four CUDA_IDS")
    with open(QUERY_DIR / "manifest.json") as handle:
        queries = json.load(handle)
    if len(queries) != 100:
        raise ValueError(f"expected 100 queries, found {len(queries)}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "projected_traj_10interval_next_previous.log"
    with open(log_path, "a", buffering=1) as stream:
        stream.write(
            "\n[launcher] 10 evenly spaced transition positions "
            f"{PAIR_INDICES}; first-order next then previous\n"
        )
        print(f"[launcher] log: {log_path}", flush=True)
        run_direction("forward", stream)
        run_direction("backward", stream)

    subprocess.run(
        [
            sys.executable,
            "-u",
            "47_eval_next_previous_alpha_sweep.py",
            "--source-method-suffix",
            METHOD_SUFFIX,
            "--output-suffix",
            METHOD_SUFFIX,
            "--result-stem",
            "traj_next_previous_alpha_sweep_10interval_even",
        ],
        check=True,
    )
    print(
        "[done] 10-interval next/previous Traj scores and alpha sweep",
        flush=True,
    )


if __name__ == "__main__":
    main()
