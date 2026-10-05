"""Launch classic EMA-parameter Trajectory-TracIn for q00-q99."""

import argparse
import subprocess
import sys
import time

from exp_config import LOG_DIR


SUFFIX = "mc10_100q"


def parse_gpus(value):
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(result) != 4 or len(set(result)) != 4:
        raise ValueError("--gpus must contain four distinct CUDA IDs")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=1280)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()
    gpus = parse_gpus(args.gpus)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / "traj_tracin_ema_mc10_100q_4gpu.log"

    with open(log, "a", buffering=1) as stream:
        stream.write(
            "\n[launcher] q00-q99; EMA parameters; no AdamW transform; "
            "100 trajectory timestamps; independent train MC10; "
            "projected4096; first order; three contractions\n"
        )
        if not args.skip_run:
            processes = []
            assignments = (
                ("prompted", 0, gpus[0]),
                ("prompted", 1, gpus[1]),
                ("unprompted", 0, gpus[2]),
                ("unprompted", 1, gpus[3]),
            )
            for family, shard, gpu in assignments:
                command = [
                    sys.executable, "-u", "run_projected_traj_bank.py",
                    "--family", family,
                    "--gpu", str(gpu),
                    "--parameter-source", "ema",
                    "--first-order-only",
                    "--timestamp-shard-index", str(shard),
                    "--timestamp-shard-count", "2",
                    "--batch-size", str(args.batch_size),
                    "--output-suffix", SUFFIX,
                ]
                process = subprocess.Popen(
                    command, stdout=stream, stderr=subprocess.STDOUT
                )
                label = f"{family}-timestamp-shard-{shard}"
                processes.append((label, process))
                print(f"[launcher] {label} gpu={gpu} pid={process.pid}", flush=True)
            active = dict(processes)
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

        for family in ("prompted", "unprompted"):
            subprocess.run(
                [
                    sys.executable, "-u", "merge_projected_traj_shards.py",
                    "--family", family,
                    "--parameter-source", "ema",
                    "--first-order-only",
                    "--timestamp-shard-count", "2",
                    "--output-suffix", SUFFIX,
                ],
                check=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        subprocess.run(
            [sys.executable, "-u", "267_eval_traj_tracin_ema_mc10_100q.py"],
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    print(f"[done] EMA/no-AdamW Trajectory-TracIn; log={log}", flush=True)


if __name__ == "__main__":
    main()
