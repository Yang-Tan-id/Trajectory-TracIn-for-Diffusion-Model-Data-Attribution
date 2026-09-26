"""Four-GPU launcher for both checkpoint/timestamp counterfactual experiments."""

import argparse
import subprocess
import sys
import time

from exp_config import CUDA_IDS, LOG_DIR


def run_group(label, commands, log_path):
    print(f"[launcher] {label} detailed log: {log_path}", flush=True)
    with open(log_path, "a", buffering=1) as stream:
        stream.write(f"\n[launcher] {label} start\n")
        active = {}
        for name, command in commands:
            stream.write(f"[launcher] {name}: {' '.join(command)}\n")
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            active[name] = process
            print(f"[launcher] started {name} pid={process.pid}", flush=True)
        while active:
            for name, process in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[name]
                print(f"[launcher] {name} exited code={code}", flush=True)
                if code != 0:
                    for other in active.values():
                        other.terminate()
                    for other in active.values():
                        other.wait()
                    raise SystemExit(code)
            if active:
                time.sleep(1)


def run_subset_unlearning():
    build_assignments = (
        ("prompted", 0, 2, CUDA_IDS[0]),
        ("prompted", 1, 2, CUDA_IDS[1]),
        ("unprompted", 0, 2, CUDA_IDS[2]),
        ("unprompted", 1, 2, CUDA_IDS[3]),
    )
    commands = []
    for family, shard, count, gpu in build_assignments:
        commands.append(
            (
                f"update-{family}-{shard}",
                [
                    sys.executable, "-u", "12_build_subset_unlearning_updates.py",
                    "--family", family, "--gpu", str(gpu),
                    "--timestamp-shard-index", str(shard),
                    "--timestamp-shard-count", str(count),
                ],
            )
        )
    run_group("subset-unlearning update banks", commands, LOG_DIR / "cf_subset_updates_4gpu.log")
    for family in ("prompted", "unprompted"):
        subprocess.run(
            [
                sys.executable, "12_merge_subset_unlearning_updates.py",
                "--family", family, "--timestamp-shard-count", "2",
            ],
            check=True,
        )
    response_commands = []
    for shard, gpu in enumerate(CUDA_IDS[:4]):
        response_commands.append(
            (
                f"response-shard-{shard}",
                [
                    sys.executable, "-u", "13_eval_subset_unlearning_counterfactual.py",
                    "--gpu", str(gpu), "--query-shard-index", str(shard),
                    "--query-shard-count", "4",
                ],
            )
        )
    run_group(
        "subset-unlearning counterfactual responses",
        response_commands,
        LOG_DIR / "cf_subset_responses_4gpu.log",
    )
    subprocess.run([sys.executable, "13_eval_subset_unlearning_lds.py"], check=True)


def run_delta_direction():
    assignments = (
        ("prompted", 0, 3, CUDA_IDS[0]),
        ("prompted", 1, 3, CUDA_IDS[1]),
        ("prompted", 2, 3, CUDA_IDS[2]),
        ("unprompted", 0, 1, CUDA_IDS[3]),
    )
    commands = []
    for family, shard, count, gpu in assignments:
        commands.append(
            (
                f"delta-{family}-{shard}",
                [
                    sys.executable, "-u", "14_run_last_noise_delta_direction_bank.py",
                    "--family", family, "--gpu", str(gpu),
                    "--timestamp-shard-index", str(shard),
                    "--timestamp-shard-count", str(count),
                ],
            )
        )
    run_group("last-noise delta-direction", commands, LOG_DIR / "cf_delta_direction_4gpu.log")
    for family, count in (("prompted", 3), ("unprompted", 1)):
        subprocess.run(
            [
                sys.executable, "14_merge_last_noise_delta_direction.py",
                "--family", family, "--timestamp-shard-count", str(count),
            ],
            check=True,
        )
    subprocess.run([sys.executable, "15_eval_last_noise_delta_direction_lds.py"], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=("subset-unlearning", "delta-direction", "all"),
        default="all",
    )
    args = parser.parse_args()
    if len(CUDA_IDS) < 4:
        raise ValueError("this launcher requires four configured GPUs")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.experiment in ("subset-unlearning", "all"):
        run_subset_unlearning()
    if args.experiment in ("delta-direction", "all"):
        run_delta_direction()
    print(f"[done] checkpoint counterfactual experiment={args.experiment}", flush=True)


if __name__ == "__main__":
    main()
