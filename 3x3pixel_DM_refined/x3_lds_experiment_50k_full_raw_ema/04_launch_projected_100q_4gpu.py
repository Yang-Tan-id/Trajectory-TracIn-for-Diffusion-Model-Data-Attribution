"""Launch restartable family-bank attribution without retraining models."""

import json
import subprocess
import sys

from exp_config import CUDA_IDS, QUERY_DIR


def run_all():
    if len(CUDA_IDS) < 4:
        raise ValueError("the bank launcher requires four CUDA_IDS")
    commands = []
    for family, gpu in zip(("prompted", "unprompted"), CUDA_IDS[:2]):
        commands.append([sys.executable, "run_projected_traj_bank.py", "--family", family, "--gpu", str(gpu)])
    for family, gpu in zip(("prompted", "unprompted"), CUDA_IDS[2:4]):
        commands.append([sys.executable, "run_das_bank.py", "--family", family, "--gpu", str(gpu)])
    processes = [subprocess.Popen(command) for command in commands]
    for process in processes:
        code = process.wait()
        if code != 0:
            for other in processes:
                if other.poll() is None:
                    other.terminate()
            raise SystemExit(code)


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
