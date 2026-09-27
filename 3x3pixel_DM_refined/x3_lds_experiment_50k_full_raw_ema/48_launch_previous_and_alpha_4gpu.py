"""Run previous-checkpoint Traj, then save/evaluate signed alpha blends."""

import subprocess
import sys


def main():
    subprocess.run(
        [sys.executable, "-u", "04_launch_projected_backward_100q_4gpu.py"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-u", "47_eval_next_previous_alpha_sweep.py"],
        check=True,
    )
    print("[done] previous Traj scores and next/previous alpha sweep", flush=True)


if __name__ == "__main__":
    main()
