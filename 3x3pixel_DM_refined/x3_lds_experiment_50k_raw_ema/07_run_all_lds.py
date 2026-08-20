import subprocess
import sys

TRAJ_METHODS = [
    "traj_ref_raw",
    "traj_next_raw",
    "traj_ref_ema",
    "traj_next_ema",
]
METRICS = ["simple_loss", "traj_ref"]

from exp_config import DAS_LAMBDAS

def run(cmd):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

for metric in METRICS:
    for method in TRAJ_METHODS:
        run([
            sys.executable,
            "06_lds_eval.py",
            "--method", method,
            "--metric", metric,
        ])

    for lam in DAS_LAMBDAS:
        run([
            sys.executable,
            "06_lds_eval.py",
            "--method", "das",
            "--metric", metric,
            "--lambda", str(lam),
        ])

print("\n[done] all LDS evaluations complete")
