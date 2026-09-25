import subprocess
import sys
from exp_config import DAS_LAMBDAS

traj_methods = [
    "traj_ref_raw",
    "traj_next_raw",
    "traj_ref_ema",
    "traj_next_ema",
    "traj_projected_first_raw_linear",
    "traj_projected_first_raw_timestamp_sum_squared",
    "traj_projected_first_raw_termwise_squared",
    "traj_projected_second_raw_linear",
    "traj_projected_second_raw_timestamp_sum_squared",
    "traj_projected_second_raw_termwise_squared",
    "traj_projected_backward_first_raw_linear",
    "traj_projected_backward_first_raw_timestamp_sum_squared",
    "traj_projected_backward_first_raw_termwise_squared",
]

das_methods = [
    "das_ema",
    "das_raw",
]

metrics = [
    "simple_loss_ema",
    "simple_loss_raw",
    "traj_ref_ema",
    "traj_ref_raw",
    "endpoint_deviation_ema",
    "endpoint_deviation_raw",
    "trajectory_state_mse_ema",
    "trajectory_state_mse_raw",
]

def run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

for metric in metrics:
    for method in traj_methods:
        run([
            sys.executable,
            "06_lds_eval.py",
            "--method", method,
            "--metric", metric,
        ])

    for method in das_methods:
        for lam in DAS_LAMBDAS:
            run([
                sys.executable,
                "06_lds_eval.py",
                "--method", method,
                "--metric", metric,
                "--lambda", str(lam),
            ])

print("[done] all LDS evaluations complete")
