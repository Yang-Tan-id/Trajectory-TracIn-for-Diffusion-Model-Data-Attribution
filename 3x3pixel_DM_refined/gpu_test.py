import os
import sys
import socket
import subprocess

def safe_run(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "not available"

def get_env(name):
    return os.environ.get(name, "not set")

# Optional: allow GPU id from command line
# Example:
#   python print_cluster.py 0
#   python print_cluster.py 3
if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

print("=== Basic info ===")
print("Hostname:", socket.gethostname())
print("Fully qualified hostname:", safe_run(["hostname", "-f"]))
print("User:", get_env("USER"))

print("\n=== Cluster / scheduler environment ===")
print("SLURM_CLUSTER_NAME:", get_env("SLURM_CLUSTER_NAME"))
print("SLURM_JOB_ID:", get_env("SLURM_JOB_ID"))
print("SLURM_JOB_NODELIST:", get_env("SLURM_JOB_NODELIST"))
print("SLURM_JOB_PARTITION:", get_env("SLURM_JOB_PARTITION"))
print("PBS_JOBID:", get_env("PBS_JOBID"))
print("PBS_QUEUE:", get_env("PBS_QUEUE"))
print("LSB_JOBID:", get_env("LSB_JOBID"))

print("\n=== CUDA environment ===")
print("CUDA_VISIBLE_DEVICES:", get_env("CUDA_VISIBLE_DEVICES"))

print("\n=== nvidia-smi ===")
print("GPU list:", safe_run(["nvidia-smi", "-L"]))

print("\n=== PyTorch CUDA info ===")
try:
    import torch

    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())

    if torch.cuda.is_available():
        current = torch.cuda.current_device()
        for i in range(torch.cuda.device_count()):
            print(f"Visible GPU {i}: {torch.cuda.get_device_name(i)}")
        print("Current GPU index:", current)
        print("Current GPU name:", torch.cuda.get_device_name(current))
    else:
        print("No GPU visible to PyTorch.")
except Exception as e:
    print("PyTorch info unavailable:", e)