"""Configuration for timestamp-aligned vector trajectory DAS Bundle."""

from exp_config import *


TAVD_METHOD = "timestamp_aligned_vector_trajectory_das_ema_mc10_projected4096_10t_bundle"
TAVD_ROOT = ATTR_DIR / f"_{TAVD_METHOD}_timestamp_tasks"
TAVD_POSITIONS = tuple(int(value) for value in __import__("numpy").linspace(0, TRAJ_SNAPSHOTS - 1, 10).round())
TAVD_LAMBDAS = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0)


def lambda_tag(value):
    return str(float(value)).replace(".", "p")

