import argparse
import json
import numpy as np
from scipy.stats import spearmanr

from exp_config import *


TRAJ_METHODS = (
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
)

DAS_METHODS = (
    "das_ema",
    "das_raw",
)

METRICS = (
    "simple_loss_ema",
    "simple_loss_raw",
    "traj_ref_ema",
    "traj_ref_raw",
    "endpoint_deviation_ema",
    "endpoint_deviation_raw",
    "trajectory_state_mse_ema",
    "trajectory_state_mse_raw",
)


def lambda_tag(lam):
    return str(float(lam)).replace(".", "p")


def load_observed(metric):
    path = LDS_DIR / f"observed_{metric}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing observed metric: {path}")
    return np.load(path)


def load_attr(method, qid, lam=None):
    if method in TRAJ_METHODS:
        path = ATTR_DIR / method / f"q{qid:02d}" / "scores.npy"
    elif method in DAS_METHODS:
        if lam is None:
            raise ValueError("--lambda required for DAS")
        path = (
            ATTR_DIR
            / method
            / f"q{qid:02d}"
            / f"lambda_{lambda_tag(lam)}"
            / "scores.npy"
        )
    else:
        raise ValueError(method)

    if not path.exists():
        raise FileNotFoundError(f"Missing attribution: {path}")
    return np.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method",
        required=True,
        choices=list(TRAJ_METHODS + DAS_METHODS),
    )
    ap.add_argument(
        "--metric",
        required=True,
        choices=list(METRICS),
    )
    ap.add_argument("--lambda", dest="lam", type=float, default=None)
    a = ap.parse_args()

    if a.method in DAS_METHODS and a.lam is None:
        ap.error("--lambda required for DAS")

    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    observed = load_observed(a.metric).astype(np.float64)

    results = []
    for qid in range(observed.shape[0]):
        attr = load_attr(a.method, qid, a.lam).astype(np.float64).reshape(-1)
        pred = -(membership @ attr)
        rho = float(spearmanr(pred, observed[qid]).statistic)

        results.append({"query_id": qid, "spearman": rho})
        print(f"q{qid:02d}: Spearman={rho:.6f}", flush=True)

    mean_rho = float(np.nanmean([x["spearman"] for x in results]))

    if a.method in DAS_METHODS:
        out = (
            LDS_DIR
            / f"{a.method}_{a.metric}_lambda_{lambda_tag(a.lam)}.json"
        )
    else:
        out = LDS_DIR / f"{a.method}_{a.metric}.json"

    payload = {
        "method": a.method,
        "metric": a.metric,
        "lambda": float(a.lam) if a.method in DAS_METHODS else None,
        "mean": mean_rho,
        "queries": results,
    }

    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nMEAN LDS = {mean_rho:.6f}")
    print(f"SAVED = {out}")


if __name__ == "__main__":
    main()
