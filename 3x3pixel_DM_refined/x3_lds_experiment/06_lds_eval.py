import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from exp_config import *


def load_attr(method, qid, lam=None):
    if method in ("traj_ref", "traj_next"):
        return np.load(ATTR_DIR / method / f"q{qid:02d}" / "scores.npy")
    if method == "das":
        if lam is None:
            raise ValueError("--lambda is required for DAS")
        tag = str(float(lam)).replace(".", "p")
        return np.load(ATTR_DIR / "das" / f"q{qid:02d}" / f"lambda_{tag}" / "scores.npy")
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["simple_loss", "traj_ref"], default="traj_ref")
    ap.add_argument("--method", choices=["traj_ref", "traj_next", "das"], required=True)
    ap.add_argument("--lambda", dest="lam", type=float, default=None)
    a = ap.parse_args()

    observed = np.load(LDS_DIR / f"observed_{a.metric}.npy")  # [Q,S]
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)  # [S,N]

    rows = []
    for qid in range(observed.shape[0]):
        attr = load_attr(a.method, qid, a.lam)
        pred = membership @ attr
        rho = float(spearmanr(pred, observed[qid]).statistic)
        rows.append({"query_id": qid, "spearman": rho})
        print(f"q{qid:02d}: LDS Spearman={rho*100:.3f}%")

    mean = float(np.nanmean([r["spearman"] for r in rows]))
    print(f"\nMEAN LDS = {mean*100:.3f}%")

    name = f"{a.method}_{a.metric}"
    if a.method == "das":
        name += f"_lambda_{str(a.lam).replace('.','p')}"
    out = LDS_DIR / f"{name}.json"
    with open(out, "w") as f:
        json.dump({"method": a.method, "metric": a.metric, "lambda": a.lam, "mean": mean, "queries": rows}, f, indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
