import json
from pathlib import Path

from exp_config import LDS_DIR, DAS_LAMBDAS

TRAJ_METHODS = [
    "traj_ref_raw",
    "traj_next_raw",
    "traj_ref_ema",
    "traj_next_ema",
]

LABELS = {
    "traj_ref_raw": "Ref raw",
    "traj_next_raw": "Next raw",
    "traj_ref_ema": "Ref EMA",
    "traj_next_ema": "Next EMA",
}

def load(path):
    with open(path) as f:
        return json.load(f)

for metric in ["simple_loss", "traj_ref"]:
    traj = {}
    for method in TRAJ_METHODS:
        d = load(LDS_DIR / f"{method}_{metric}.json")
        traj[method] = d

    das_candidates = []
    for lam in DAS_LAMBDAS:
        tag = str(float(lam)).replace(".", "p")
        p = LDS_DIR / f"das_{metric}_lambda_{tag}.json"
        if p.exists():
            d = load(p)
            das_candidates.append((float(d["mean"]), float(lam), d))

    if not das_candidates:
        print(f"\nNo DAS LDS results for metric={metric}")
        continue

    das_mean, best_lam, das = max(das_candidates, key=lambda x: x[0])

    perq = {
        m: {r["query_id"]: r["spearman"] for r in d["queries"]}
        for m, d in traj.items()
    }
    das_q = {r["query_id"]: r["spearman"] for r in das["queries"]}

    print("\n" + "=" * 108)
    print(f"METRIC = {metric} | BEST DAS lambda = {best_lam:g}")
    print("=" * 108)
    print(
        f"{'Query':<8}"
        f"{'Ref raw':>14}"
        f"{'Next raw':>14}"
        f"{'Ref EMA':>14}"
        f"{'Next EMA':>14}"
        f"{'Best DAS':>14}"
    )
    print("-" * 108)

    for q in range(len(das["queries"])):
        print(
            f"q{q:02d}{'':<5}"
            f"{perq['traj_ref_raw'][q]:14.6f}"
            f"{perq['traj_next_raw'][q]:14.6f}"
            f"{perq['traj_ref_ema'][q]:14.6f}"
            f"{perq['traj_next_ema'][q]:14.6f}"
            f"{das_q[q]:14.6f}"
        )

    print("-" * 108)
    print(
        f"{'MEAN':<8}"
        f"{traj['traj_ref_raw']['mean']:14.6f}"
        f"{traj['traj_next_raw']['mean']:14.6f}"
        f"{traj['traj_ref_ema']['mean']:14.6f}"
        f"{traj['traj_next_ema']['mean']:14.6f}"
        f"{das_mean:14.6f}"
    )
