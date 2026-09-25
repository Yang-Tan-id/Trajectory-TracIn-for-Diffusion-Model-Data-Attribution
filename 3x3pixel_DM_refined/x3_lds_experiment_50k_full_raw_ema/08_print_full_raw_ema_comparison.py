import json
from exp_config import LDS_DIR, DAS_LAMBDAS

traj_methods = [
    "traj_ref_raw",
    "traj_next_raw",
    "traj_ref_ema",
    "traj_next_ema",
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

def load(p):
    with open(p) as f:
        return json.load(f)

def tag(lam):
    return str(float(lam)).replace(".", "p")

for metric in metrics:
    traj = {}
    for m in traj_methods:
        p = LDS_DIR / f"{m}_{metric}.json"
        if p.exists():
            traj[m] = load(p)

    best = {}
    for m in das_methods:
        candidates = []
        for lam in DAS_LAMBDAS:
            p = LDS_DIR / f"{m}_{metric}_lambda_{tag(lam)}.json"
            if p.exists():
                d = load(p)
                candidates.append((float(d["mean"]), float(lam), d))
        if candidates:
            best[m] = max(candidates, key=lambda x: x[0])

    if not traj or not best:
        continue

    print("\n" + "=" * 132)
    print(
        f"METRIC={metric} | "
        f"DAS EMA best λ={best.get('das_ema',(float('nan'),float('nan'),None))[1]:g} "
        f"| DAS RAW best λ={best.get('das_raw',(float('nan'),float('nan'),None))[1]:g}"
    )
    print("=" * 132)
    print(
        f"{'Query':<8}"
        f"{'Ref raw':>14}"
        f"{'Next raw':>14}"
        f"{'Ref EMA':>14}"
        f"{'Next EMA':>14}"
        f"{'DAS EMA':>14}"
        f"{'DAS raw':>14}"
    )
    print("-" * 132)

    tq = {
        m: {x["query_id"]: x["spearman"] for x in d["queries"]}
        for m, d in traj.items()
    }
    dq = {
        m: {x["query_id"]: x["spearman"] for x in tup[2]["queries"]}
        for m, tup in best.items()
    }

    for q in range(16):
        def v(d, m):
            return d.get(m, {}).get(q, float("nan"))

        print(
            f"q{q:02d}{'':<5}"
            f"{v(tq,'traj_ref_raw'):14.6f}"
            f"{v(tq,'traj_next_raw'):14.6f}"
            f"{v(tq,'traj_ref_ema'):14.6f}"
            f"{v(tq,'traj_next_ema'):14.6f}"
            f"{v(dq,'das_ema'):14.6f}"
            f"{v(dq,'das_raw'):14.6f}"
        )

    print("-" * 132)
    def mean(m):
        return float(traj[m]["mean"]) if m in traj else float("nan")
    def dmean(m):
        return float(best[m][0]) if m in best else float("nan")

    print(
        f"{'MEAN':<8}"
        f"{mean('traj_ref_raw'):14.6f}"
        f"{mean('traj_next_raw'):14.6f}"
        f"{mean('traj_ref_ema'):14.6f}"
        f"{mean('traj_next_ema'):14.6f}"
        f"{dmean('das_ema'):14.6f}"
        f"{dmean('das_raw'):14.6f}"
    )
