"""Merge four DAS term shards, divide by 1000, and save q00-q09."""

import json

import numpy as np

from staged_lds_config import *


def tag(value):
    return str(float(value)).replace(".", "p")


def main():
    sums = {float(lam): np.zeros((len(STAGED_QUERY_IDS), N_TRAIN), dtype=np.float64) for lam in DAS_LAMBDAS}
    term_count = 0
    for shard in range(4):
        root = STAGED_ATTR_DIR / "_das_shards" / f"shard_{shard:02d}_of_04"
        with open(root / "done.json") as handle:
            term_count += int(json.load(handle)["term_count"])
        with np.load(root / "scores.npz") as payload:
            for lam in DAS_LAMBDAS:
                sums[float(lam)] += payload[f"lambda_{tag(lam)}"]
    expected = len(DAS_TIMESTEPS) * int(DAS_NUM_MC)
    if term_count != expected:
        raise ValueError(f"DAS terms {term_count} != {expected}")
    for position, qid in enumerate(STAGED_QUERY_IDS):
        for lam, values in sums.items():
            out = STAGED_ATTR_DIR / STAGED_DAS_METHOD / f"q{qid:02d}" / f"lambda_{tag(lam)}"
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "scores.npy", values[position] / expected)
            with open(out / "info.json", "w") as handle:
                json.dump(
                    {
                        "query_id": qid, "lambda": lam, "parameter_source": "final_ema",
                        "train_pool": "all_50000", "terms": expected,
                        "timesteps": len(DAS_TIMESTEPS), "mc": DAS_NUM_MC,
                        "train_gradient_mc": DAS_TRAIN_GRAD_MC,
                    },
                    handle, indent=2,
                )
    print(f"[done] {STAGED_DAS_METHOD} q00-q09 terms={expected}", flush=True)


if __name__ == "__main__":
    main()
