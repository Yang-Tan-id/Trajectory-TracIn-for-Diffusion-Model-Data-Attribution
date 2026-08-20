import csv
import json
import sys
from pathlib import Path

import numpy as np

from exp_config import *
from dataset_generator import generate_dataset, save_dataset


def ensure_dirs():
    for p in (ROOT, DATA_DIR, MASK_DIR, MODEL_DIR, QUERY_DIR, ATTR_DIR, LDS_DIR, LOG_DIR):
        p.mkdir(parents=True, exist_ok=True)


def make_base_csv():
    if BASE_CSV.exists():
        print(f"[data] exists: {BASE_CSV}")
        return
    print(f"[data] generating seed={DATA_SEED}, N={N_TRAIN}")
    ds = generate_dataset(num_samples=N_TRAIN, seed=DATA_SEED)
    made = Path(save_dataset(ds, output_dir=str(DATA_DIR), seed=DATA_SEED, num_samples=N_TRAIN))
    if made.resolve() != BASE_CSV.resolve():
        raise RuntimeError(f"Unexpected dataset path: {made} != {BASE_CSV}")


def save_mask(seed, subset_id, idx):
    out = MASK_DIR / f"seed_{seed:02d}" / f"subset_{subset_id:02d}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.asarray(idx, dtype=np.int64))
    return out


def make_masks():
    manifest = []
    for seed in LDS_SEEDS:
        rng = np.random.default_rng(seed)
        for subset_id in range(SUBSETS_PER_SEED):
            # independent LDS-style random 25% subset
            idx = np.sort(rng.choice(N_TRAIN, size=SUBSET_SIZE, replace=False))
            path = save_mask(seed, subset_id, idx)
            manifest.append({
                "lds_seed": int(seed),
                "subset_id": int(subset_id),
                "mask_path": str(path),
                "n": int(len(idx)),
                "fraction": float(len(idx) / N_TRAIN),
            })

    with open(MASK_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    membership = np.zeros((len(manifest), N_TRAIN), dtype=np.uint8)
    for r, rec in enumerate(manifest):
        membership[r, np.load(rec["mask_path"])] = 1
    np.save(MASK_DIR / "membership.npy", membership)
    print(f"[masks] {len(manifest)} masks x {N_TRAIN}; each n={SUBSET_SIZE}")


def make_train_jobs():
    jobs = []
    # Two full models.
    for family in FAMILIES:
        jobs.append({
            "kind": "base",
            "family": family,
            "seed": BASE_MODEL_SEED,
            "mask_path": None,
            "lds_seed": None,
            "subset_id": None,
        })

    # 192 mask slots. Each slot can train one or both families.
    with open(MASK_DIR / "manifest.json") as f:
        masks = json.load(f)
    for rec in masks:
        jobs.append({
            "kind": "subset_pair",
            "families": list(SUBSET_TRAIN_FAMILIES),
            "seed": TRAIN_SEED,
            "mask_path": rec["mask_path"],
            "lds_seed": rec["lds_seed"],
            "subset_id": rec["subset_id"],
        })

    with open(ROOT / "train_jobs.json", "w") as f:
        json.dump(jobs, f, indent=2)

    actual_models = 2 + len(masks) * len(SUBSET_TRAIN_FAMILIES)
    print(f"[jobs] GPU jobs={len(jobs)} = 2 base + {len(masks)} subset slots")
    print(f"[jobs] actual model trainings={actual_models}")


def main():
    ensure_dirs()
    make_base_csv()
    make_masks()
    make_train_jobs()
    print("[done] prepare complete")


if __name__ == "__main__":
    main()
