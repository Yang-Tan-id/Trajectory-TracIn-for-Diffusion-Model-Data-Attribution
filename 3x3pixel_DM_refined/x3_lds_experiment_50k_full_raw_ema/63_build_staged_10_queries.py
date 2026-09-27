"""Regenerate q00-q09 with the staged base model's final EMA weights."""

import json
from pathlib import Path

import numpy as np
import torch

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from staged_lds_config import *


def main():
    with open(QUERY_DIR / "manifest.json") as handle:
        original = {int(r["query_id"]): r for r in json.load(handle)}
    records = [dict(original[qid]) for qid in STAGED_QUERY_IDS]
    if any(r["family"] != STAGED_FAMILY for r in records):
        raise ValueError("staged q00-q09 must all be prompted")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    checkpoint = torch.load(staged_base_checkpoint(STAGED_EPOCHS), map_location=device, weights_only=False)
    model = base.CondEpsModel(3, len(dataset.vocab), BASE_CH, TIME_DIM).to(device)
    model.load_state_dict(checkpoint["ema_model_state"], strict=True)
    model.eval()
    sched = base.make_linear_schedule(T, device=device)
    save_steps = np.linspace(0, DDIM_STEPS - 1, TRAJ_SNAPSHOTS, dtype=np.int64).tolist()
    all_t = np.linspace(T - 1, 0, DDIM_STEPS, dtype=np.int64)
    t_seq = np.asarray([int(all_t[k]) for k in save_steps], dtype=np.int64)
    STAGED_QUERY_DIR.mkdir(parents=True, exist_ok=True)
    output_records = []
    for source in records:
        qid = int(source["query_id"])
        cond = torch.zeros((1, len(dataset.vocab)), device=device)
        for label in source["labels"]:
            cond[0, dataset.vocab[label]] = 1.0
        trajectory = base.ddim_sample(
            model=model, sched=sched, cond=cond, shape=(1, 3, 3, 3),
            seed=int(source["initial_seed"]), steps=DDIM_STEPS, eta=0.0,
            device=str(device), save_steps=save_steps,
        )
        out = STAGED_QUERY_DIR / f"q{qid:02d}"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "trajectory_xt.npy", np.stack([x.detach().cpu().numpy() for x in trajectory]))
        np.save(out / "trajectory_t.npy", t_seq)
        np.save(out / "final_state.npy", trajectory[-1].detach().cpu().numpy())
        record = {
            "query_id": qid, "family": STAGED_FAMILY,
            "initial_seed": int(source["initial_seed"]),
            "prompt_rep": source.get("prompt_rep"), "labels": source["labels"],
            "dir": str(out), "parameter_source": "staged_final_ema",
        }
        with open(out / "query.json", "w") as handle:
            json.dump(record, handle, indent=2)
        output_records.append(record)
        print(f"[saved] staged q{qid:02d}", flush=True)
    with open(STAGED_QUERY_DIR / "manifest.json", "w") as handle:
        json.dump(output_records, handle, indent=2)


if __name__ == "__main__":
    main()
