"""Verify prerequisites for Adam/clipping-aware raw SOURCE-DAS."""

import json

import torch

from adam_clip_source_das_config import *
from adam_clip_source_das_x3 import _optimizer_state_by_parameter_name
from attribution_one_query import build_model
from source_das_config import SOURCE_DAS_SIMPLE_INFLUENCE_ROOT


def main():
    if not (SOURCE_DAS_SIMPLE_INFLUENCE_ROOT / "src" / "source.py").is_file():
        raise FileNotFoundError(SOURCE_DAS_SIMPLE_INFLUENCE_ROOT)
    with open(QUERY_DIR / "manifest.json") as handle:
        records = json.load(handle)
    if len(records) != 100:
        raise ValueError(f"expected 100 queries, found {len(records)}")
    for family in FAMILIES:
        for epoch in ADAM_CLIP_SOURCE_CHECKPOINT_EPOCHS:
            path = MODEL_DIR / "base" / family / f"epoch_{epoch:04d}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if "optimizer_state" not in payload:
                raise ValueError(f"missing optimizer_state: {path}")
            if int(payload["global_step"]) <= 0:
                raise ValueError(f"invalid global_step: {path}")
            states = payload["optimizer_state"]["state"].values()
            if not states or any("exp_avg_sq" not in item for item in states):
                raise ValueError(f"missing Adam exp_avg_sq: {path}")
            model, _, _ = build_model(path, "raw", torch.device("cpu"))
            mapped = _optimizer_state_by_parameter_name(payload, model)
            if set(mapped) != set(dict(model.named_parameters())):
                raise ValueError(f"optimizer parameter mapping failed: {path}")
    print(f"simple-influence = {SOURCE_DAS_SIMPLE_INFLUENCE_ROOT}")
    print(f"methods = {ADAM_CLIP_SOURCE_METHODS}")
    print(f"checkpoints = {ADAM_CLIP_SOURCE_CHECKPOINT_EPOCHS}")
    print(f"iterations/segment = {adam_clip_source_iters_per_segment()}")
    print(f"LR sums/segment = {adam_clip_source_lr_sums_per_segment()}")
    print(f"train MC = {ADAM_CLIP_SOURCE_TRAIN_MC}")
    print(f"train batch = {ADAM_CLIP_SOURCE_TRAIN_BATCH_SIZE}")
    print(f"clip norm = {ADAM_CLIP_SOURCE_CLIP_NORM}")
    print(f"final/query parameter source = {ADAM_CLIP_SOURCE_FINAL_PARAM_SOURCE}")
    print("query trajectory source = cached EMA DDIM trajectory_xt.npy")
    print("[OK] Adam/clipping-aware raw SOURCE-DAS prerequisites verified")


if __name__ == "__main__":
    main()
