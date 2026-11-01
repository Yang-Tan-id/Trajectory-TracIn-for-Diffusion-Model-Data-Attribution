#!/usr/bin/env python3
"""Merge 100t x MC4 LDS simple loss and update core LDS summaries."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from diffusion_retrac_config import RETRAC_METHODS
from endpoint_tracin_adamw_mc10_config import (
    ETA_QUERY_IDS,
    eta_method,
    eta_raw_gradient_method,
)
from exp_config import ATTR_DIR, LDS_DIR, MASK_DIR, QUERY_DIR
from tracin_das_norm_adam_config import tdna_methods


METRICS = ("simple_loss_ema", "simple_loss_raw")
TIMESTAMPS = tuple(
    int(value) for value in np.linspace(0, 999, 100, dtype=np.int64)
)
MC = 4


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def evaluate_method(method, membership, observed, query_ids):
    scores = np.stack(
        [
            np.load(ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy")
            for query_id in query_ids
        ]
    ).astype(np.float64)
    predictions = membership @ scores.T
    result = {}
    for metric in METRICS:
        positive = np.asarray(
            [
                spearmanr(
                    predictions[:, position], observed[metric][query_id]
                ).statistic
                for position, query_id in enumerate(query_ids)
            ],
            dtype=np.float64,
        )
        result[metric] = {
            "negative": {
                "mean": float(np.nanmean(-positive)),
                "std": float(np.nanstd(-positive)),
                "per_query": (-positive).tolist(),
            },
            "positive": {
                "mean": float(np.nanmean(positive)),
                "std": float(np.nanstd(positive)),
                "per_query": positive.tolist(),
            },
        }
    return result


def backup_and_write_observed(arrays):
    for metric, values in arrays.items():
        canonical = LDS_DIR / f"observed_{metric}.npy"
        backup = LDS_DIR / f"observed_{metric}_10t_mc4.npy"
        if canonical.is_file() and not backup.is_file():
            np.save(backup, np.load(canonical))
            print(f"[backup] {backup}", flush=True)
        np.save(LDS_DIR / f"observed_{metric}_100t_mc4.npy", values)
        np.save(canonical, values)
        print(f"[saved] {canonical} shape={values.shape}", flush=True)


def update_query_shards(arrays):
    shard_dir = LDS_DIR / "observed_query_shards"
    for query_id in range(arrays[METRICS[0]].shape[0]):
        path = shard_dir / f"q{query_id:02d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            values = {key: np.asarray(payload[key]) for key in payload.files}
        for metric in METRICS:
            values[metric] = arrays[metric][query_id]
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, **values)
        temporary.replace(path)


def update_summary(path, entries, membership, observed):
    if not path.is_file():
        raise FileNotFoundError(path)
    backup = path.with_name(path.stem + "_before_simple_loss_100t_mc4.json")
    if not backup.is_file():
        backup.write_text(path.read_text())
    data = json.loads(path.read_text())
    for target, method, query_ids in entries(data):
        values = evaluate_method(method, membership, observed, query_ids)
        target.setdefault("targets", {}).update(values)
        print(
            f"[LDS 100x4] {method} "
            f"ema={values['simple_loss_ema']['negative']['mean']:+.6f}/"
            f"{values['simple_loss_ema']['positive']['mean']:+.6f}",
            flush=True,
        )
    data["simple_loss_observed_bank"] = {
        "timestamps": list(TIMESTAMPS),
        "mc": MC,
        "samples": len(TIMESTAMPS) * MC,
    }
    atomic_json(path, data)


def main():
    queries = json.loads((QUERY_DIR / "manifest.json").read_text())
    masks = json.loads((MASK_DIR / "manifest.json").read_text())
    shard_dir = LDS_DIR / "observed_simple_loss_100t_mc4_mask_shards"
    arrays = {
        metric: np.empty((len(queries), len(masks)), dtype=np.float64)
        for metric in METRICS
    }
    for mask_index in range(len(masks)):
        path = shard_dir / f"mask_{mask_index:03d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            if int(payload["mask_index"]) != mask_index:
                raise ValueError(f"mask mismatch: {path}")
            if tuple(map(int, payload["timestamps"])) != TIMESTAMPS:
                raise ValueError(f"timestamp mismatch: {path}")
            if int(payload["mc"]) != MC:
                raise ValueError(f"MC mismatch: {path}")
            for metric in METRICS:
                arrays[metric][:, mask_index] = payload[metric]

    backup_and_write_observed(arrays)
    update_query_shards(arrays)
    metadata = {
        "timestamps": list(TIMESTAMPS),
        "mc": MC,
        "samples_per_model_query": len(TIMESTAMPS) * MC,
        "query_count": len(queries),
        "mask_count": len(masks),
    }
    atomic_json(LDS_DIR / "observed_simple_loss_100t_mc4.json", metadata)
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)

    pint_path = LDS_DIR / "tracin_das_norm4_gradient_and_adamw_full_q00_q99.json"

    def pint_entries(data):
        methods = tdna_methods()
        query_ids = [int(value) for value in data["query_ids"]]
        for transform in ("gradient", "adamw_full"):
            for variant in ("raw", "query_train_l2"):
                target = data["variants"][transform][variant]["timestamp_sum_squared"]
                yield target, methods[transform][variant]["timestamp_sum_squared"], query_ids

    update_summary(pint_path, pint_entries, membership, arrays)

    endpoint_specs = (
        (
            LDS_DIR / "endpoint_tracin_raw_gradient_100t_querymc10_trainmc10_norm4_q00_q99.json",
            "raw_gradient",
        ),
        (
            LDS_DIR / "endpoint_tracin_adamw_full_100t_mc10_norm4_q00_q99.json",
            "adamw_full",
        ),
    )
    for endpoint_path, transform in endpoint_specs:

        def endpoint_entries(data, transform=transform):
            query_ids = [int(value) for value in data["query_ids"]]
            for variant in ("raw", "query_train_l2"):
                target = data["variants"][variant]["linear"]
                method = (
                    eta_method(variant, "linear")
                    if transform == "adamw_full"
                    else eta_raw_gradient_method(
                        variant, "linear", query_mc=10, train_mc=10
                    )
                )
                yield target, method, query_ids

        update_summary(endpoint_path, endpoint_entries, membership, arrays)

    retrac_path = LDS_DIR / "diffusion_tracin_vs_retrac_adamw_full_replayed_100q.json"

    def retrac_entries(data):
        query_ids = [int(value) for value in data["query_ids"]]
        for key in ("diffusion_retrac", "diffusion_retrac_adamw_full"):
            target = data["methods"][key]
            yield target, RETRAC_METHODS[key], query_ids

    update_summary(retrac_path, retrac_entries, membership, arrays)
    print("[done] LDS simple loss is now 100 timestamps x MC4", flush=True)


if __name__ == "__main__":
    main()
