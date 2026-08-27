from __future__ import annotations

"""
Build a counterfactual CIFAR10 model by removing top-k attributed training items.

The script reads one or more attribution result folders, combines their scores,
maps the selected attribution indices back to original CIFAR batch rows, and
re-trains DM__training_CIFAR10_pixel with those rows excluded.

Example
-------
CUDA_VISIBLE_DEVICES=2 python3 DM_counterfactual_retrain_from_attribution.py \
  --result-dirs \
    attribution_results/traj_tracein/cifar2_traj_attr_cifar10_horse_automobile_from_sample_range_1_2000 \
    attribution_results/traj_tracein/cifar2_traj_attr_cifar10_horse_automobile_from_sample_range_2001_4000 \
    attribution_results/traj_tracein/cifar2_traj_attr_cifar10_horse_automobile_from_sample_range_4001_6000 \
    attribution_results/traj_tracein/cifar2_traj_attr_cifar10_horse_automobile_from_sample_range_6001_8000 \
    attribution_results/traj_tracein/cifar2_traj_attr_cifar10_horse_automobile_from_sample_range_8001_10000 \
  --base-checkpoint models/cifar10_checkpoints_horse_automobile/seed_0_epoch_0200.ckpt \
  --topk 5000 \
  --dataset-tag cifar \
  --model-tag horse_automobile \
  --query horse \
  --seed 0 \
  --score-tag traj_tracin \
  --prefer-device gpu
  
CUDA_VISIBLE_DEVICES=3 python3 DM_counterfactual_retrain_from_attribution.py \
  --result-dirs \
    attribution_results/endpoint_das/cifar2_endpoint_das_horse_automobile_from_sample_range_1_10000 \
  --base-checkpoint models/cifar10_checkpoints_horse_automobile/seed_0_epoch_0200.ckpt \
  --topk 5000 \
  --dataset-tag cifar \
  --model-tag horse_automobile \
  --query horse \
  --seed 0 \
  --score-tag endpoint_das \
  --prefer-device gpu  
"""

import argparse
import json
import os
import pickle
import re
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from DM__training_CIFAR10_pixel import TrainConfig, train


def _decode_if_bytes(x):
    return x.decode("utf-8") if isinstance(x, bytes) else x


def load_cifar_batch(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    return np.asarray(d[b"data"], dtype=np.uint8), np.asarray(d[b"labels"], dtype=np.int32)


def load_cifar_label_names(data_root: str) -> List[str]:
    with open(os.path.join(data_root, "batches.meta"), "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    return [_decode_if_bytes(x) for x in meta[b"label_names"]]


def sanitize_tag(text: Optional[str], default: str = "unknown") -> str:
    if text is None or str(text).strip() == "":
        return default
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_base_config(base_checkpoint: Optional[str]) -> TrainConfig:
    if base_checkpoint is None:
        return TrainConfig()
    with open(base_checkpoint, "rb") as f:
        payload = pickle.load(f)
    cfg_dict = dict(payload.get("config", {}))
    if not cfg_dict:
        raise ValueError(f"No training config found in checkpoint: {base_checkpoint}")
    valid_keys = set(TrainConfig.__dataclass_fields__.keys())
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    return TrainConfig(**cfg_dict)


def load_scores_from_result_dir(result_dir: str) -> Tuple[np.ndarray, np.ndarray, str]:
    result_dir = os.path.abspath(result_dir)
    scores_path = os.path.join(result_dir, "scores.npy")
    if not os.path.isfile(scores_path):
        raise FileNotFoundError(f"Missing scores.npy in {result_dir}")
    scores = np.asarray(np.load(scores_path), dtype=np.float64).reshape(-1)

    score_indices_npy = os.path.join(result_dir, "score_indices.npy")
    score_indices_json = os.path.join(result_dir, "score_indices.json")
    traj_json = os.path.join(result_dir, "traj_attr_result.json")

    if os.path.isfile(score_indices_npy):
        indices = np.asarray(np.load(score_indices_npy), dtype=np.int64).reshape(-1)
        source_kind = "score_indices.npy"
    elif os.path.isfile(score_indices_json):
        payload = load_json(score_indices_json)
        indices = np.asarray(payload["picked_indices"], dtype=np.int64).reshape(-1)
        source_kind = "score_indices.json"
    elif os.path.isfile(traj_json):
        payload = load_json(traj_json)
        indices = np.asarray(payload["score_indices"], dtype=np.int64).reshape(-1)
        source_kind = "traj_attr_result.json"
    else:
        # Last-resort support for result_topk.json only. This cannot recover
        # non-top-k scored items, but it is still useful for small manual runs.
        topk_json = os.path.join(result_dir, "result_topk.json")
        if not os.path.isfile(topk_json):
            raise FileNotFoundError(f"Missing score_indices file in {result_dir}")
        top_payload = load_json(topk_json)
        top_items = top_payload.get("top") or top_payload.get("topk") or []
        indices = np.asarray([item["idx"] for item in top_items], dtype=np.int64)
        scores = np.asarray([item["score"] for item in top_items], dtype=np.float64)
        source_kind = "result_topk.json"

    if len(indices) != len(scores):
        raise ValueError(
            f"Length mismatch in {result_dir}: len(indices)={len(indices)} len(scores)={len(scores)}"
        )
    return indices, scores, source_kind


def combine_attribution_scores(
    result_dirs: Sequence[str],
    duplicate_policy: str = "max",
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    combined: Dict[int, float] = {}
    sources = []
    for result_dir in result_dirs:
        indices, scores, source_kind = load_scores_from_result_dir(result_dir)
        sources.append(
            {
                "result_dir": os.path.abspath(result_dir),
                "source_kind": source_kind,
                "num_scores": int(len(scores)),
            }
        )
        for idx, score in zip(indices.tolist(), scores.tolist()):
            idx = int(idx)
            score = float(score)
            if idx not in combined:
                combined[idx] = score
            elif duplicate_policy == "max":
                combined[idx] = max(combined[idx], score)
            elif duplicate_policy == "sum":
                combined[idx] += score
            elif duplicate_policy == "mean":
                # Mean is handled below with a second pass for clarity.
                combined[idx] = max(combined[idx], score)
            else:
                raise ValueError(f"Unknown duplicate_policy={duplicate_policy!r}")

    if duplicate_policy == "mean":
        totals: Dict[int, float] = {}
        counts: Dict[int, int] = {}
        for result_dir in result_dirs:
            indices, scores, _ = load_scores_from_result_dir(result_dir)
            for idx, score in zip(indices.tolist(), scores.tolist()):
                idx = int(idx)
                totals[idx] = totals.get(idx, 0.0) + float(score)
                counts[idx] = counts.get(idx, 0) + 1
        combined = {idx: totals[idx] / counts[idx] for idx in totals}

    all_indices = np.asarray(list(combined.keys()), dtype=np.int64)
    all_scores = np.asarray([combined[int(i)] for i in all_indices], dtype=np.float64)
    return all_indices, all_scores, sources


def build_filtered_index_to_cifar_row_map(
    data_root: str,
    batch_names: Optional[Sequence[str]],
    class_names: Optional[Sequence[str]],
) -> List[Tuple[int, int, int]]:
    npz_path = os.path.join(data_root, "dataset.npz") if os.path.isdir(data_root) else data_root
    if os.path.isfile(npz_path) and os.path.basename(npz_path) == "dataset.npz":
        with np.load(npz_path, allow_pickle=False) as payload:
            labels = np.asarray(payload["labels"], dtype=np.int32)
            label_names = [str(x) for x in payload["label_names"].tolist()]
        keep_ids = None
        if class_names is not None:
            name_to_id = {name: i for i, name in enumerate(label_names)}
            keep_ids = set(int(name_to_id[name]) for name in class_names)
        mapping: List[Tuple[int, int, int]] = []
        for row_idx, label in enumerate(labels):
            if labels.ndim == 1:
                primary = int(label)
                keep = keep_ids is None or primary in keep_ids
            else:
                active = np.flatnonzero(label)
                primary = int(active[0]) if active.size else -1
                keep = keep_ids is None or bool(set(active.tolist()) & keep_ids)
            if keep:
                mapping.append((0, int(row_idx), primary))
        return mapping

    label_names = load_cifar_label_names(data_root)
    name_to_id = {name: i for i, name in enumerate(label_names)}
    keep_ids = None
    if class_names is not None:
        keep_ids = set(int(name_to_id[name]) for name in class_names)

    if batch_names is None:
        batch_names = [
            "data_batch_1",
            "data_batch_2",
            "data_batch_3",
            "data_batch_4",
            "data_batch_5",
        ]

    mapping: List[Tuple[int, int, int]] = []
    for batch_name in batch_names:
        if not batch_name.startswith("data_batch_"):
            continue
        batch_id = int(batch_name.split("_")[-1])
        _, labels = load_cifar_batch(os.path.join(data_root, batch_name))
        for row_idx, label in enumerate(labels.tolist()):
            label = int(label)
            if keep_ids is None or label in keep_ids:
                mapping.append((batch_id, int(row_idx), label))
    return mapping


def selected_indices_to_exclude_indices(
    selected_filtered_indices: Iterable[int],
    filtered_to_cifar_rows: Sequence[Tuple[int, int, int]],
) -> Dict[int, Tuple[int, ...]]:
    exclude: Dict[int, List[int]] = {}
    n = len(filtered_to_cifar_rows)
    for idx in selected_filtered_indices:
        idx = int(idx)
        if idx < 0 or idx >= n:
            raise IndexError(f"Attribution index {idx} is out of range for filtered dataset size {n}")
        batch_id, row_idx, _ = filtered_to_cifar_rows[idx]
        exclude.setdefault(int(batch_id), []).append(int(row_idx))
    return {batch_id: tuple(sorted(set(rows))) for batch_id, rows in sorted(exclude.items())}


def parse_class_names(text: Optional[str]):
    if text is None:
        return None
    names = tuple(tok.strip() for tok in text.split(",") if tok.strip())
    return names or None


def main():
    parser = argparse.ArgumentParser(
        description="Retrain a CIFAR10 counterfactual model after removing top-k attributed items."
    )
    parser.add_argument("--result-dirs", nargs="+", required=True, help="Attribution result directories to combine.")
    parser.add_argument("--topk", type=int, required=True, help="Number of highest-scoring items to remove.")
    parser.add_argument("--base-checkpoint", type=str, default=None, help="Checkpoint whose saved TrainConfig is reused.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--class-names", type=str, default=None, help="Comma-separated class subset override, e.g. horse,automobile")
    parser.add_argument("--dataset-tag", type=str, default="cifar")
    parser.add_argument("--model-tag", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--score-tag", type=str, default=None, help="Optional score method label, e.g. traj_tracin or das.")
    parser.add_argument("--out-root", type=str, default="./counterfactual_models")
    parser.add_argument("--duplicate-policy", choices=["max", "sum", "mean"], default="max")
    parser.add_argument("--prefer-device", type=str, default=None, choices=["auto", "cpu", "gpu"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-every-epochs", type=int, default=None)
    parser.add_argument("--keep-last-k", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Write removal metadata but do not train.")

    args = parser.parse_args()

    cfg = load_base_config(args.base_checkpoint)
    cfg.resume_from = None
    cfg.exclude_ranges = None

    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.class_names is not None:
        cfg.class_names = parse_class_names(args.class_names)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.prefer_device is not None:
        cfg.prefer_device = args.prefer_device
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.save_every_epochs is not None:
        cfg.save_every_epochs = int(args.save_every_epochs)
    if args.keep_last_k is not None:
        cfg.keep_last_k = int(args.keep_last_k)

    all_indices, all_scores, sources = combine_attribution_scores(
        args.result_dirs,
        duplicate_policy=args.duplicate_policy,
    )
    if len(all_indices) == 0:
        raise RuntimeError("No attribution scores loaded.")

    topk = min(int(args.topk), len(all_indices))
    order = np.argsort(-all_scores)[:topk]
    selected_indices = all_indices[order]
    selected_scores = all_scores[order]

    filtered_to_cifar_rows = build_filtered_index_to_cifar_row_map(
        data_root=cfg.data_root,
        batch_names=cfg.batch_names,
        class_names=cfg.class_names,
    )
    exclude_indices = selected_indices_to_exclude_indices(selected_indices, filtered_to_cifar_rows)
    cfg.exclude_indices = exclude_indices

    dataset_tag = sanitize_tag(args.dataset_tag, "cifar")
    model_tag = sanitize_tag(args.model_tag, "model")
    query_tag = sanitize_tag(args.query, "query")
    score_tag = sanitize_tag(args.score_tag, "score") if args.score_tag else None
    name_parts = [dataset_tag, model_tag, query_tag, f"seed_{cfg.seed}", f"remove_top{topk}"]
    if score_tag:
        name_parts.append(score_tag)
    run_name = "__".join(name_parts)
    out_dir = os.path.join(args.out_root, run_name)
    cfg.checkpoint_dir = out_dir

    os.makedirs(out_dir, exist_ok=True)
    removal = []
    for rank, (idx, score) in enumerate(zip(selected_indices.tolist(), selected_scores.tolist()), start=1):
        batch_id, row_idx, label = filtered_to_cifar_rows[int(idx)]
        removal.append(
            {
                "rank": int(rank),
                "attribution_idx": int(idx),
                "attribution_idx_1based": int(idx) + 1,
                "score": float(score),
                "cifar_batch_id": int(batch_id),
                "cifar_row_idx": int(row_idx),
                "label_id": int(label),
            }
        )

    metadata = {
        "run_name": run_name,
        "checkpoint_dir": os.path.abspath(out_dir),
        "base_checkpoint": None if args.base_checkpoint is None else os.path.abspath(args.base_checkpoint),
        "result_sources": sources,
        "duplicate_policy": args.duplicate_policy,
        "num_combined_scores": int(len(all_scores)),
        "topk_removed": int(topk),
        "dataset_tag": args.dataset_tag,
        "model_tag": args.model_tag,
        "query": args.query,
        "class_names": None if cfg.class_names is None else list(cfg.class_names),
        "exclude_indices": {str(k): list(v) for k, v in exclude_indices.items()},
        "removed_items": removal,
        "train_config": asdict(cfg),
    }
    save_json(os.path.join(out_dir, "counterfactual_removal.json"), metadata)
    np.save(os.path.join(out_dir, "removed_attribution_indices.npy"), selected_indices.astype(np.int64))
    np.save(os.path.join(out_dir, "removed_scores.npy"), selected_scores.astype(np.float64))

    print("=" * 88)
    print("Counterfactual retraining setup")
    print(f"output_dir          : {out_dir}")
    print(f"base_checkpoint     : {args.base_checkpoint}")
    print(f"class_names         : {cfg.class_names}")
    print(f"combined_scores     : {len(all_scores)}")
    print(f"remove_topk         : {topk}")
    print(f"exclude batch rows  : {sum(len(v) for v in exclude_indices.values())}")
    print(f"dry_run             : {args.dry_run}")
    print("=" * 88)

    if args.dry_run:
        print("[dry-run] wrote counterfactual_removal.json and skipped training.")
        return

    train(cfg)


if __name__ == "__main__":
    main()
