import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def parse_int_list(text):
    return [int(part) for part in text.split(",") if part != ""]


def load_losses(loss_root, subset_id, seeds, eval_seeds, pattern):
    arrays = []
    for seed in seeds:
        for eval_seed in eval_seeds:
            path = Path(loss_root) / pattern.format(subset=subset_id, seed=seed, eval_seed=eval_seed)
            with path.open("rb") as f:
                arr = pickle.load(f)
            arr = np.asarray(arr)
            if arr.ndim == 2:
                arr = arr.mean(axis=1)
            arrays.append(arr)
    return np.stack(arrays, axis=0).mean(axis=0)


def load_subset_mask(subset_index_dir, subset_id, train_indices):
    path = Path(subset_index_dir) / f"sub-idx-{subset_id}.pkl"
    with path.open("rb") as f:
        subset = np.asarray(pickle.load(f))
    return np.isin(train_indices, subset)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LDS Spearman correlation from attribution scores and subset losses.")
    parser.add_argument("--scores", required=True, help=".npy scores; query x train preferred, train vector accepted")
    parser.add_argument("--subset-index-dir", required=True)
    parser.add_argument("--loss-root", required=True)
    parser.add_argument("--output", default="runs/cifar2/lds/lds_results.csv")
    parser.add_argument("--num-subsets", type=int, default=64)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--eval-seeds", default="0")
    parser.add_argument("--loss-pattern", default="subset_{subset:04d}/seed_{seed}/eval_seed_{eval_seed}/losses.pkl")
    parser.add_argument("--train-index", default=None, help="Optional idx-train.pkl; defaults to arange(train columns)")
    return parser.parse_args()


def main():
    args = parse_args()
    scores = np.load(args.scores)
    if scores.ndim == 1:
        scores = scores[None, :]
    n_query, n_train = scores.shape
    if args.train_index:
        with Path(args.train_index).open("rb") as f:
            train_indices = np.asarray(pickle.load(f))
    else:
        train_indices = np.arange(n_train)
    if len(train_indices) != n_train:
        raise ValueError(f"train-index length {len(train_indices)} != score train dimension {n_train}")

    seeds = parse_int_list(args.seeds)
    eval_seeds = parse_int_list(args.eval_seeds)
    masks = []
    losses = []
    for subset_id in range(args.num_subsets):
        masks.append(load_subset_mask(args.subset_index_dir, subset_id, train_indices))
        losses.append(load_losses(args.loss_root, subset_id, seeds, eval_seeds, args.loss_pattern))
    masks = np.stack(masks).astype(np.float32)
    losses = np.stack(losses)
    if losses.shape[1] != n_query:
        raise ValueError(f"loss query dimension {losses.shape[1]} != score query dimension {n_query}")

    preds = masks @ (-scores.T)
    rows = []
    rs = []
    ps = []
    for query_id in range(n_query):
        r, p = spearmanr(preds[:, query_id], losses[:, query_id])
        rows.append((query_id, r, p))
        rs.append(r)
        ps.append(p)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "spearman_r", "p_value"])
        writer.writerows(rows)
        writer.writerow(["mean", float(np.nanmean(rs)), float(np.nanmean(ps))])
    print(f"saved LDS results to {output}")
    print(f"mean Spearman: {np.nanmean(rs):.4f}")


if __name__ == "__main__":
    main()
