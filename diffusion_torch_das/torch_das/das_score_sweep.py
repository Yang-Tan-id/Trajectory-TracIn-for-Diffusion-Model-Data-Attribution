import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


DEFAULT_LAMBDAS = (
    "1e-2,2e-2,5e-2,"
    "1e-1,2e-1,5e-1,"
    "1e0,2e0,5e0,"
    "1e1,2e1,5e1,"
    "1e2,2e2,5e2,"
    "1e3,2e3,5e3,"
    "1e4,2e4,5e4,"
    "1e5,2e5,5e5,"
    "1e6,2e6,5e6"
)


def parse_shape(text):
    rows, cols = text.split(",")
    return int(rows), int(cols)


def parse_int_list(text):
    return [int(part) for part in text.split(",") if part]


def parse_float_list(text):
    return [float(part) for part in text.split(",") if part]


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


def batched_mm(left, right, batch_size):
    blocks = []
    for start in range(0, left.shape[0], batch_size):
        blocks.append(left[start : start + batch_size] @ right)
    return torch.cat(blocks, dim=0)


def spearman_by_query(scores, masks, losses):
    preds = masks @ (-scores.T)
    rows = []
    rs = []
    ps = []
    for query_id in range(scores.shape[0]):
        r, p = spearmanr(preds[:, query_id], losses[:, query_id])
        rows.append((query_id, r, p))
        rs.append(r)
        ps.append(p)
    return rows, float(np.nanmean(rs)), float(np.nanmean(ps))


def parse_args():
    parser = argparse.ArgumentParser(description="Original-DAS-style lambda sweep for LDS.")
    parser.add_argument("--train-grads", required=True)
    parser.add_argument("--query-grads", required=True)
    parser.add_argument("--error-train", required=True)
    parser.add_argument("--train-shape", required=True)
    parser.add_argument("--query-shape", required=True)
    parser.add_argument("--subset-index-dir", required=True)
    parser.add_argument("--loss-root", required=True)
    parser.add_argument("--output-dir", default="runs/cifar2/lds/original_das")
    parser.add_argument("--num-subsets", type=int, default=64)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--eval-seeds", default="0")
    parser.add_argument("--loss-pattern", default="subset_{subset:04d}/seed_{seed}/eval_seed_{eval_seed}/losses.pkl")
    parser.add_argument("--train-index", default=None)
    parser.add_argument("--lambdas", default=DEFAULT_LAMBDAS)
    parser.add_argument("--method", choices=["dtrak", "das1", "das0"], default="das1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--normalize-inv", action="store_true", help="Match original code: divide inverse by mean absolute value.")
    parser.add_argument("--save-best-scores", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    train_shape = parse_shape(args.train_shape)
    query_shape = parse_shape(args.query_shape)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_np = np.memmap(args.train_grads, dtype=np.float32, mode="r", shape=train_shape)
    query_np = np.memmap(args.query_grads, dtype=np.float32, mode="r", shape=query_shape)
    error_train = np.load(args.error_train).astype(np.float32)
    if error_train.shape != (train_shape[0],):
        raise ValueError(f"error-train shape {error_train.shape} != ({train_shape[0]},)")

    if args.train_index:
        with Path(args.train_index).open("rb") as f:
            train_indices = np.asarray(pickle.load(f))
    else:
        train_indices = np.arange(train_shape[0])
    if len(train_indices) != train_shape[0]:
        raise ValueError(f"train-index length {len(train_indices)} != score train dimension {train_shape[0]}")

    seeds = parse_int_list(args.seeds)
    eval_seeds = parse_int_list(args.eval_seeds)
    masks = []
    losses = []
    for subset_id in range(args.num_subsets):
        masks.append(load_subset_mask(args.subset_index_dir, subset_id, train_indices))
        losses.append(load_losses(args.loss_root, subset_id, seeds, eval_seeds, args.loss_pattern))
    masks = np.stack(masks).astype(np.float32)
    losses = np.stack(losses)
    if losses.shape[1] != query_shape[0]:
        raise ValueError(f"loss query dimension {losses.shape[1]} != query dimension {query_shape[0]}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train = torch.from_numpy(np.asarray(train_np)).to(device)
    query = torch.from_numpy(np.asarray(query_np)).to(device)
    error = torch.from_numpy(error_train).to(device)

    print(f"train gradients: {tuple(train.shape)}")
    print(f"query gradients: {tuple(query.shape)}")
    print(f"method: {args.method}")
    print("building train Gram K = G.T @ G")
    kernel = train.T @ train

    lambda_rows = []
    best = {"spearman": -np.inf, "lambda": None, "rows": None, "scores": None}
    eye = torch.eye(kernel.shape[0], dtype=kernel.dtype, device=device)
    lambdas = parse_float_list(args.lambdas)
    for lamb in lambdas:
        print(f"lambda={lamb:g}")
        kernel_inv = torch.linalg.inv(kernel + lamb * eye)
        if args.normalize_inv:
            kernel_inv = kernel_inv / kernel_inv.abs().mean()

        features = batched_mm(train, kernel_inv, args.batch_size)
        scores = batched_mm(query, features.T, args.batch_size)

        if args.method in {"das1", "das0"}:
            scores = scores * error[None, :]

        if args.method == "das0":
            leverage = (features * train).sum(dim=1)
            denom = (1.0 - leverage).clamp_min(1e-6)
            scores = scores / denom[None, :]

        scores_np = scores.detach().cpu().numpy()
        rows, mean_r, mean_p = spearman_by_query(scores_np, masks, losses)
        print(f"mean Spearman: {mean_r:.6f} avg p: {mean_p:.6f}")
        lambda_rows.append((lamb, mean_r, mean_p))
        if mean_r > best["spearman"]:
            best = {"spearman": mean_r, "lambda": lamb, "rows": rows, "scores": scores_np}

        del kernel_inv, features, scores
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (output_dir / "lambda_sweep.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lambda", "mean_spearman_r", "mean_p_value"])
        writer.writerows(lambda_rows)

    with (output_dir / "best_lds_results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "spearman_r", "p_value"])
        writer.writerows(best["rows"])
        writer.writerow(["mean", best["spearman"], ""])
        writer.writerow(["best_lambda", best["lambda"], ""])

    if args.save_best_scores:
        np.save(output_dir / "best_query_train_scores.npy", best["scores"])

    print(f"best lambda: {best['lambda']:g}")
    print(f"best mean Spearman: {best['spearman']:.6f}")
    print(f"saved sweep to {output_dir / 'lambda_sweep.csv'}")
    print(f"saved best LDS to {output_dir / 'best_lds_results.csv'}")


if __name__ == "__main__":
    main()
