import argparse
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Convert train denoising losses into DAS1 train residual weights.")
    parser.add_argument("--losses", required=True, help="Pickle from torch_das.eval_loss, shape N x T")
    parser.add_argument("--output", default="runs/cifar2/error/error_train.npy")
    parser.add_argument("--mode", choices=["das_clone", "mean_sqrt"], default="das_clone")
    return parser.parse_args()


def main():
    args = parse_args()
    with Path(args.losses).open("rb") as f:
        losses = np.asarray(pickle.load(f), dtype=np.float32)
    if losses.ndim != 2:
        raise ValueError(f"expected losses with shape N x T, got {losses.shape}")
    root_loss = np.sqrt(np.maximum(losses, 0.0))
    if args.mode == "das_clone":
        norm = np.linalg.norm(root_loss, axis=1, keepdims=True)
        error_train = (root_loss / (norm + 1e-8)).mean(axis=1)
    else:
        error_train = root_loss.mean(axis=1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, error_train.astype(np.float32))
    print(f"loaded losses {losses.shape}")
    print(f"saved error_train {error_train.shape} to {output}")
    print(f"min/mean/max: {error_train.min():.6g} {error_train.mean():.6g} {error_train.max():.6g}")


if __name__ == "__main__":
    main()
