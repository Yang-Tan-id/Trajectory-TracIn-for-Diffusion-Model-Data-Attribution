import argparse
import pickle
from pathlib import Path

import numpy as np

from .data import build_dataset
from .utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Create LDS subset index files.")
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--dataset-kind", default="synthetic", choices=["synthetic", "cifar2", "cifar10"])
    parser.add_argument("--output-dir", default="runs/cifar2/lds/indices")
    parser.add_argument("--num-subsets", type=int, default=64)
    parser.add_argument("--subset-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--center-crop", action="store_true")
    parser.add_argument("--synthetic-samples", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    dataset = build_dataset(args, split="train")
    if args.subset_size > len(dataset):
        raise ValueError(f"subset-size {args.subset_size} exceeds dataset size {len(dataset)}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    full_indices = np.arange(len(dataset))
    with (output_dir / "idx-train.pkl").open("wb") as f:
        pickle.dump(full_indices.tolist(), f)
    for subset_id in range(args.num_subsets):
        subset = rng.choice(full_indices, size=args.subset_size, replace=False)
        subset.sort()
        with (output_dir / f"sub-idx-{subset_id}.pkl").open("wb") as f:
            pickle.dump(subset.tolist(), f)
    print(f"wrote {args.num_subsets} subsets of {args.subset_size} to {output_dir}")


if __name__ == "__main__":
    main()
