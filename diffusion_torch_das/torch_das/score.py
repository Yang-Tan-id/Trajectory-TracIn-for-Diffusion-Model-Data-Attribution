import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Score train points against query gradients.")
    parser.add_argument("--train-grads", required=True)
    parser.add_argument("--query-grads", required=True)
    parser.add_argument("--output", default="runs/smoke/scores.npy")
    parser.add_argument("--train-shape", required=True, help="rows,projection_dim for memmap input")
    parser.add_argument("--query-shape", required=True, help="rows,projection_dim for memmap input")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--method", choices=["dot", "ridge"], default="ridge")
    return parser.parse_args()


def parse_shape(text):
    rows, cols = text.split(",")
    return int(rows), int(cols)


def main():
    args = parse_args()
    train_shape = parse_shape(args.train_shape)
    query_shape = parse_shape(args.query_shape)
    train = np.memmap(args.train_grads, dtype=np.float32, mode="r", shape=train_shape)
    query = np.memmap(args.query_grads, dtype=np.float32, mode="r", shape=query_shape)
    query_vec = np.asarray(query).mean(axis=0)
    if args.method == "dot":
        scores = np.asarray(train @ query_vec)
    else:
        gram = np.asarray(train.T @ train)
        inv = np.linalg.inv(gram + args.ridge * np.eye(gram.shape[0], dtype=np.float32))
        scores = np.asarray(train @ (inv @ query_vec))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, scores)
    print(f"saved scores {scores.shape} to {output}")
    order = np.argsort(-scores)[:10]
    print("top10:", order.tolist())


if __name__ == "__main__":
    main()
