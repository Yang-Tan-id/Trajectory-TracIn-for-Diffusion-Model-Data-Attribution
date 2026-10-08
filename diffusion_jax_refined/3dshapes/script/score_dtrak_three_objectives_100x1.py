from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from dataset_config import ATTRIBUTION_INDICES_PATH, _prompt_tag


OBJECTIVES = ("simple_loss", "square", "average")
DEFAULT_LAMBDAS = (
    1e-5, 3e-5, 1e-4, 3e-4,
    1e-3, 3e-3, 1e-2, 3e-2,
    1e-1, 3e-1, 1.0, 3.0,
    10.0, 30.0, 100.0, 300.0,
)


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(path)


def lambda_tag(value: float) -> str:
    return f"{float(value):g}".replace("+", "").replace("-", "neg_").replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch all Q0-Q99 D-TRAK solves for three 100x1 objectives.")
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", default=",".join(str(i) for i in range(100)))
    parser.add_argument("--objectives", default=",".join(OBJECTIVES))
    parser.add_argument(
        "--lambdas",
        default=os.environ.get(
            "DTRAK_DAMPING_SWEEP_VALUES",
            ",".join(f"{value:g}" for value in DEFAULT_LAMBDAS),
        ),
        help="Comma/space-separated ridge parameters. Train/query features are reused.",
    )
    args = parser.parse_args()

    query_ids = [int(x) for x in args.query_ids.replace(",", " ").split()]
    objectives = [x for x in args.objectives.replace(",", " ").split() if x]
    lambdas = [float(x) for x in args.lambdas.replace(",", " ").split() if x]
    if not lambdas or any(value <= 0 or not np.isfinite(value) for value in lambdas):
        raise ValueError(f"lambdas must be finite and positive, got {lambdas}")
    if len(set(lambdas)) != len(lambdas):
        raise ValueError(f"lambdas must be unique, got {lambdas}")
    invalid = sorted(set(objectives) - set(OBJECTIVES))
    if invalid:
        raise ValueError(f"invalid objectives: {invalid}")
    records = json.loads(args.query_file.read_text())["queries"]
    result_root = SHAPES_ROOT / "result" / args.experiment
    sample_root = result_root / "sample_ddim_eta0_1000" / "cifar"
    model_root = result_root / "model" / "prompted_solo" / f"seed_{args.train_seed}_train_gradient"

    for objective in objectives:
        train_path = model_root / f"dtrak_{objective}_100x1" / "train_datapoint_gradient_artifact.npz"
        with np.load(train_path, allow_pickle=False) as payload:
            train = np.asarray(payload["train_features"], dtype=np.float64)
            gram = np.asarray(payload["gram"], dtype=np.float64)
            indices = np.asarray(payload["score_indices"], dtype=np.int64)
            artifact_damping = float(np.asarray(payload["damping"]).reshape(()))
        if train.shape[0] != 1 or gram.shape[0] != 1:
            raise ValueError(f"expected one final checkpoint in {train_path}, got train={train.shape}, gram={gram.shape}")
        expected_indices = np.asarray(np.load(ATTRIBUTION_INDICES_PATH), dtype=np.int64).reshape(-1)
        if not np.array_equal(np.sort(indices), np.sort(expected_indices)):
            raise ValueError(f"{train_path} score indices do not match the fixed attribution_5k subset")

        query_features = []
        query_paths = []
        for query_id in query_ids:
            record = records[query_id]
            seed = int(record.get("initial_seed", record.get("seed")))
            model_dir = (
                sample_root
                / f"prompt_{_prompt_tag(record['prompt'])}"
                / f"model_prompted_solo__ckpt_seed_{args.train_seed}_epoch_0200"
            )
            query_path = (
                model_dir
                / f"seed_{seed:06d}_query_gradient_dtrak_{objective}_100x1"
                / "dtrak"
                / "query_gradient_artifact.npz"
            )
            with np.load(query_path, allow_pickle=False) as payload:
                feature = np.asarray(payload["query_features"], dtype=np.float64)
            if feature.shape != (1, train.shape[2]):
                raise ValueError(f"query feature mismatch at {query_path}: {feature.shape} vs {train.shape}")
            query_features.append(feature[0])
            query_paths.append(query_path)

        rhs = np.stack(query_features, axis=1)
        # Legacy D-TRAK artifacts store G + lambda_artifact I. Recover the
        # undamped Gram once, diagonalize it once, and reuse that factorization
        # for the complete lambda sweep. Tiny negative eigenvalues are solely
        # float32 roundoff because the true Gram is Phi.T @ Phi.
        gram_undamped = gram[0] - artifact_damping * np.eye(gram.shape[-1], dtype=np.float64)
        gram_undamped = 0.5 * (gram_undamped + gram_undamped.T)
        print(
            f"[factor] objective={objective} gram={gram_undamped.shape} rhs={rhs.shape} "
            f"lambdas={','.join(f'{value:g}' for value in lambdas)}",
            flush=True,
        )
        eigenvalues, eigenvectors = np.linalg.eigh(gram_undamped)
        negative_count = int(np.count_nonzero(eigenvalues < 0))
        eigenvalues = np.maximum(eigenvalues, 0.0)
        rotated_rhs = eigenvectors.T @ rhs

        for damping in lambdas:
            solved = eigenvectors @ (rotated_rhs / (eigenvalues[:, None] + damping))
            scores = (train[0] @ solved).T
            if scores.shape != (len(query_ids), len(indices)):
                raise ValueError(f"unexpected score shape {scores.shape}")

            namespace = (
                f"dtrak_{objective}_train100x1_query100x1_q0_99_"
                f"lambda_{lambda_tag(damping)}"
            )
            for row, query_id, query_path in zip(scores, query_ids, query_paths):
                record = records[query_id]
                seed = int(record.get("initial_seed", record.get("seed")))
                output = (
                    result_root
                    / "attribution_score"
                    / "prompted_solo"
                    / f"train_seed_{args.train_seed}"
                    / f"query_{_prompt_tag(record['prompt'])}"
                    / f"initial_seed_{seed}"
                    / namespace
                    / "score"
                )
                atomic_save_npy(output / "scores.npy", row.astype(np.float64))
                atomic_save_npy(output / "score_indices.npy", indices)
                manifest = {
                    "algorithm": "dtrak",
                    "output_function": objective,
                    "formula": "Phi @ solve(Phi.T @ Phi + lambda * I, phi_query)",
                    "train_timestamps": 100,
                    "train_mc_per_timestamp": 1,
                    "query_timestamps": 100,
                    "query_mc_per_timestamp": 1,
                    "timestep_grid": "uniform_explicit_0_999",
                    "num_checkpoints": 1,
                    "projection_dim": int(train.shape[2]),
                    "damping": damping,
                    "artifact_damping": artifact_damping,
                    "gram_factorization": "symmetric_eigendecomposition",
                    "clipped_negative_eigenvalues": negative_count,
                    "train_artifact": str(train_path),
                    "query_artifact": str(query_path),
                    "query_id": query_id,
                    "num_scores": int(len(indices)),
                }
                temporary = output / f".score_manifest.{os.getpid()}.tmp"
                temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
                temporary.replace(output / "score_manifest.json")
            print(
                f"[done] {objective} lambda={damping:g}: wrote {len(query_ids)} score vectors",
                flush=True,
            )


if __name__ == "__main__":
    main()
