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


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch all Q0-Q99 D-TRAK solves for three 100x1 objectives.")
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", default=",".join(str(i) for i in range(100)))
    parser.add_argument("--objectives", default=",".join(OBJECTIVES))
    args = parser.parse_args()

    query_ids = [int(x) for x in args.query_ids.replace(",", " ").split()]
    objectives = [x for x in args.objectives.replace(",", " ").split() if x]
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
            damping = float(np.asarray(payload["damping"]).reshape(()))
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
        print(
            f"[solve] objective={objective} gram={gram[0].shape} rhs={rhs.shape} "
            f"lambda={damping:g}",
            flush=True,
        )
        solved = np.linalg.solve(gram[0], rhs)
        scores = (train[0] @ solved).T
        if scores.shape != (len(query_ids), len(indices)):
            raise ValueError(f"unexpected score shape {scores.shape}")

        namespace = f"dtrak_{objective}_train100x1_query100x1_q0_99"
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
                "train_artifact": str(train_path),
                "query_artifact": str(query_path),
                "query_id": query_id,
                "num_scores": int(len(indices)),
            }
            temporary = output / f".score_manifest.{os.getpid()}.tmp"
            temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
            temporary.replace(output / "score_manifest.json")
        print(f"[done] {objective}: wrote {len(query_ids)} score vectors", flush=True)


if __name__ == "__main__":
    main()
