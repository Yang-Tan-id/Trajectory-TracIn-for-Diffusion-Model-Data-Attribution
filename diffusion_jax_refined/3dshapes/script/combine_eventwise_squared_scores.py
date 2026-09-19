#!/usr/bin/env python3
"""Sum already-squared E1--E4 score arrays without event cross terms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dataset_config import _prompt_tag

COMPONENTS = (
    "score",
    "score_query_normalized",
    "score_train_l2_normalized",
    "score_query_train_l2_normalized",
)


def storage_namespace(value: str) -> str:
    return value if value.startswith("traj_tracin_") else f"traj_tracin_{value}"


def atomic_save(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--input-namespaces", required=True)
    parser.add_argument("--output-namespace", required=True)
    args = parser.parse_args()

    inputs = [
        storage_namespace(value.strip())
        for value in args.input_namespaces.split(",")
        if value.strip()
    ]
    output_namespace = storage_namespace(args.output_namespace.strip())
    if len(inputs) != 4:
        raise ValueError(f"expected exactly E1--E4 score namespaces, got {inputs}")
    records = json.loads((ROOT / "queries_seed_0_9.json").read_text())["queries"]
    score_root = ROOT / "result" / args.experiment / "attribution_score" / "prompted_solo" / f"train_seed_{args.train_seed}"

    for query_id, record in enumerate(records):
        query_root = score_root / f"query_{_prompt_tag(str(record['prompt']))}" / f"initial_seed_{int(record['initial_seed'])}"
        for component in COMPONENTS:
            arrays = []
            indices = None
            for namespace in inputs:
                source = query_root / namespace / component
                values = np.asarray(np.load(source / "scores.npy"), dtype=np.float64)
                current_indices = np.asarray(np.load(source / "score_indices.npy"), dtype=np.int64)
                if indices is None:
                    indices = current_indices
                elif not np.array_equal(indices, current_indices):
                    raise ValueError(f"score index mismatch for query={query_id}, component={component}")
                arrays.append(values)
            destination = query_root / output_namespace / component
            atomic_save(destination / "scores.npy", np.sum(arrays, axis=0))
            atomic_save(destination / "score_indices.npy", indices)
            manifest = {
                "algorithm": output_namespace,
                "score_variant": component,
                "definition": "sum_event_E1_to_E4(mean_probe(square(per_event_directional_contraction)))",
                "event_cross_terms": False,
                "source_namespaces": inputs,
                "num_events": 4,
                "num_output_probes_per_term": 12,
            }
            (destination / "score_artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        print(f"[combined] query={query_id}: E1^2+E2^2+E3^2+E4^2", flush=True)


if __name__ == "__main__":
    main()
