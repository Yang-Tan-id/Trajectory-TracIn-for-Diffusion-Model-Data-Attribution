#!/usr/bin/env python3
"""Compare Q8 probe 7 with the other timestamp-shared probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
for root in (SHAPES_ROOT, REFINE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from analyze_two_timestamp_shared_probe_score_average import load_scores  # noqa: E402
from dataset_config import _prompt_tag  # noqa: E402


SEEDS = (20260917, 20260918, 73194261, 418507293, 90216487, 563809241, 247196803, 816430927)
NAMESPACES = (
    "loss_direction_predicted_noise_probe1_timestamp_shared_checkpoint_own_trajectory_r0",
    "loss_direction_predicted_noise_probe1_timestamp_shared_seed20260918_checkpoint_own_trajectory_r0",
    "loss_direction_predicted_noise_probe1_timestamp_shared_seed73194261_checkpoint_own_trajectory_r0",
    "loss_direction_predicted_noise_probe1_timestamp_shared_seed418507293_checkpoint_own_trajectory_r0",
    "loss_direction_predicted_noise_probe1_timestamp_shared_seed90216487_checkpoint_own_trajectory_r0",
    "loss_direction_predicted_noise_probe1_timestamp_shared_seed563809241_checkpoint_own_trajectory_r0",
    "loss_direction_predicted_noise_probe1_timestamp_shared_seed247196803_checkpoint_own_trajectory_r0",
    "loss_direction_predicted_noise_probe1_timestamp_shared_seed816430927_checkpoint_own_trajectory_r0",
)
VARIANTS = {
    "raw": "score",
    "query_l2": "score_query_normalized",
    "train_l2": "score_train_l2_normalized",
    "both_l2": "score_query_train_l2_normalized",
}


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def artifact_path(result_root: Path, record: dict, namespace: str, train_seed: int) -> Path:
    checkpoint = result_root / "model" / "prompted_jax" / f"seed_{train_seed}_epoch_0200.ckpt"
    run_root = (
        result_root / "sample_ddim_eta0_1000" / "cifar"
        / f"prompt_{_prompt_tag(str(record['prompt']))}"
        / f"model_prompted_solo__ckpt_{checkpoint.stem}"
    )
    return (
        run_root / f"seed_{int(record['initial_seed']):06d}_query_gradient_{namespace}"
        / "traj_tracin" / "query_gradient_artifact.npz"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--query", type=int, default=8)
    args = parser.parse_args()

    result_root = SHAPES_ROOT / "result" / args.experiment
    record = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"][args.query]
    banks = []
    ckpts = timesteps = None
    for namespace in NAMESPACES:
        path = artifact_path(result_root, record, namespace, args.train_seed)
        with np.load(path, allow_pickle=False) as payload:
            banks.append(np.asarray(payload["query_features"], dtype=np.float64))
            current_ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            current_timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        if ckpts is None:
            ckpts, timesteps = current_ckpts, current_timesteps
        elif not (np.array_equal(ckpts, current_ckpts) and np.array_equal(timesteps, current_timesteps)):
            raise ValueError(f"metadata mismatch: {path}")
    bank = np.stack(banks)
    assert timesteps is not None
    ref = bank[6]

    print(f"Q{args.query} P7 PARAMETER-SPACE QUERY-DIRECTION GEOMETRY")
    print(f"{'PAIR':>6} {'COS':>9} {'|COS|':>9} {'COS<0':>8} {'NORM7/NORMR':>13}")
    print("-" * 52)
    for other in range(8):
        if other == 6:
            continue
        dots = np.sum(ref * bank[other], axis=1)
        ref_norm = np.linalg.norm(ref, axis=1)
        other_norm = np.linalg.norm(bank[other], axis=1)
        cosine = dots / np.maximum(ref_norm * other_norm, 1e-30)
        print(
            f"P7/P{other + 1:1d} {np.mean(cosine):+9.4f} "
            f"{np.mean(np.abs(cosine)):9.4f} {np.mean(cosine < 0):8.3f} "
            f"{np.mean(ref_norm / np.maximum(other_norm, 1e-30)):13.4f}"
        )

    print("\nBY TIMESTAMP: P7 vs mean(other seven)")
    print(f"{'T':>4} {'COS':>9} {'|COS|':>9} {'COS<0':>8} {'NORM7/OTHER':>13}")
    print("-" * 51)
    for timestep in sorted(np.unique(timesteps), reverse=True):
        mask = timesteps == timestep
        cosines = []
        ratios = []
        for other in range(8):
            if other == 6:
                continue
            a, b = ref[mask], bank[other, mask]
            an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
            cosines.extend((np.sum(a * b, axis=1) / np.maximum(an * bn, 1e-30)).tolist())
            ratios.extend((an / np.maximum(bn, 1e-30)).tolist())
        values = np.asarray(cosines)
        print(
            f"{int(timestep):4d} {np.mean(values):+9.4f} {np.mean(np.abs(values)):9.4f} "
            f"{np.mean(values < 0):8.3f} {np.mean(ratios):13.4f}"
        )

    prompt_tag = _prompt_tag(str(record["prompt"]))
    score_root = (
        result_root / "attribution_score" / "prompted_solo"
        / f"train_seed_{args.train_seed}" / f"query_{prompt_tag}"
        / f"initial_seed_{int(record['initial_seed'])}"
    )
    print("\nQ8 ROOT-SCORE RANK CORRELATION: P7 vs OTHER PROBES")
    print(f"{'PAIR':>6} " + " ".join(f"{name.upper():>10}" for name in VARIANTS))
    print("-" * 54)
    for other in range(8):
        if other == 6:
            continue
        values = []
        for variant, directory in VARIANTS.items():
            def score(probe_index: int) -> np.ndarray:
                seed = SEEDS[probe_index]
                namespace = (
                    f"traj_tracin_predicted_noise_jvp_probe_l2_"
                    f"timestamp_shared_individual_seed{seed}_own_trajectory"
                )
                _, result = load_scores(score_root / namespace / directory)
                return result
            values.append(corr(rankdata(score(6)), rankdata(score(other))))
        print(f"P7/P{other + 1:1d} " + " ".join(f"{value:+10.4f}" for value in values))


if __name__ == "__main__":
    main()
