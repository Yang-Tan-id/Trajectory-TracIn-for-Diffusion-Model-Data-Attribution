#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from run_cifar5_multi_experiment import Job, gpu_env, parse_gpus, run_parallel_jobs, slot_for, worker_gpus
from run_cifar5_multi_random_prompted_queries import build_query_specs, query_tag
from run_cifar5_multi_traj_tracin_norm_sweep import query_artifact


ADDON_A_POSITIONS = (50, 161, 272, 383, 494, 605, 716, 827, 938, 988)
ADDON_B_POSITIONS = (20, 141, 242, 363, 474, 585, 696, 807, 918, 968)


def artifact_complete(path: Path, expected_terms: int = 490) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            return (
                "query_features" in payload.files
                and np.asarray(payload["query_features"]).shape[0] == expected_terms
            )
    except Exception:
        return False


def run_checked(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(f"[repair] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def merge_query_artifacts(
    source: Path,
    supplement: Path,
    output: Path,
    *,
    supplement_timestep: int,
) -> None:
    with np.load(source, allow_pickle=False) as payload:
        source_data = {key: np.asarray(payload[key]) for key in payload.files}
    with np.load(supplement, allow_pickle=False) as payload:
        supplement_data = {key: np.asarray(payload[key]) for key in payload.files}
    source_terms = int(source_data["query_features"].shape[0])
    supplement_terms = int(supplement_data["query_features"].shape[0])
    supplement_timesteps = np.asarray(supplement_data["timesteps"]).reshape(-1)
    supplement_mask = supplement_timesteps == int(supplement_timestep)
    if int(supplement_mask.sum()) != 49:
        raise ValueError(
            f"expected 49 supplemental t={supplement_timestep} terms in {supplement}, "
            f"found {int(supplement_mask.sum())}"
        )
    merged = {}
    for key, source_value in source_data.items():
        supplement_value = supplement_data.get(key)
        if (
            supplement_value is not None
            and source_value.ndim > 0
            and supplement_value.ndim > 0
            and source_value.shape[0] == source_terms
            and supplement_value.shape[0] == supplement_terms
            and source_value.shape[1:] == supplement_value.shape[1:]
        ):
            merged[key] = np.concatenate(
                [source_value, supplement_value[supplement_mask]], axis=0
            )
        else:
            merged[key] = source_value
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp.npz")
    np.savez_compressed(temp, **merged)
    temp.replace(output)


def run_worker(root: Path, args: argparse.Namespace, spec_index: int) -> None:
    spec = build_query_specs(args)[spec_index]
    query = str(spec["query"])
    seed = int(spec["initial_seed"])
    tag = query_tag(query)
    temp_root = (
        root / "result" / args.experiment / "sample" / ".traj_query_repair"
        / f"query_{tag}_seed_{seed}"
    )
    model_root = (
        temp_root / "cifar" / f"prompt_{tag}"
        / f"model_prompted_solo__ckpt_seed_{args.train_seed}_epoch_{args.epochs:04d}"
    )
    sample_dir = model_root / f"seed_{seed:06d}"

    base_env = os.environ.copy()
    base_env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHON_BIN": args.python_bin,
            "EXPERIMENT_TAG": args.experiment,
            "CIFAR5_MULTI_SIZE": str(args.size),
            "TRAIN_SEED": str(args.train_seed),
            "JAX_EPOCHS": str(args.epochs),
            "QUERY": query,
            "INITIAL_SEED": str(seed),
            "SAMPLE_SEEDS": str(seed),
            "SAMPLE_MODEL_MODE": "prompted_solo",
            "ATTRIBUTION_SAMPLE_MODEL_MODE": "prompted_solo",
            "ATTRIBUTION_SCORE_MODEL_MODE": "prompted_solo",
            "SAMPLE_ROOT": str(temp_root),
            "SAMPLE_TRAJECTORY_STEPS": "1000",
            "TRAJ_QUERY_OBJECTIVE": "trajectory_next_checkpoint_noise_mse",
            "TRAJ_PARAMETER_SOURCE": "raw",
            "TRACIN_PARAMETER_SOURCE": "raw",
            "TRAJ_USE_SAVED_TRAJECTORY": "1",
            "TRAJ_QUERY_USE_CONFIG_SNAPSHOTS": "1",
            "TRAJ_NUM_SNAPSHOTS": "2",
            "TRAJ_SNAPSHOT_CHUNK_SIZE": str(args.snapshot_chunk_size),
            "TRAJ_TRACIN_PROJ_DIM": "4096",
            "PROJECTED_CACHE_DIM": "4096",
            "PROJECTED_DIMS": "4096",
            "JAX_BFLOAT16": "1",
            "JAX_PREFETCH_SIZE": "1",
            "TF_GPU_ALLOCATOR": "cuda_malloc_async",
        }
    )

    outputs = (
        (
            args.addon_a_source_namespace,
            args.addon_a_query_namespace,
            11,
        ),
        (
            args.addon_b_source_namespace,
            args.addon_b_query_namespace,
            31,
        ),
    )
    if all(
        artifact_complete(query_artifact(root, args, spec, exact_namespace), expected_terms=4949)
        for _source_namespace, exact_namespace, _timestep in outputs
    ):
        print(f"[repair] both exact query artifacts already complete: {tag} seed={seed}", flush=True)
        return

    try:
        if not (sample_dir / "trajectory_xt.npy").is_file():
            run_checked(
                [args.python_bin, str(root / "sampling" / "run_sampling.py")],
                cwd=root.parent,
                env=base_env,
            )
        if not (sample_dir / "trajectory_xt.npy").is_file():
            raise FileNotFoundError(f"full temporary trajectory was not created: {sample_dir}")

        supplement_namespace = args.supplement_namespace
        supplement = query_artifact(root, args, spec, supplement_namespace)
        if not artifact_complete(supplement, expected_terms=98):
            if supplement.exists():
                supplement.unlink()
            env = base_env.copy()
            env.update(
                {
                    "ATTRIBUTION_ARTIFACT_NAMESPACE": supplement_namespace,
                    "TRAJ_ATTRIBUTION_ARTIFACT_NAMESPACE": supplement_namespace,
                    "ATTRIBUTION_SAMPLE_DIR": str(model_root),
                    "TRAJ_SNAPSHOT_POSITIONS": "968,988",
                    "QUERY_GRADIENT_ARTIFACT_PATH": str(supplement),
                }
            )
            run_checked(
                [args.python_bin, "02_query_gradient.py"],
                cwd=root / "data_attribution" / "traj_tracin",
                env=env,
            )
        if not artifact_complete(supplement, expected_terms=98):
            raise RuntimeError(f"combined supplement has unexpected shape: {supplement}")

        for source_namespace, exact_namespace, timestep in outputs:
            source = query_artifact(root, args, spec, source_namespace)
            output = query_artifact(root, args, spec, exact_namespace)
            if artifact_complete(output, expected_terms=4949):
                print(f"[repair] exact query artifact already complete: {output}", flush=True)
                continue
            if not artifact_complete(source, expected_terms=4900):
                raise FileNotFoundError(f"missing original 100-grid query artifact: {source}")
            if output.exists():
                output.unlink()
            merge_query_artifacts(
                source,
                supplement,
                output,
                supplement_timestep=timestep,
            )
            if not artifact_complete(output, expected_terms=4949):
                raise RuntimeError(f"merged query artifact has unexpected shape: {output}")
            print(f"[repair] wrote exact 10-timestep query artifact: {output}", flush=True)
        shutil.rmtree(supplement.parent.parent, ignore_errors=True)
    finally:
        if not args.keep_temporary_trajectories and temp_root.exists():
            shutil.rmtree(temp_root)
            print(f"[repair] deleted temporary trajectory: {temp_root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate exact CIFAR5 add-on A/B query gradients using matching train timestamps."
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--experiment", default="cifar5_multi_exp1")
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num-queries", type=int, default=20)
    parser.add_argument("--random-query-seed", type=int, default=0)
    parser.add_argument("--initial-seed-start", type=int, default=1000)
    parser.add_argument("--initial-seeds", default="")
    parser.add_argument("--extra-prompted-queries", default="")
    parser.add_argument("--extra-initial-seed", type=int, default=0)
    parser.add_argument(
        "--addon-a-source-namespace",
        default="raw_nextckpt_school_traj_addon_mid10",
    )
    parser.add_argument(
        "--addon-a-query-namespace",
        default="raw_nextckpt_school_traj_addon_mid10_exact10",
    )
    parser.add_argument(
        "--addon-b-source-namespace",
        default="raw_nextckpt_school_traj_addon_mid10_b",
    )
    parser.add_argument(
        "--addon-b-query-namespace",
        default="raw_nextckpt_school_traj_addon_mid10_b_exact10",
    )
    parser.add_argument(
        "--supplement-namespace",
        default="raw_nextckpt_school_traj_addon_ab_missing2",
    )
    parser.add_argument("--snapshot-chunk-size", type=int, default=10)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--gpu-per-node", type=int, default=4)
    parser.add_argument("--cpus-per-worker", type=int, default=8)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--slot-backend", choices=("local", "ibrun", "srun"), default="local")
    parser.add_argument("--use-task-affinity", action="store_true")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--keep-temporary-trajectories", action="store_true")
    parser.add_argument("--worker-spec-index", type=int, default=-1)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    repo_root = root.parent.parent
    specs = build_query_specs(args)
    print("Exact add-on query-gradient repair", flush=True)
    print(f"queries={len(specs)}", flush=True)
    print(f"addon_a_positions={ADDON_A_POSITIONS}", flush=True)
    print(f"addon_b_positions={ADDON_B_POSITIONS}", flush=True)
    print(f"addon_a_query_namespace={args.addon_a_query_namespace}", flush=True)
    print(f"addon_b_query_namespace={args.addon_b_query_namespace}", flush=True)
    print("supplements: addon_a position=988 t=11; addon_b position=968 t=31", flush=True)
    if not args.execute:
        print("[dry-run] add --execute to generate exact query gradients", flush=True)
        return
    if args.worker_spec_index >= 0:
        run_worker(root, args, args.worker_spec_index)
        return

    gpus = parse_gpus(args)
    worker_gpu_ids = worker_gpus(args, gpus)
    jobs = []
    for spec_index, spec in enumerate(specs):
        slot = slot_for(spec_index, len(worker_gpu_ids))
        jobs.append(
            Job(
                name=f"repair_addon_query_{query_tag(str(spec['query']))}_seed_{spec['initial_seed']}",
                cmd=[
                    sys.executable,
                    str(Path(__file__).resolve()),
                    *sys.argv[1:],
                    "--worker-spec-index",
                    str(spec_index),
                ],
                cwd=repo_root,
                env=gpu_env(os.environ.copy(), worker_gpu_ids[slot]),
                log_path=(
                    root / "result" / args.experiment / "logs" / "repair_addon_queries"
                    / f"query_{query_tag(str(spec['query']))}_seed_{spec['initial_seed']}_gpu_{worker_gpu_ids[slot]}.log"
                ),
                slot=slot,
            )
        )
    run_parallel_jobs(
        jobs,
        args=args,
        execute=True,
        max_parallel=max(1, min(args.max_parallel, len(worker_gpu_ids))),
    )
    print("[done] exact add-on query gradients generated", flush=True)


if __name__ == "__main__":
    main()
