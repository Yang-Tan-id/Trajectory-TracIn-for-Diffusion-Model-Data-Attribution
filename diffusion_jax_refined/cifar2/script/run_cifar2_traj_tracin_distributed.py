#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path


TRAIN_ARTIFACT = "train_datapoint_gradient_artifact.npz"
QUERY_ARTIFACT = "query_gradient_artifact.npz"
TARGET_FUNCTIONS = (
    "noise_trajectory",
    "endpoint_contarfactual",
    "traj_contarfactual",
    "simple_loss",
)


@dataclass
class QuerySpec:
    query: str
    seed: int
    sample_mode: str
    score_mode: str
    unprompted: bool = False


@dataclass
class Job:
    name: str
    cmd: list[str]
    cwd: Path
    env: dict[str, str]
    log_path: Path
    slot: int = 0


def query_tag(query: str) -> str:
    if query == "unprompted":
        return "unprompted"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", query.replace(",", "_"))
    return re.sub(r"_+", "_", text).strip("_")


def artifact_namespace(args: argparse.Namespace) -> str:
    raw = str(getattr(args, "artifact_namespace", "") or "").strip()
    if not raw:
        return ""
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)
    return safe.strip("._-")


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in text.replace(" ", ",").split(",") if part.strip()]


def parse_gpus(args: argparse.Namespace) -> list[str]:
    if args.gpus:
        return parse_csv(args.gpus)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return parse_csv(visible)
    return ["0"]


def worker_gpus(args: argparse.Namespace, gpus: list[str]) -> list[str]:
    slots = int(args.slots) if args.slots is not None else len(gpus)
    gpu_per_node = max(1, int(args.gpu_per_node))
    return [gpus[i % min(len(gpus), gpu_per_node)] for i in range(slots)]


def slot_for(index: int, worker_count: int) -> int:
    return index % worker_count


def gpu_env(env: dict[str, str], gpu: str) -> dict[str, str]:
    child = env.copy()
    child["CUDA_VISIBLE_DEVICES"] = str(gpu)
    child["JAX_DATA_PARALLEL"] = "0"
    child["JAX_NUM_DEVICES"] = "1"
    child["LDS_NUM_DEVICES"] = "1"
    return child


def launch_cmd(args: argparse.Namespace, job: Job) -> list[str]:
    if args.slot_backend == "ibrun":
        cmd = ["ibrun", "-n", "1", "-o", str(job.slot)]
        if args.use_task_affinity:
            cmd.append("task_affinity")
        return cmd + job.cmd
    if args.slot_backend == "srun":
        return [
            "srun",
            "--nodes=1",
            "--ntasks=1",
            "--exclusive",
            "--gres=gpu:1",
            "--cpus-per-task",
            str(args.cpus_per_worker),
        ] + job.cmd
    return job.cmd


def run(cmd: list[str], env: dict[str, str], *, cwd: Path, execute: bool) -> None:
    prefix = "RUN" if execute else "DRY"
    print(f"[{prefix}] {' '.join(cmd)}", flush=True)
    if execute:
        subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def run_parallel_jobs(jobs: list[Job], *, args: argparse.Namespace, execute: bool, max_parallel: int) -> None:
    if not jobs:
        return
    prefix = "RUN" if execute else "DRY"
    for job in jobs:
        gpu = job.env.get("CUDA_VISIBLE_DEVICES", "?")
        print(
            f"[{prefix}][slot={job.slot}][gpu={gpu}][log={job.log_path}] "
            f"{job.name}: {' '.join(launch_cmd(args, job))}",
            flush=True,
        )
    if not execute:
        return

    active: list[tuple[Job, subprocess.Popen, object]] = []
    pending = list(jobs)
    while pending or active:
        while pending and len(active) < max_parallel:
            job = pending.pop(0)
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_f = job.log_path.open("ab")
            header = (
                f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} | {job.name} | "
                f"slot={job.slot} | gpu={job.env.get('CUDA_VISIBLE_DEVICES', '?')} =====\n"
            )
            log_f.write(header.encode("utf-8"))
            log_f.flush()
            proc = subprocess.Popen(
                launch_cmd(args, job),
                cwd=str(job.cwd),
                env=job.env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            active.append((job, proc, log_f))

        time.sleep(5)
        still_active: list[tuple[Job, subprocess.Popen, object]] = []
        for job, proc, log_f in active:
            rc = proc.poll()
            if rc is None:
                still_active.append((job, proc, log_f))
                continue
            log_f.close()
            if rc != 0:
                for _, live_proc, live_log in still_active:
                    live_proc.terminate()
                    live_log.close()
                print(f"[FAIL][{job.name}] exit={rc} log={job.log_path}", flush=True)
                raise subprocess.CalledProcessError(rc, job.cmd)
            print(f"[DONE][{job.name}] log={job.log_path}", flush=True)
        active = still_active


def split_1based_ranges(size: int, shards: int) -> list[tuple[int, int]]:
    shards = max(1, min(int(shards), int(size)))
    base = size // shards
    rem = size % shards
    out: list[tuple[int, int]] = []
    start = 1
    for i in range(shards):
        count = base + (1 if i < rem else 0)
        end = start + count - 1
        out.append((start, end))
        start = end + 1
    return out


def parse_1based_ranges(text: str, *, size: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for raw_part in text.replace(";", ",").replace(" ", ",").split(","):
        part = raw_part.strip()
        if not part:
            continue
        fields = part.replace(":", "-").split("-")
        if len(fields) != 2:
            raise ValueError(f"Bad range {part!r}; expected START-END.")
        start, end = int(fields[0]), int(fields[1])
        if start < 1 or end < start or end > size:
            raise ValueError(f"Bad range {part!r}; valid bounds are 1-{size}.")
        ranges.append((start, end))
    if not ranges:
        raise ValueError("No ranges parsed from range override.")
    return ranges


def npz_array_shape(path: Path, key: str) -> tuple[int, ...] | None:
    try:
        import numpy as np

        with zipfile.ZipFile(path) as zf:
            with zf.open(f"{key}.npy") as fh:
                version = np.lib.format.read_magic(fh)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(fh)
                else:
                    shape, _, _ = np.lib.format.read_array_header_2_0(fh)
        return tuple(int(x) for x in shape)
    except Exception:
        return None


def train_artifact_complete(path: Path, *, expected_points: int) -> bool:
    if not path.is_file():
        return False
    train_shape = npz_array_shape(path, "train_features")
    if train_shape is not None and len(train_shape) >= 2:
        return int(train_shape[-2]) == int(expected_points)
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            if "score_indices" not in data:
                return False
            return int(np.asarray(data["score_indices"]).reshape(-1).shape[0]) == int(expected_points)
    except Exception:
        return False


def score_complete(score_dir: Path) -> bool:
    return (score_dir / "scores.npy").is_file()


def eval_complete(out_dir: Path) -> bool:
    return (out_dir / "lds_summary.json").is_file()


def base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["EXPERIMENT_TAG"] = args.experiment
    env["TRAIN_SEED"] = str(args.train_seed)
    env["JAX_EPOCHS"] = str(args.epochs)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("JAX_BFLOAT16", "1")
    env.setdefault("JAX_PREFETCH_SIZE", "1")
    env.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    env.setdefault("TRAJ_TRACIN_PROJ_DIM", "4096")
    env.setdefault("PROJECTED_CACHE_DIM", "4096")
    env.setdefault("PROJECTED_DIMS", "4096")
    env.setdefault("TRAJ_QUERY_OBJECTIVE", "trajectory_next_checkpoint_noise_mse")
    env.setdefault("TRACIN_PARAMETER_SOURCE", "raw")
    env.setdefault("TRAJ_PARAMETER_SOURCE", env.get("TRACIN_PARAMETER_SOURCE", "raw"))
    env.setdefault("TRAJ_NUM_SNAPSHOTS", "10")
    env.setdefault("TRAJ_TRAIN_MC_SAMPLES", "10")
    env.setdefault("TRAJ_SCORE_BATCH_SIZE", str(args.train_score_batch_size))
    env.setdefault("TRAJ_SNAPSHOT_CHUNK_SIZE", str(args.snapshot_chunk_size))
    env.setdefault("TRAJ_TRACIN_TRAIN_BATCH_MODE", args.train_batch_mode)
    env.setdefault("TRAJ_TRACIN_TRAIN_BATCH_DTYPE", args.train_batch_dtype)
    env.setdefault("DTRAK_COUNT_SKETCH_MODE", args.countsketch_mode)
    env.setdefault("TRAJ_USE_SAVED_TRAJECTORY", "0")
    env.setdefault("TRACIN_USE_SHARED_TRAIN_GRADIENT", "1")
    env.setdefault("TRACIN_SCORE_QUERY_NORMALIZE", "0")
    env.setdefault("LDS_M", str(args.lds_m))
    env.setdefault("LDS_DATASET_PERCENTAGE", str(args.lds_percentage))
    env.setdefault("LDS_EPOCHS", str(args.lds_epochs))
    env.setdefault("LDS_SAVE_EVERY_EPOCHS", str(args.lds_epochs))
    env.setdefault("LDS_KEEP_LAST_K", "1")
    env.setdefault("LDS_NUM_DEVICES", "1")
    namespace = artifact_namespace(args)
    if namespace:
        env["ATTRIBUTION_ARTIFACT_NAMESPACE"] = namespace
        env["TRAJ_ATTRIBUTION_ARTIFACT_NAMESPACE"] = namespace
    return env


def train_artifact_path(root: Path, args: argparse.Namespace, mode: str) -> Path:
    train_root = f"seed_{args.train_seed}_train_gradient"
    namespace = artifact_namespace(args)
    if namespace:
        train_root = f"{train_root}_{namespace}"
    return root / "result" / args.experiment / "model" / mode / train_root / "traj_tracin" / TRAIN_ARTIFACT


def shard_artifact_path(root: Path, args: argparse.Namespace, mode: str, start: int, end: int) -> Path:
    return train_artifact_path(root, args, mode).parent / "datapoint_shards" / f"range_{start}_{end}" / TRAIN_ARTIFACT


def sample_dir(root: Path, args: argparse.Namespace, spec: QuerySpec) -> Path:
    prompt_tag = "unconditional" if spec.unprompted else query_tag(spec.query)
    return (
        root
        / "result"
        / args.experiment
        / "sample"
        / "cifar"
        / f"prompt_{prompt_tag}"
        / f"model_{spec.sample_mode}__ckpt_seed_{args.train_seed}_epoch_{args.epochs:04d}"
    )


def query_artifact_path(root: Path, args: argparse.Namespace, spec: QuerySpec) -> Path:
    if args.query_artifact_layout == "projected":
        return projected_query_artifact_path(root, args, spec)
    if args.query_artifact_layout == "auto":
        projected = projected_query_artifact_path(root, args, spec)
        if projected.is_file():
            return projected
    query_root = f"seed_{spec.seed:06d}_query_gradient"
    namespace = artifact_namespace(args) if args.namespace_query_gradient else ""
    if namespace:
        query_root = f"{query_root}_{namespace}"
    return sample_dir(root, args, spec) / query_root / "traj_tracin" / QUERY_ARTIFACT


def projected_query_artifact_path(root: Path, args: argparse.Namespace, spec: QuerySpec) -> Path:
    projected_root = args.projected_artifact_dir_name
    query_component = "unprompted" if spec.unprompted else f"query_{query_tag(spec.query)}"
    return (
        root
        / "result"
        / args.experiment
        / projected_root
        / spec.score_mode
        / f"train_seed_{args.train_seed}"
        / query_component
        / f"initial_seed_{spec.seed}"
        / "shared_query"
        / f"proj_{args.projected_cache_dim}"
        / QUERY_ARTIFACT
    )


def score_dir(root: Path, args: argparse.Namespace, spec: QuerySpec) -> Path:
    query_component = "unprompted" if spec.unprompted else f"query_{query_tag(spec.query)}"
    base = (
        root
        / "result"
        / args.experiment
        / "attribution_score"
        / spec.score_mode
        / f"train_seed_{args.train_seed}"
        / query_component
        / f"initial_seed_{spec.seed}"
    )
    namespace = artifact_namespace(args)
    if namespace:
        base = base / namespace
    return base / "traj_tracin" / "score"


def score_shard_dir(root: Path, args: argparse.Namespace, spec: QuerySpec, start: int, end: int) -> Path:
    return score_dir(root, args, spec) / "datapoint_shards" / f"range_{start}_{end}"


def lds_model_dirs(root: Path, args: argparse.Namespace, mode: str) -> str:
    fraction = args.lds_percentage if args.lds_percentage <= 1 else args.lds_percentage / 100.0
    k = round(args.size * fraction)
    pct_tag = f"pct_{args.lds_percentage:g}".replace(".", "p")
    seeds = [x.strip() for x in args.lds_subset_seeds.split(",") if x.strip()]
    dirs = [
        root
        / "result"
        / args.experiment
        / "lds_model"
        / mode
        / f"train_seed_{args.train_seed}"
        / f"m_{args.lds_m}_k_{k}_{pct_tag}_subset_seed_{seed}"
        for seed in seeds
    ]
    return ",".join(str(path) for path in dirs)


def lds_eval_out_dir(root: Path, args: argparse.Namespace, spec: QuerySpec, target: str) -> Path:
    query_component = "unprompted" if spec.unprompted else f"query_{query_tag(spec.query)}"
    lds_component = "lds_unprompted" if spec.unprompted else "lds"
    base = (
        root
        / "result"
        / args.experiment
        / "eval"
        / spec.score_mode
        / query_component
        / f"initial_seed_{spec.seed}"
    )
    namespace = artifact_namespace(args)
    if namespace:
        base = base / namespace
    return base / lds_component / "traj_tracin" / target


def query_env(args: argparse.Namespace, env0: dict[str, str], spec: QuerySpec) -> dict[str, str]:
    env = env0 | {
        "QUERY": "unconditional" if spec.unprompted else spec.query,
        "INITIAL_SEED": str(spec.seed),
        "SAMPLE_MODEL_MODE": spec.sample_mode,
        "ATTRIBUTION_SAMPLE_MODEL_MODE": spec.sample_mode,
        "ATTRIBUTION_SCORE_MODEL_MODE": spec.score_mode,
        "DATAPOINT_MODEL_MODE": spec.score_mode,
        "ATTRIBUTION_SAMPLE_DIR": str(sample_dir(args.root, args, spec)),
    }
    if spec.unprompted:
        env["UNPROMPTED"] = "1"
    else:
        env.pop("UNPROMPTED", None)
    return env


def query_specs(args: argparse.Namespace) -> list[QuerySpec]:
    prompted_seeds = [int(x) for x in args.prompted_seeds.replace(",", " ").split() if x.strip()]
    prompted_queries = [x.strip() for x in args.prompted_queries.replace("|", ";").split(";") if x.strip()]
    specs: list[QuerySpec] = []
    for seed in prompted_seeds:
        for query in prompted_queries:
            sample_mode = "prompted_multi" if "," in query else "prompted_solo"
            specs.append(QuerySpec(query=query, seed=seed, sample_mode=sample_mode, score_mode="prompted_solo"))
    if args.include_unprompted:
        unprompted_seeds = [int(x) for x in args.unprompted_seeds.replace(",", " ").split() if x.strip()]
        specs.extend(
            QuerySpec(
                query="unprompted",
                seed=seed,
                sample_mode="unprompted_solo",
                score_mode="unprompted_solo",
                unprompted=True,
            )
            for seed in unprompted_seeds
        )
    return specs


def train_modes(specs: list[QuerySpec]) -> list[str]:
    modes = sorted({spec.score_mode for spec in specs})
    return modes or ["prompted_solo"]


def resume_log_root(args: argparse.Namespace) -> Path:
    root = args.root / "result" / args.experiment / "logs" / "traj_tracin_distributed"
    namespace = artifact_namespace(args)
    return root / namespace if namespace else root


def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR2 clean TrajTracIn distributed train/query/score/LDS runner.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--experiment", default=os.environ.get("EXPERIMENT_TAG", "experiment_67"))
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--train-seed", type=int, default=int(os.environ.get("TRAIN_SEED", "67")))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("JAX_EPOCHS", "200")))
    parser.add_argument("--prompted-seeds", default=os.environ.get("PROMPTED_INITIAL_SEEDS", "0 1 2 3 4 5 6 7"))
    parser.add_argument("--prompted-queries", default=os.environ.get("PROMPTED_QUERIES", "horse;automobile;horse,automobile"))
    parser.add_argument("--include-unprompted", action="store_true")
    parser.add_argument("--unprompted-seeds", default=os.environ.get("UNPROMPTED_INITIAL_SEEDS", "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23"))
    parser.add_argument("--artifact-namespace", "--namespace", default=os.environ.get("ATTRIBUTION_ARTIFACT_NAMESPACE", ""))
    parser.add_argument("--namespace-query-gradient", action="store_true")
    parser.add_argument(
        "--query-artifact-layout",
        choices=("auto", "sample", "projected"),
        default=os.environ.get("CIFAR2_QUERY_ARTIFACT_LAYOUT", "auto"),
        help="auto/projected reuses the old CIFAR2 query cache; sample matches the CIFAR5 sample-directory layout.",
    )
    parser.add_argument(
        "--projected-artifact-dir-name",
        default=os.environ.get("PROJECTED_ARTIFACT_DIR_NAME", "projected_traj_tracin_artifacts_raw_next_10x10"),
    )
    parser.add_argument("--projected-cache-dim", type=int, default=int(os.environ.get("PROJECTED_CACHE_DIM", "4096")))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-query-gradient", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--skip-lds-eval", action="store_true")
    parser.add_argument("--only-train-gradient", action="store_true")
    parser.add_argument("--only-score", action="store_true")
    parser.add_argument("--only-lds-eval", action="store_true")
    parser.add_argument("--lds-m", type=int, default=int(os.environ.get("LDS_M", "64")))
    parser.add_argument("--lds-percentage", type=float, default=float(os.environ.get("LDS_DATASET_PERCENTAGE", "25")))
    parser.add_argument("--lds-subset-seeds", default=os.environ.get("LDS_SUBSET_SEEDS", "0,1,2"))
    parser.add_argument("--lds-epochs", type=int, default=int(os.environ.get("LDS_EPOCHS", "200")))
    parser.add_argument("--target-functions", default=",".join(TARGET_FUNCTIONS))
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--slots", type=int, default=None)
    parser.add_argument("--gpu-per-node", type=int, default=4)
    parser.add_argument("--cpus-per-worker", type=int, default=8)
    parser.add_argument("--max-parallel", type=int, default=int(os.environ.get("MAX_PARALLEL", "0")) or None)
    parser.add_argument("--slot-backend", choices=("local", "ibrun", "srun"), default=os.environ.get("TACC_SLOT_BACKEND", "local"))
    parser.add_argument("--use-task-affinity", action="store_true")
    parser.add_argument("--score-index-ranges", "--index-ranges", dest="index_ranges", default=os.environ.get("SCORE_INDEX_RANGES", ""))
    parser.add_argument("--train-score-batch-size", type=int, default=int(os.environ.get("TRAJ_SCORE_BATCH_SIZE", "8")))
    parser.add_argument("--snapshot-chunk-size", type=int, default=int(os.environ.get("TRAJ_SNAPSHOT_CHUNK_SIZE", "8")))
    parser.add_argument(
        "--train-batch-mode",
        choices=("vmap", "loop"),
        default=os.environ.get("TRAJ_TRACIN_TRAIN_BATCH_MODE", "vmap"),
        help="vmap matches the original batched per-example gradient path; loop runs the same per-example gradient one item at a time inside each score batch.",
    )
    parser.add_argument(
        "--train-batch-dtype",
        choices=("float32", "bfloat16"),
        default=os.environ.get("TRAJ_TRACIN_TRAIN_BATCH_DTYPE", "float32"),
        help="Input dtype for TrajTracIn train-gradient batches. float32 matches the original CIFAR5 10x10 path.",
    )
    parser.add_argument(
        "--countsketch-mode",
        choices=("scatter", "segment_sum"),
        default=os.environ.get("DTRAK_COUNT_SKETCH_MODE", "scatter"),
        help="CountSketch implementation for projected gradients. scatter matches the original CIFAR5 10x10 path.",
    )
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", "python3"))
    args = parser.parse_args()

    args.root = Path(__file__).resolve().parents[1]
    specs = query_specs(args)
    env0 = base_env(args)
    gpus = parse_gpus(args)
    worker_gpu_ids = worker_gpus(args, gpus)
    max_parallel = max(1, min(args.max_parallel or len(worker_gpu_ids), len(worker_gpu_ids)))
    ranges = parse_1based_ranges(args.index_ranges, size=args.size) if args.index_ranges.strip() else split_1based_ranges(args.size, len(worker_gpu_ids))

    if args.only_train_gradient:
        args.skip_query_gradient = True
        args.skip_score = True
        args.skip_lds_eval = True
    if args.only_score:
        args.skip_train = True
        args.skip_query_gradient = True
        args.skip_lds_eval = True
    if args.only_lds_eval:
        args.skip_train = True
        args.skip_query_gradient = True
        args.skip_score = True
        args.skip_lds_eval = False

    print(f"experiment={args.experiment} train_seed={args.train_seed} epochs={args.epochs}", flush=True)
    print(f"namespace={artifact_namespace(args) or '(none)'} namespace_query_gradient={int(args.namespace_query_gradient)}", flush=True)
    print(f"queries={len(specs)} train_modes={train_modes(specs)}", flush=True)
    for spec in specs:
        print(f"  seed={spec.seed:06d} query={spec.query} sample_mode={spec.sample_mode} score_mode={spec.score_mode}", flush=True)
    print(f"ranges={','.join(f'{s}-{e}' for s, e in ranges)}", flush=True)
    print(f"worker_gpus={worker_gpu_ids} max_parallel={max_parallel} backend={args.slot_backend}", flush=True)
    print(
        "traj settings: "
        f"objective={env0['TRAJ_QUERY_OBJECTIVE']} parameter_source={env0['TRAJ_PARAMETER_SOURCE']} "
        f"snapshots={env0['TRAJ_NUM_SNAPSHOTS']} mc={env0['TRAJ_TRAIN_MC_SAMPLES']} "
        f"train_batch={env0['TRAJ_SCORE_BATCH_SIZE']} "
        f"train_batch_mode={env0['TRAJ_TRACIN_TRAIN_BATCH_MODE']} "
        f"train_batch_dtype={env0['TRAJ_TRACIN_TRAIN_BATCH_DTYPE']} "
        f"countsketch_mode={env0['DTRAK_COUNT_SKETCH_MODE']}",
        flush=True,
    )
    print(
        f"xla_flags={env0.get('XLA_FLAGS', '(default)')} "
        f"cudnn_autotune={env0.get('TF_CUDNN_USE_AUTOTUNE', '(default)')}",
        flush=True,
    )

    if not args.skip_train:
        for mode in train_modes(specs):
            jobs: list[Job] = []
            for shard_id, (start, end) in enumerate(ranges):
                shard_path = shard_artifact_path(args.root, args, mode, start, end)
                if train_artifact_complete(shard_path, expected_points=end - start + 1):
                    print(f"[skip] train shard {mode} {start}-{end}: {shard_path}", flush=True)
                    continue
                env = env0 | {
                    "DATAPOINT_MODEL_MODE": mode,
                    "SAMPLE_MODEL_MODE": mode,
                    "ATTRIBUTION_SAMPLE_MODEL_MODE": mode,
                    "ATTRIBUTION_SCORE_MODEL_MODE": mode,
                    "SCORE_INDEX_RANGES": f"{start}-{end}",
                    "TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH": str(shard_path),
                    "TRAJ_USE_SAVED_TRAJECTORY": "0",
                }
                if mode.startswith("unprompted"):
                    env["UNPROMPTED"] = "1"
                slot = slot_for(shard_id, len(worker_gpu_ids))
                jobs.append(
                    Job(
                        name=f"traj_train_{mode}_range_{start}_{end}",
                        cmd=[args.python_bin, "01_train_datapoint_gradient.py"],
                        cwd=args.root / "data_attribution" / "traj_tracin",
                        env=gpu_env(env, worker_gpu_ids[slot]),
                        log_path=resume_log_root(args)
                        / "train_shards"
                        / mode
                        / f"range_{start}_{end}_slot_{slot}_gpu_{worker_gpu_ids[slot]}.log",
                        slot=slot,
                    )
                )
            run_parallel_jobs(jobs, args=args, execute=args.execute, max_parallel=max_parallel)
        if args.only_train_gradient:
            print("[done] train-gradient-only complete", flush=True)
            return

    if not args.skip_query_gradient:
        jobs = []
        for i, spec in enumerate(specs):
            query_artifact = query_artifact_path(args.root, args, spec)
            if query_artifact.is_file():
                print(f"[skip] query gradient {spec.query} seed={spec.seed}: {query_artifact}", flush=True)
                continue
            slot = slot_for(i, len(worker_gpu_ids))
            env = query_env(args, env0, spec) | {"QUERY_GRADIENT_ARTIFACT_PATH": str(query_artifact)}
            jobs.append(
                Job(
                    name=f"traj_query_{query_tag(spec.query)}_seed_{spec.seed}",
                    cmd=[args.python_bin, "02_query_gradient.py"],
                    cwd=args.root / "data_attribution" / "traj_tracin",
                    env=gpu_env(env, worker_gpu_ids[slot]),
                    log_path=resume_log_root(args) / "query" / f"{query_tag(spec.query)}_seed_{spec.seed}_slot_{slot}_gpu_{worker_gpu_ids[slot]}.log",
                    slot=slot,
                )
            )
        run_parallel_jobs(jobs, args=args, execute=args.execute, max_parallel=max_parallel)

    if not args.skip_score:
        for spec in specs:
            final_score_dir = score_dir(args.root, args, spec)
            if score_complete(final_score_dir):
                print(f"[skip] score complete {spec.query} seed={spec.seed}: {final_score_dir}", flush=True)
                continue
            jobs = []
            for shard_id, (start, end) in enumerate(ranges):
                shard_score = score_shard_dir(args.root, args, spec, start, end)
                if score_complete(shard_score):
                    print(f"[skip] score shard {spec.query} seed={spec.seed} {start}-{end}: {shard_score}", flush=True)
                    continue
                train_path = shard_artifact_path(args.root, args, spec.score_mode, start, end)
                query_path = query_artifact_path(args.root, args, spec)
                env = query_env(args, env0, spec) | {
                    "TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH": str(train_path),
                    "QUERY_GRADIENT_ARTIFACT_PATH": str(query_path),
                    "SCORE_OUTPUT_DIR": str(shard_score),
                }
                slot = slot_for(shard_id, len(worker_gpu_ids))
                jobs.append(
                    Job(
                        name=f"traj_score_{query_tag(spec.query)}_seed_{spec.seed}_range_{start}_{end}",
                        cmd=[args.python_bin, "03_score.py"],
                        cwd=args.root / "data_attribution" / "traj_tracin",
                        env=gpu_env(env, worker_gpu_ids[slot]),
                        log_path=resume_log_root(args)
                        / "score_shards"
                        / f"{query_tag(spec.query)}_seed_{spec.seed}"
                        / f"range_{start}_{end}_slot_{slot}_gpu_{worker_gpu_ids[slot]}.log",
                        slot=slot,
                    )
                )
            run_parallel_jobs(jobs, args=args, execute=args.execute, max_parallel=max_parallel)

            shard_dirs = [score_shard_dir(args.root, args, spec, start, end) for start, end in ranges]
            if args.execute:
                missing = [str(path / "scores.npy") for path in shard_dirs if not score_complete(path)]
                if missing:
                    raise FileNotFoundError(f"Missing score shard(s) for {spec.query} seed={spec.seed}: {missing[:3]}")
            run(
                [
                    args.python_bin,
                    str(args.root.parent / "common" / "merge_score_shards.py"),
                    "--output-dir",
                    str(final_score_dir),
                    *map(str, shard_dirs),
                ],
                env0,
                cwd=args.root,
                execute=args.execute,
            )

    if not args.skip_lds_eval:
        target_functions = [part.strip() for part in args.target_functions.replace(",", " ").split() if part.strip()]
        for spec in specs:
            score_path = score_dir(args.root, args, spec) / "scores.npy"
            if not score_path.is_file():
                print(f"[skip] missing score for LDS eval: {score_path}", flush=True)
                continue
            for target in target_functions:
                out_dir = lds_eval_out_dir(args.root, args, spec, target)
                if eval_complete(out_dir):
                    print(f"[skip] LDS eval complete: {out_dir}", flush=True)
                    continue
                cmd = [
                    args.python_bin,
                    "lds/run_eval.py",
                    "--algorithm",
                    "traj_tracin",
                    "--lds-model-dirs",
                    lds_model_dirs(args.root, args, spec.score_mode),
                    "--score-file",
                    str(score_path),
                    "--target-function",
                    target,
                    "--out-dir",
                    str(out_dir),
                ]
                if spec.unprompted:
                    cmd.insert(2, "--unprompted")
                run(cmd, query_env(args, env0, spec), cwd=args.root, execute=args.execute)

    print("[done] CIFAR2 TrajTracIn distributed flow complete", flush=True)


if __name__ == "__main__":
    main()
