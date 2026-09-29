"""Launch q00-q99 replayed Diffusion-TracIn/ReTrac on four GPUs."""

import argparse
import subprocess
import sys

from diffusion_retrac_config import CUDA_IDS, LOG_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=2)
    args = parser.parse_args()
    if len(CUDA_IDS) != 4:
        raise ValueError("launcher requires exactly four CUDA_IDS")
    subprocess.run([sys.executable, "111_verify_diffusion_retrac.py"], check=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    processes = []
    handles = []
    jobs = [
        ("prompted", 0, 2, CUDA_IDS[0]),
        ("prompted", 1, 2, CUDA_IDS[1]),
        ("unprompted", 0, 2, CUDA_IDS[2]),
        ("unprompted", 1, 2, CUDA_IDS[3]),
    ]
    for family, shard_index, shard_count, gpu in jobs:
        log_path = LOG_DIR / f"diffusion_retrac_100q_{family}_gpu{gpu}.log"
        handle = open(log_path, "a", buffering=1)
        command = [
            sys.executable,
            "-u",
            "112_run_diffusion_retrac_checkpoint_shard.py",
            "--family",
            family,
            "--gpu",
            str(gpu),
            "--checkpoint-shard-index",
            str(shard_index),
            "--checkpoint-shard-count",
            str(shard_count),
            "--batch-size",
            str(args.batch_size),
            "--query-batch-size",
            str(args.query_batch_size),
        ]
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((family, shard_index, gpu, process, log_path))
        handles.append(handle)
        print(
            f"[launcher] family={family} shard={shard_index}/{shard_count} "
            f"gpu={gpu} pid={process.pid} log={log_path}",
            flush=True,
        )
    failed = []
    for family, shard_index, gpu, process, log_path in processes:
        code = process.wait()
        print(
            f"[launcher] family={family} shard={shard_index} gpu={gpu} code={code}",
            flush=True,
        )
        if code != 0:
            failed.append((family, shard_index, gpu, code, str(log_path)))
    for handle in handles:
        handle.close()
    if failed:
        raise RuntimeError(f"Diffusion-ReTrac workers failed: {failed}")
    subprocess.run(
        [sys.executable, "113_merge_diffusion_retrac_shards.py"], check=True
    )
    subprocess.run(
        [sys.executable, "114_eval_diffusion_retrac_10q_lds.py"], check=True
    )
    print("[done] Diffusion-TracIn + Diffusion-ReTrac q00-q99 and LDS", flush=True)


if __name__ == "__main__":
    main()
