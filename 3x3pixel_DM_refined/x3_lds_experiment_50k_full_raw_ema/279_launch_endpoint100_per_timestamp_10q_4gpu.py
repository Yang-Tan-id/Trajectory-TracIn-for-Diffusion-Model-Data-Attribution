"""Launch the 100-timestamp per-t aligned run on four GPUs."""

import argparse
import subprocess
import sys
import time

from exp_config import LOG_DIR


def parse_gpus(text):
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if len(values) != 4 or len(set(values)) != 4:
        raise ValueError("--gpus must contain four distinct CUDA IDs")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=640)
    parser.add_argument("--grad-microbatch-size", type=int, default=4)
    parser.add_argument("--train-t-chunk-size", type=int, default=4)
    parser.add_argument("--query-term-batch-size", type=int, default=64)
    args = parser.parse_args()
    gpus = parse_gpus(args.gpus)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "endpoint100_per_timestamp_10q_4gpu.log"
    active = {}
    with open(log_path, "a", buffering=1) as stream:
        stream.write(
            "\n[launcher] q00-q09; 10 checkpoint pairs; all 100 trajectory t; "
            "per-t inverse-noise alignment; full AdamW; projected4096; "
            "same projection seed as endpoint20\n"
        )
        for shard_index, gpu in enumerate(gpus):
            command = [
                sys.executable,
                "-u",
                "277_run_endpoint100_per_timestamp_shard.py",
                "--gpu", str(gpu),
                "--checkpoint-shard-index", str(shard_index),
                "--checkpoint-shard-count", str(len(gpus)),
                "--batch-size", str(args.batch_size),
                "--grad-microbatch-size", str(args.grad_microbatch_size),
                "--train-t-chunk-size", str(args.train_t_chunk_size),
                "--query-term-batch-size", str(args.query_term_batch_size),
            ]
            label = f"checkpoint-shard-{shard_index}"
            stream.write(f"[launcher] {label}: {' '.join(command)}\n")
            process = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT
            )
            active[label] = process
            print(f"[launcher] {label} gpu={gpu} pid={process.pid}", flush=True)
        print(f"[launcher] log={log_path}", flush=True)
        while active:
            for label, process in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[label]
                print(f"[launcher] {label} code={code}", flush=True)
                if code:
                    for other in active.values():
                        other.terminate()
                    for other in active.values():
                        other.wait()
                    raise SystemExit(code)
            if active:
                time.sleep(2)
        subprocess.run(
            [
                sys.executable,
                "-u",
                "278_merge_eval_endpoint100_per_timestamp.py",
                "--checkpoint-shard-count", str(len(gpus)),
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(f"[done] endpoint100 per-t run; log={log_path}", flush=True)


if __name__ == "__main__":
    main()
