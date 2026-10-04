"""Launch the q00-q09 full-AdamW TracIn-DAS noise-pairing ablation."""

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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-microbatch-size", type=int, default=4)
    parser.add_argument("--query-term-batch-size", type=int, default=128)
    args = parser.parse_args()
    gpus = parse_gpus(args.gpus)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "tracin_das_noise_pairing_ablation_4gpu.log"
    active = {}
    with open(log_path, "a", buffering=1) as stream:
        stream.write(
            "\n[launcher] q00-q09; 10 checkpoint pairs; 20 timestamps; "
            "MC10; aligned/cyclic/random/independent; full AdamW; projected4096\n"
        )
        for shard_index, gpu in enumerate(gpus):
            command = [
                sys.executable,
                "-u",
                "235_run_tracin_das_noise_pairing_ablation_shard.py",
                "--gpu",
                str(gpu),
                "--timestamp-shard-index",
                str(shard_index),
                "--timestamp-shard-count",
                "4",
                "--batch-size",
                str(args.batch_size),
                "--grad-microbatch-size",
                str(args.grad_microbatch_size),
                "--query-term-batch-size",
                str(args.query_term_batch_size),
            ]
            process = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT
            )
            label = f"timestamp-shard-{shard_index}"
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
                "236_merge_eval_tracin_das_noise_pairing_ablation.py",
                "--timestamp-shard-count",
                "4",
            ],
            check=True,
        )
    print("[done] full-AdamW TracIn-DAS noise-pairing ablation", flush=True)


if __name__ == "__main__":
    main()

