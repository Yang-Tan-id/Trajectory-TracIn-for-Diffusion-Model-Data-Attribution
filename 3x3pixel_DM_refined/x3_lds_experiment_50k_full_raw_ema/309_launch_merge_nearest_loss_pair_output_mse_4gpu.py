"""Run direct nearest-loss pair output comparison on four GPUs and merge."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import numpy as np

from exp_config import LDS_DIR, LOG_DIR, QUERY_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--sources", default="ema,raw")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    if not gpus:
        raise ValueError("no GPUs supplied")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "nearest_simple_loss_pair_output_mse_4gpu.log"
    active = []
    with log_path.open("a", buffering=1) as stream:
        for shard, gpu in enumerate(gpus):
            command = [
                sys.executable,
                "-u",
                "308_collect_nearest_loss_pair_output_mse.py",
                "--gpu",
                str(gpu),
                "--query-shard-index",
                str(shard),
                "--query-shard-count",
                str(len(gpus)),
                "--sources",
                args.sources,
            ]
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            active.append((shard, gpu, process))
            print(f"[launcher] shard={shard} gpu={gpu} pid={process.pid}", flush=True)
        while active:
            for item in list(active):
                shard, gpu, process = item
                code = process.poll()
                if code is None:
                    continue
                active.remove(item)
                print(f"[launcher] shard={shard} gpu={gpu} code={code}", flush=True)
                if code:
                    raise SystemExit(code)
            if active:
                time.sleep(2)

    queries = json.loads((QUERY_DIR / "manifest.json").read_text())
    shard_dir = LDS_DIR / "nearest_simple_loss_pair_output_mse_shards"
    records = [
        json.loads((shard_dir / f"q{query_id:02d}.json").read_text())
        for query_id in range(len(queries))
    ]
    sources = [value.strip() for value in args.sources.split(",") if value.strip()]
    summary = {}
    lines = [
        "NEAREST SIMPLE-LOSS PAIR: DIRECT OUTPUT COMPARISON",
        "Pair selected independently per query/source among 192 subset models.",
        "",
    ]
    for source in sources:
        source_records = [record["sources"][source] for record in records]
        stats = {}
        for key in ("loss_gap", "pairwise_endpoint_mse", "pairwise_trajectory_mse"):
            values = np.asarray([record[key] for record in source_records])
            stats[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        summary[source] = stats
        lines.extend(
            [
                f"SOURCE: {source.upper()}",
                "query pair      loss-gap endpoint-pair-MSE trajectory-pair-MSE",
            ]
        )
        for record, values in zip(records, source_records):
            lines.append(
                f"q{record['query_id']:02d}   {values['subset_a']:03d}/{values['subset_b']:03d} "
                f"{values['loss_gap']:11.3e} {values['pairwise_endpoint_mse']:17.6e} "
                f"{values['pairwise_trajectory_mse']:19.6e}"
            )
        lines.extend(
            [
                "MEAN +/- STD",
                f"  loss gap       {stats['loss_gap']['mean']:.6e} +/- {stats['loss_gap']['std']:.3e}",
                f"  endpoint MSE   {stats['pairwise_endpoint_mse']['mean']:.6e} +/- {stats['pairwise_endpoint_mse']['std']:.3e}",
                f"  trajectory MSE {stats['pairwise_trajectory_mse']['mean']:.6e} +/- {stats['pairwise_trajectory_mse']['std']:.3e}",
                "",
            ]
        )

    output = {"summary": summary, "queries": records}
    json_path = LDS_DIR / "nearest_simple_loss_pair_output_mse.json"
    txt_path = LDS_DIR / "nearest_simple_loss_pair_output_mse.txt"
    json_path.write_text(json.dumps(output, indent=2) + "\n")
    txt_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[saved] {json_path}")
    print(f"[saved] {txt_path}")


if __name__ == "__main__":
    main()
