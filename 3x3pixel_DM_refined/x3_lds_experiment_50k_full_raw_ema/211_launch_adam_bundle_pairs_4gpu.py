"""Launch pairwise frozen-start AdamW Bundle tangents on four GPUs."""

import argparse
import json
import subprocess
import sys
import time

from adam_bundle_pair_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--query-ids", default="0-9")
    args = parser.parse_args()
    gpus = [int(v) for v in args.gpus.split(",") if v.strip()]
    query_ids = parse_query_ids(args.query_ids)
    query_arg = ",".join(map(str, query_ids))
    with open(QUERY_DIR / "manifest.json") as handle:
        by_id = {int(r["query_id"]): r for r in json.load(handle)}
    families = list(dict.fromkeys(by_id[qid]["family"] for qid in query_ids))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"adam_bundle_pairs_q{query_ids[0]:02d}_q{query_ids[-1]:02d}_4gpu.log"
    for family in families:
        with open(log_path, "a", buffering=1) as stream:
            workers = []
            for shard, gpu in enumerate(gpus):
                command = [sys.executable, "-u", "209_run_adam_bundle_pair_shard.py", "--family", family, "--gpu", str(gpu), "--pair-shard-index", str(shard), "--pair-shard-count", str(len(gpus)), "--query-ids", query_arg]
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                workers.append((shard, process))
                print(f"[launcher] family={family} shard={shard} gpu={gpu} pid={process.pid}", flush=True)
            active = dict(workers)
            while active:
                for shard, process in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    del active[shard]
                    print(f"[launcher] shard={shard} code={code}", flush=True)
                    if code:
                        for other in active.values():
                            other.terminate()
                        raise SystemExit(code)
                if active:
                    time.sleep(2)
    subprocess.run([sys.executable, "-u", "210_merge_eval_adam_bundle_pairs.py", "--query-ids", query_arg], check=True)


if __name__ == "__main__":
    main()
