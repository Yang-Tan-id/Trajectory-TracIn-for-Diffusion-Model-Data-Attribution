"""Train staged base, then dynamically schedule 192 subset models on four GPUs."""

import subprocess
import sys
import time

from staged_lds_config import *


def main():
    STAGED_LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not staged_base_checkpoint(STAGED_EPOCHS).is_file():
        log = STAGED_LOG_DIR / "train_base.log"
        print(f"[launcher] staged base on gpu {CUDA_IDS[0]} log={log}", flush=True)
        with open(log, "a", buffering=1) as stream:
            subprocess.run(
                [sys.executable, "-u", "61_train_staged_50k_worker.py", "--kind", "base", "--gpu", str(CUDA_IDS[0])],
                stdout=stream, stderr=subprocess.STDOUT, check=True,
            )
    pending = [i for i in range(STAGED_LDS_MASK_COUNT) if not staged_subset_checkpoint(i).is_file()]
    active = {}
    while pending or active:
        for gpu in CUDA_IDS[:4]:
            if gpu in active or not pending:
                continue
            mask_id = pending.pop(0)
            log = STAGED_LOG_DIR / f"train_subset_{mask_id:03d}.log"
            stream = open(log, "a", buffering=1)
            command = [
                sys.executable, "-u", "61_train_staged_50k_worker.py", "--kind", "subset",
                "--mask-id", str(mask_id), "--gpu", str(gpu),
            ]
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            active[gpu] = (process, stream, mask_id)
            print(f"[gpu {gpu}] START subset {mask_id:03d} pid={process.pid} log={log}", flush=True)
        for gpu, (process, stream, mask_id) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            del active[gpu]
            print(f"[gpu {gpu}] END subset {mask_id:03d} code={code}", flush=True)
            if code != 0:
                raise SystemExit(code)
        if active:
            time.sleep(2)
    print("[done] staged base + 192 subset models", flush=True)


if __name__ == "__main__":
    main()
