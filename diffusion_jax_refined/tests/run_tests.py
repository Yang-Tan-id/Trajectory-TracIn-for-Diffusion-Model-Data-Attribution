from __future__ import annotations

import io
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent


def main() -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = THIS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"test_{timestamp}.log"

    started = time.time()
    suite = unittest.defaultTestLoader.discover(str(THIS_DIR), pattern="test_*.py")
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    elapsed = time.time() - started

    details = stream.getvalue()
    header = (
        f"diffusion_jax_refined test run\n"
        f"timestamp: {timestamp}\n"
        f"repo_root: {REPO_ROOT}\n"
        f"elapsed_sec: {elapsed:.2f}\n"
        f"tests_run: {result.testsRun}\n"
        f"failures: {len(result.failures)}\n"
        f"errors: {len(result.errors)}\n"
        f"skipped: {len(result.skipped)}\n"
        f"success: {result.wasSuccessful()}\n"
        f"{'=' * 80}\n"
    )
    log_path.write_text(header + details)

    status = "PASS" if result.wasSuccessful() else "FAIL"
    print(f"[{status}] diffusion_jax_refined tests")
    print(f"  tests:   {result.testsRun}")
    print(f"  failed:  {len(result.failures)}")
    print(f"  errors:  {len(result.errors)}")
    print(f"  skipped: {len(result.skipped)}")
    print(f"  elapsed: {elapsed:.2f}s")
    print(f"  log:     {log_path}")

    if not result.wasSuccessful():
        print("\nFailed/error tests:")
        for test, _ in result.failures + result.errors:
            print(f"  - {test.id()}")
        print("\nOpen the log above for full tracebacks.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

