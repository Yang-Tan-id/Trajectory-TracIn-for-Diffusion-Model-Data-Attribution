# diffusion_jax_refined tests

Lightweight regression tests for the CIFAR2 attribution/LDS workflow. These tests are CPU-only and use small fake data or static checks, so they should finish well under 15 minutes.

Run all tests from the repository root:

```bash
bash diffusion_jax_refined/tests/run_all_tests.sh
```

Optional: choose a specific Python executable, for example on TACC:

```bash
PYTHON_BIN="$SCRATCH/conda-envs/trajectory-tracin/bin/python" \
  bash diffusion_jax_refined/tests/run_all_tests.sh
```

The command prints a compact report to the terminal. A detailed timestamped log is written to:

```text
diffusion_jax_refined/tests/logs/test_YYYYmmdd_HHMMSS.log
```

Detailed explanation of every test group is in:

```text
diffusion_jax_refined/tests/TESTS_OVERVIEW.txt
```

Current coverage focuses on:

- query/initial-seed result folders;
- score index range and shard alignment;
- toy attribution/LDS logic, including sign direction and shard merge equivalence;
- DAS batched-vs-unbatched score equivalence;
- endpoint MC objective batching equivalence;
- LDS Spearman/prediction sign/subset math;
- sample prompt/seed/shape validation;
- LDS subset metadata invariants;
- script-layout static checks that keep TACC launchers dataset-local and ensure
  algorithm-local scripts call dataset-local refined scripts, while training
  remains independent of sampling/query prompts;
- datapoint-gradient script checks for optional `TRAIN_MODES`,
  `DATAPOINT_GRADIENT_MODES`, comma/space-separated algorithm selection, and
  model-seed-rooted output folders;
- data-attribution sample script checks for `SAMPLE_MODEL_MODE`,
  `EXPERIMENT_TAG`, and the result-level `sample` output root;
- sample+query-gradient job checks that sampling and `02_query_gradient.py`
  run together, keyed by `SAMPLE_MODEL_MODE`, `SAMPLE_SEEDS`, and algorithm;
- LDS model-training checks for selectable `SAMPLE_MODEL_MODE`,
  `LDS_MODEL_TRAIN_SEED`, `LDS_M`/`LDS_NUM_SUBSETS`, and either
  `LDS_DATASET_PERCENTAGE` or `LDS_K` subset sizing;
- script-layout checks that old algorithm-local `run_attribution.py`,
  `run_training.py`, `run_eval.py`, and `script.sh` entrypoints are removed;
- target-specific LDS eval overwrite protection;
- per-seed LDS aggregate summary and combined scatter-grid generation;
- attribution-code contracts such as DAS squared scores, EndTracIn MC sample counts,
  method-specific three-stage facades, tunable GPU batch/chunk settings, and
  D-TRAK's batched train-feature cache that prevents PASS2 from recomputing
  train gradients.
