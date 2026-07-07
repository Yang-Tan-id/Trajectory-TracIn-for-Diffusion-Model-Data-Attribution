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
- TACC Slurm script static safety checks;
- target-specific LDS eval overwrite protection;
- per-seed LDS aggregate summary and combined scatter-grid generation;
- attribution-code contracts such as DAS squared scores and EndTracIn MC sample counts.
