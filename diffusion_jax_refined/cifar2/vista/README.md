# CIFAR2 Vista Pipeline

Submit from the repository root on Vista:

```bash
bash diffusion_jax_refined/cifar2/vista/submit_vista_pipeline.sh
```

The pipeline submits six Slurm jobs:

```text
00_train_four_models_vista.sh          2 nodes   2h   train prompted/unprompted base checkpoints, seed 67
01_train_lds_models_vista.sh          64 nodes  24h  4 modes x 16 LDS subset seeds, M=50, 50%
02_train_datapoint_gradients_vista.sh  8 nodes  48h  2 checkpoint families x 4 algorithms
03_sample_query_gradients_vista.sh    21 nodes  48h  21 sample/query-gradient tasks
04_score_vista.sh                     21 nodes  24h  21 pure score-combine tasks
05_lds_eval_report_vista.sh           21 nodes  12h  LDS eval, scatter aggregates, report
```

Dependencies:

```text
00 train prompted/unprompted base checkpoints
  -> 01 train LDS models
  -> 02 train datapoint gradients
  -> 03 sample + query gradients
01 + 02 + 03 -> 04 pure score combine -> 05 LDS eval + report
```

`04_score_vista.sh` does not rerun the legacy attribution engine. It calls each
algorithm's `03_score.py`, which only reads train-side artifacts from
`02_train_datapoint_gradients_vista.sh` and query-side artifacts from
`03_sample_query_gradients_vista.sh`, then writes `scores.npy` under
`attribution_score`.

Default experiment and model seed:

```text
EXPERIMENT_TAG=experiment_67
TRAIN_SEED=67
```

Training parameters:

```text
prompted_solo and prompted_multi share the same prompted_jax checkpoint.
unprompted_solo and unprompted_multi share the same unprompted_jax checkpoint.
solo/multi only changes launch/sample grouping, not model architecture.
prompted vs unprompted differs by conditioning: class_cond=True vs class_cond=False.
Both prompted and unprompted default to JAX_LEARNING_RATE=2e-4 unless overridden.
```

The 21 query tasks are:

```text
unprompted_solo  : seeds 24,48,96
unprompted_multi : seeds 24,48,96
prompted_solo    : horse and automobile, seeds 24,48,96
prompted_multi   : horse, automobile, and horse,automobile, seeds 24,48,96
```

Per-task logs are written under:

```text
result/experiment_67/vista_logs/<job_name>/<slurm_job_id>/*.log
```

Final LDS eval report:

```text
result/experiment_67/eval/reports/lds_eval_report.md
result/experiment_67/eval/reports/lds_eval_report.csv
```

Useful checks:

```bash
squeue -u "$USER"
sacct -j <job_ids> --format=JobID,JobName%32,State,Elapsed,Timelimit,NodeList%24
find diffusion_jax_refined/cifar2/result/experiment_67/vista_logs -type f -name '*.log' | sort
find diffusion_jax_refined/cifar2/result/experiment_67/eval -name 'per_seed_scatter_grid.svg' | sort
```
