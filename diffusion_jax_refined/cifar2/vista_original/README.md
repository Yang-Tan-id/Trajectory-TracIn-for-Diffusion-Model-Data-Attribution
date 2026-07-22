# CIFAR2 Vista Compact/Original Pipeline

This folder is a separate Vista pipeline for the original monolithic attribution
flow. It keeps the current experiment parameters, samples trajectories first,
then runs the legacy attribution engines directly and writes final scores under
`result/<experiment>/attribution_score/...`.

## Jobs

| Step | Script | Time / Nodes | Dependency | Purpose |
| --- | --- | --- | --- | --- |
| 00 | `00_train_base_models_vista.sh` | 2h / 2 nodes | none | Train the two checkpoint families: `prompted_jax` and `unprompted_jax`, both with `TRAIN_SEED=67` by default. The solo/multi labels share these checkpoints. |
| 01 | `01_train_lds_models_vista.sh` | 24h / 16 nodes | after 00 | Train LDS models for the two real checkpoint families, `prompted_solo` and `unprompted_solo`: 8 LDS seeds (`0..7`) x `m=64` subsets each, `dataset_percentage=50`, `k=5000`. |
| 02a-02f | `02*_attribution_priority_vista.sh` | 24h / 64 nodes each | after 00 | Six prioritized original attribution chunks. The ordered task list is TrajTracIn first (`1-2500`, `2501-5000`, `5001-7500`, `7501-10000` for prompted and unprompted), then DAS, then D-TRAK and EndTracIn. |
| 03 | `03_lds_eval_report_vista.sh` | 24h / 48 nodes | after 01 and all 02 chunks | Evaluate `simple_loss` and `noise_trajectory` LDS targets. Each query task runs 8 LDS seeds x 4 algorithms, then writes per-seed 2x4 scatter grids, all-seed scatter plots, correlations, and summary reports. |

## Run

```bash
EXPERIMENT_TAG=experiment_67 TRAIN_SEED=67 \
bash diffusion_jax_refined/cifar2/vista_original/submit_vista_original_pipeline.sh
```

The default attribution workload has 48 query samples:

- 24 unprompted initial seeds (`0..23`)
- 24 prompted initial seeds: 8 horse, 8 automobile, and 8 horse+automobile

Across seven attribution units (four TrajTracIn ranges plus DAS, D-TRAK, and
EndTracIn), this is `48 x 7 = 336` original attribution tasks split into five
64-node jobs plus one final 16-node job.

## Important Difference From `../vista`

This pipeline does not use the strict three-stage artifacts:

- It does not call `01_train_datapoint_gradient.py`.
- It does not call `02_query_gradient.py`.
- It does not call `03_score.py`.

That means EndTracIn and TrajTracIn follow the original exact monolithic path
instead of the reusable projected-feature artifact path. The tradeoff is that
train-side work is not reusable across query tasks here; it is recomputed inside
each original attribution run.
