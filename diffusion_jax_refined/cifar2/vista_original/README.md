# CIFAR2 Vista Compact/Original Pipeline

This folder is a separate Vista pipeline for the original monolithic attribution
flow. It keeps the current experiment parameters, samples trajectories first,
then runs the legacy attribution engines directly and writes final scores under
`result/<experiment>/attribution_score/...`.

## Jobs

| Step | Script | Time / Nodes | Dependency | Purpose |
| --- | --- | --- | --- | --- |
| 00 | `00_train_base_models_vista.sh` | 2h / 2 nodes | none | Train the two checkpoint families: `prompted_jax` and `unprompted_jax`, both with `TRAIN_SEED=67` by default. The solo/multi labels share these checkpoints. |
| 01 | `01_train_lds_models_vista.sh` | 24h / 64 nodes | after 00 | Train LDS models for four sample modes, 16 subset seeds, `m=50`, `dataset_percentage=50`, `k=5000`. |
| 02 | `02_sample_and_original_attribution_vista.sh` | 48h / 21 nodes | after 00 | For each of the 21 query/sample tasks, run sampling, then run original monolithic `dtrak`, `das`, `end_tracin`, and `traj_tracin` attribution. |
| 03 | `03_lds_eval_report_vista.sh` | 12h / 21 nodes | after 01 and 02 | Evaluate attribution scores with LDS and write aggregate scatter/report outputs. |

## Run

```bash
bash diffusion_jax_refined/cifar2/vista_original/submit_vista_original_pipeline.sh
```

## Important Difference From `../vista`

This pipeline does not use the strict three-stage artifacts:

- It does not call `01_train_datapoint_gradient.py`.
- It does not call `02_query_gradient.py`.
- It does not call `03_score.py`.

That means EndTracIn and TrajTracIn follow the original exact monolithic path
instead of the reusable projected-feature artifact path. The tradeoff is that
train-side work is not reusable across query tasks here; it is recomputed inside
each original attribution run.
