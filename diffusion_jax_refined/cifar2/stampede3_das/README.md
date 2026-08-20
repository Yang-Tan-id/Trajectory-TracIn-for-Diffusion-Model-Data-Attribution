# Stampede3 H100 DAS-only pipeline

This folder runs the CIFAR2 DAS-only workflow on Stampede3 `h100`.

Stampede3 `h100` has 4 H100 GPUs per node and a 4-node/user limit. The
attribution stage is split into three explicit 4-node chunk jobs so each chunk
has an independent job id and can be inspected or retried cleanly.

Jobs:

| Step | Script | Resources | Dependency | Work |
| --- | --- | --- | --- | --- |
| 00 | `00_train_base_models_stampede3.sh` | 1 H100 node, 2h | none | Train `prompted_solo` on GPUs 0,1 and `unprompted_solo` on GPUs 2,3. |
| 01 | `01_train_lds_models_stampede3.sh` | 4 H100 nodes, 24h | after 00 | Train LDS models for `prompted_solo` and `unprompted_solo`, seeds 0-7, `M=64`, `50%`. |
| 02 smoke | `02_smoke_das_attribution_stampede3.sh` | 1 H100 node, 30m | after 00 | One sample/query/lambda smoke test. Verifies `scores.npy` before large attribution jobs. |
| 02 smoke RTX | `02_smoke_das_attribution_rtx_small.sh` | 1 `rtx-small` node, 30m | after 00 | Same one sample/query/lambda smoke test on the smaller RTX queue. |
| 02a/02b/02c | `02a_das_attribution_stampede3.sh`, `02b_das_attribution_stampede3.sh`, `02c_das_attribution_stampede3.sh` | each 4 H100 nodes, 48h | after 00 if `TRAIN_JOB_ID` is set | 48 query/sample tasks split into 3 chunks of 16 GPUs. Runs DAS with 21 lambda sweep values. |
| 03 | `03_das_lds_eval_report_stampede3.sh` | 4 H100 nodes, 24h | after 01 and 02a/02b/02c | LDS eval for two targets and DAS lambda variants. Each GPU handles three query specs. |

Submit in stages from the repository root on Stampede3. The default stage only
submits base training and LDS training.

```bash
STAMPEDE3_ACCOUNT=IRI26004 EXPERIMENT_TAG=experiment_67 TRAIN_SEED=67 \
  bash diffusion_jax_refined/cifar2/stampede3_das/submit_stampede3_das_pipeline.sh
```

After the base training job finishes and submit slots are free:

```bash
STAMPEDE3_ACCOUNT=IRI26004 EXPERIMENT_TAG=experiment_67 TRAIN_SEED=67 \
  sbatch -A IRI26004 --parsable diffusion_jax_refined/cifar2/stampede3_das/02_smoke_das_attribution_stampede3.sh
```

Or use the smaller RTX queue:

```bash
STAMPEDE3_ACCOUNT=IRI26004 EXPERIMENT_TAG=experiment_67 TRAIN_SEED=67 \
  sbatch -A IRI26004 --parsable diffusion_jax_refined/cifar2/stampede3_das/02_smoke_das_attribution_rtx_small.sh
```

After the smoke job completes with `COMPLETED 0:0`:

```bash
STAMPEDE3_ACCOUNT=IRI26004 EXPERIMENT_TAG=experiment_67 TRAIN_SEED=67 \
STAMPEDE3_SUBMIT_STAGE=attr \
  bash diffusion_jax_refined/cifar2/stampede3_das/submit_stampede3_das_pipeline.sh
```

After LDS and attribution finish:

```bash
STAMPEDE3_ACCOUNT=IRI26004 EXPERIMENT_TAG=experiment_67 TRAIN_SEED=67 \
STAMPEDE3_SUBMIT_STAGE=eval LDS_JOB_ID=<01_job_id> ATTR_JOB_IDS=<02a_job_id>:<02b_job_id>:<02c_job_id> \
  bash diffusion_jax_refined/cifar2/stampede3_das/submit_stampede3_das_pipeline.sh
```

DAS sweep lambdas:

```text
0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10 20 50 100 200 500 1000 2000 5000 10000 20000 50000
```

Scores are stored under lambda-specific folders, for example:

```text
result/experiment_67/attribution_score/prompted_solo/train_seed_67/query_horse/initial_seed_0/das/lambda_2/scores.npy
result/experiment_67/attribution_score/unprompted_solo/train_seed_67/unprompted/initial_seed_0/das_unprompted/lambda_2/scores.npy
```

## Raw Traj-TracIn next-checkpoint run

These scripts are independent `sbatch` jobs. Submit them manually in this order
after the base `prompted_solo` / `unprompted_solo` models exist:

```bash
sbatch -A IRI26004 diffusion_jax_refined/cifar2/stampede3_das/10_train_lds_25pct_h100_stampede3.sh
sbatch -A IRI26004 diffusion_jax_refined/cifar2/stampede3_das/11_traj_tracin_raw_nextckpt_h100_stampede3.sh
sbatch -A IRI26004 diffusion_jax_refined/cifar2/stampede3_das/12_eval_traj_raw_nextckpt_lds25_targets_stampede3.sh
```

Defaults:

```text
LDS: M=64, 25%, subset seeds 0-7, h100 4 nodes / 16 GPUs, 12h
Traj-TracIn: raw TrainState.params, trajectory_next_checkpoint_noise_mse target, h100 4 nodes / 16 GPUs, 24h
Eval targets: endpoint_contarfactual, traj_contarfactual, simple_loss, trajectory_state_mse
Eval algorithm folder: traj_tracin_trajectory_next_checkpoint_noise_mse_raw
```

Override `EXPERIMENT_TAG`, `TRAIN_SEED`, `LDS_SEEDS`, `PROMPTED_INITIAL_SEEDS`,
or `UNPROMPTED_INITIAL_SEEDS` in the `sbatch` environment if needed.
