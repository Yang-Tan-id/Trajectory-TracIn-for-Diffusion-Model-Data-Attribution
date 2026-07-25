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
