# 3D Shapes prompted DDPM + data attribution

This folder is the 3D Shapes peer of `cifar2`, `cifar5_multi`, and `cifar10`.
It implements the complete prompted-DDPM, DAS, Trajectory TracIn, and LDS
experiment while reusing the repository's common checkpoint and artifact
contracts.

## Exact experiment contract

- Source: official 480,000-image `3dshapes.h5`, RGB 64x64.
- Balanced train set: all `4 shape x 10 object hue x 10 wall hue x 10 floor
  hue = 4,000` groups, with 5 samples without replacement per group.
- Dataset sampling seed: 42; each group is checked to contain exactly `8 scale
  x 15 orientation = 120` candidates.
- Condition: unordered 34-way multi-hot set. The vocabulary is 4 shape tokens,
  10 object-hue tokens, 10 wall-hue tokens, and 10 floor-hue tokens. Prompt
  order is erased; repeated categories are legal (for example
  `floor_hue_2,floor_hue_7,shape_cube,wall_hue_1`). Scale and orientation are
  never condition tokens.
- Base model: epsilon-prediction UNet, DDPM linear beta schedule, 1,000 training
  timesteps, seed 42, 200 epochs, cosine learning-rate decay with 10% warmup.
- Attribution universe: a seed-42 random sample of 5,000 rows from the balanced
  20k set. It is saved explicitly and is not the first 5,000 rows.
- Queries: query/initial seeds 0 through 9. Each query samples 4 unique tokens
  uniformly from the 34-token vocabulary; there is no one-token-per-category
  constraint. Sampling is deterministic DDIM (`eta=0`) and saves all 1,000
  trajectory states.
- LDS models: for each subset seed 0, 1, and 2, train 64 random 2,500-of-5,000
  subset models. Each GPU runs an independent model-training process (four GPUs
  handle 16 subset IDs each); models are not shared across GPUs. Only epoch 200
  is retained for each model.
- DAS: projected train gradients, residuals, undamped Gram, query gradients,
  and denominator caches are persisted. It uses 100 uniformly spaced timestamps
  over `0..999` and 1 Monte Carlo sample per timestamp. Lambda sweep:
  `0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000`.
- Trajectory TracIn: next-checkpoint target, raw parameters, and one shared
  train-gradient artifact. Attribution uses 10 uniformly selected trajectory
  timestamps and 10 train Monte Carlo samples per timestamp; the full 1,000-step
  DDIM trajectory is still retained. Scores are emitted as `raw`, `query_l2`,
  `train_l2`, and `query_train_l2`. The objective remains an environment setting
  so future target functions can be selected with `TRAJ_QUERY_OBJECTIVE`.
- LDS true functions: endpoint counterfactual, trajectory counterfactual,
  simple-loss counterfactual, and noise trajectory. True values are cached by
  query and LDS model group and reused by every DAS lambda and TracIn variant.

## Data location

The raw HDF5 file may be placed at:

```text
diffusion_jax_refined/dataset/3dshapes/3dshapes.h5
```

Prepare the balanced experiment dataset:

```bash
cd diffusion_jax_refined/3dshapes
python script/prepare_3dshapes.py \
  --input ../dataset/3dshapes/3dshapes.h5
```

This writes `dataset.npz`, the original 20k source indices, metadata, and the
fixed random 5k attribution indices under
`diffusion_jax_refined/dataset/3dshapes/20000/`.

## End-to-end runner

Print the complete command plan without launching expensive jobs:

```bash
python script/run_3dshapes_experiment.py \
  --input ../dataset/3dshapes/3dshapes.h5
```

Execute it:

```bash
python script/run_3dshapes_experiment.py --execute \
  --input ../dataset/3dshapes/3dshapes.h5
```

Every phase has a `--skip-*` flag, so cluster runs can be resumed at phase
boundaries. Outputs live below `3dshapes/result/<EXPERIMENT_TAG>/`. On one node,
the LDS phase uses `lds/run_training_multi_gpu.py` to launch one independent
training process per GPU.

## Three-node LDS training

The base model is one job and must finish first. Submit it, then submit a
three-element LDS job array with an `afterok` dependency:

```bash
base_job=$(sbatch --parsable slurm/train_base_single_gpu.sbatch)
sbatch --dependency="afterok:${base_job}" slurm/train_lds_3node_array.sbatch
```

Set the cluster-specific partition/account on the `sbatch` command line when
needed. Array tasks 0, 1, and 2 run on three nodes and own LDS subset seeds 0,
1, and 2 respectively. Inside each node, GPUs 0--3 independently train the
round-robin subsets `gpu, gpu+4, ..., gpu+60`. Deterministic subset files are
prepared once before workers launch, so parallel workers never regenerate or
race on subset definitions.

Useful overrides include `JAX_BATCH_SIZE`, `DAS_PROJ_DIM`,
`TRAJ_TRACIN_PROJ_DIM`, `DAS_NUM_MC_NOISE`, `TRAJ_TRAIN_MC_SAMPLES`, and
`TRAJ_QUERY_OBJECTIVE`.

## TACC RTX-small training

Stampede3's `rtx-small` nodes expose two GPUs. The TACC launcher therefore
runs two independent LDS workers on GPU 0 and GPU 1. Because the account limit
permits only two submitted jobs and one concurrent `rtx-small` node, submit
the base and LDS seed jobs from a TACC login node one stage at a time. TACC
compute nodes are not valid submission hosts. Each seed receives its own
48-hour wall-clock allocation:

```bash
cd diffusion_jax_refined/3dshapes/tacc/rtx_small
bash submit_training_pipeline.sh
```

Set `TACC_ACCOUNT` when an explicit allocation is required by `sbatch`. The
scripts default to the same conda environment used by the CIFAR5 RTX-small
jobs; set `ENV_SETUP=/path/to/setup.sh` to use another environment setup.

After LDS training, submit the query-independent DAS train-gradient stage from
a TACC login node. It uses the fixed 5k attribution subset, 100 uniformly
spaced timestamps, one MC sample per timestamp, and projection dimension 4096:

```bash
sbatch diffusion_jax_refined/3dshapes/tacc/rtx_small/run_das_train_rtx_small.sh
```

After the DAS train artifact and all ten DDIM query trajectories exist, run
the ten DAS query gradients followed by the complete 16-lambda score sweep:

```bash
sbatch diffusion_jax_refined/3dshapes/tacc/rtx_small/run_das_query_score_rtx_small.sh
```

GPU 0 and GPU 1 first own disjoint query-gradient jobs. Scoring then batches
all ten queries and splits the lambda values across the GPUs, loading the
shared train/Gram artifact once per worker. Existing query-gradient artifacts
are skipped when the launcher is resumed.

After all three LDS subset-seed folders and the ten full DDIM trajectories are
available, compute the four reusable true-f targets before running any LDS
score correlation:

```bash
sbatch diffusion_jax_refined/3dshapes/tacc/rtx_small/run_lds_true_f_rtx_small.sh
```

The two RTX GPUs run disjoint model shards (96 of the 192 LDS models each).
For every query/model pair the checkpoint is restored once, and endpoint and
trajectory counterfactual targets share one generated DDIM trajectory. Existing
target JSON caches are skipped, so resubmitting safely resumes missing work.

Once the true-f job completes, compute all four Traj TracIn LDS variants from
the caches on a CPU node (no LDS checkpoints or trajectories are recomputed):

```bash
sbatch diffusion_jax_refined/3dshapes/tacc/rtx_small/run_traj_tracin_lds_cached.sh
```

Two additional aligned rescoring schemes reuse the same train/query gradients:
`constant_lr_uniform` replaces every checkpoint learning rate by one (with
uniform `1/K` timestamp averaging), while `cosine_lr_ddim_step_squared` keeps
the original cosine-warmup checkpoint learning rate and distributes it over
timestamps using normalized squared DDIM step coefficients. Run both score
schemes and their cached LDS evaluations on the two RTX GPUs with:

```bash
sbatch diffusion_jax_refined/3dshapes/tacc/rtx_small/run_traj_tracin_weighted_scores_lds.sh
```

To compare against timestep-aligned Traj TracIn, build one checkpoint-shared
train gradient by averaging 100 evenly spaced diffusion timestamps with one
independent noise draw per timestamp. This stores 50 train terms (one per
checkpoint) and runs the two checkpoint shards on the two RTX GPUs:

```bash
sbatch diffusion_jax_refined/3dshapes/tacc/rtx_small/run_traj_tracin_checkpoint_shared_100x1_train_rtx_small.sh
```

After that artifact is complete, compute 100 query trajectory gradients per
checkpoint transition and score them against the shared train gradient. The
scorer aggregates the 100 query terms before the train matrix multiplication,
so it writes the raw and three normalized score variants without repeating
the large multiplication 100 times:

```bash
sbatch diffusion_jax_refined/3dshapes/tacc/rtx_small/run_traj_tracin_checkpoint_shared_100x1_query_score_rtx_small.sh
```

For an exact timestamp-aligned `100 timestamps x 1 MC` comparison with retained
train gradients, the H100 launcher requests four nodes and four GPUs per node.
It first caches the ten query-gradient artifacts, then shards the 49 usable
training checkpoints across all 16 GPUs. Each checkpoint stores 100 timestamp
gradients for the fixed random 5k attribution subset:

```bash
sbatch diffusion_jax_refined/3dshapes/tacc/h100/run_traj_tracin_aligned100x1_stream_h100.sh
```

The launcher is restartable: completed query artifacts and checkpoint parts are
skipped. It keeps 49 independent checkpoint parts under the namespace
`traj_tracin_aligned100x1_saved`, totaling roughly 374 GiB. It intentionally
does not merge them into a second 374-GiB artifact. Once all parts exist, the
same 16 workers read their assigned checkpoint parts, compute all ten queries'
raw/query-normalized/train-normalized/both-normalized scores, and merge only
the small score shards.

To run the timestamp-aligned DAS comparison with 10 uniformly spaced
timestamps and one stored gradient per timestamp. Each datapoint forms the
mean loss over 10 independent Monte Carlo noises and differentiates that mean
once; it does not run or save ten separate backwards. The Gram is formed from
these averaged-loss gradients. First build its isolated train/Gram artifact,
then compute all ten query artifacts and the 16-lambda score sweep:

```bash
sbatch diffusion_jax_refined/3dshapes/tacc/rtx_small/run_das_aligned10x10_train_rtx_small.sh
sbatch diffusion_jax_refined/3dshapes/tacc/rtx_small/run_das_aligned10x10_query_score_rtx_small.sh
```

Submit the second job only after the first completes successfully. These jobs
write under `das_aligned10x10`, leaving the existing DAS 100x1 artifacts and
scores unchanged. After scoring, reuse the existing true-f cache for LDS:

```bash
sbatch diffusion_jax_refined/3dshapes/tacc/rtx_small/run_das_aligned10x10_lds_cached.sh
```

### Predicted-noise JVP L2-squared score

This score reuses the original Traj TracIn train artifact (50 checkpoints,
10 timestamps, and one raw projected gradient of the mean 10-MC denoising
loss per datapoint/term). For each query term it differentiates one Gaussian
scalar probe of the vector predicted-noise output, immediately computes and
squares its dot products with all 5,000 saved train gradients, and retains
only the final scores. It produces the original raw, query-L2, train-L2, and
query+train-L2 variants, using learning-rate-squared term weights because the
directional output change is squared. The transient query-gradient artifacts
are deleted after score materialization.

Run the complete one-node/two-GPU RTX-small pipeline, including cached LDS evaluation:

```bash
sbatch diffusion_jax_refined/3dshapes/tacc/rtx_small/run_predicted_noise_jvp_l2_squared_rtx_small.sh
```

Permanent scores are written below `traj_tracin_predicted_noise_jvp_l2_squared`
in `score`, `score_query_normalized`, `score_train_l2_normalized`, and
`score_query_train_l2_normalized`.

### Normalized expected-Jacobian times expected-residual score

For each checkpoint, timestamp, and attribution point, this variant forms the
full-vector 10-MC expected predicted-noise residual and the expected
predicted-noise Jacobian. The same four probes produce and retain two train
features without a second train pass:

- `train_features = P(E[J]^T E[r]) / estimated_frobenius_norm(E[PJ])`;
- `train_features_v_l2_normalized = mean_l((v_l^T E[r]/sqrt(D)) * unit(P E[J]^T v_l/sqrt(D)))`.

The first normalizes the complete expected Jacobian before residual
contraction. The second normalizes each randomly probed gradient and also
projects the residual with that probe. The full residual remains a vector in
both definitions. The method produces these two fixed train normalizations,
not four reconstructable post-hoc variants.

```bash
sbatch -p rtx-small \
  --export=ALL,EXPERIMENT_TAG=experiment1,TRAIN_SEED=42 \
  diffusion_jax_refined/3dshapes/tacc/rtx_small/run_traj_tracin_expected_residual_jacobian_train_rtx_small.sh
```

The two GPUs split the 50 checkpoints, and all checkpoint parts are retained
for restart and direct score streaming. No duplicate monolithic merged copy is
written. Each of the two train features is crossed with both the original
next-checkpoint-noise-MSE target and the new vector predicted-noise target,
using both raw and per-term L2-normalized query gradients. This gives eight
scores per query (and 320 cached LDS evaluations over 10 queries and four true-f
targets). Run the complete RTX-small pipeline with:

```bash
sbatch -p rtx-small \
  --export=ALL,EXPERIMENT_TAG=experiment1,TRAIN_SEED=42 \
  diffusion_jax_refined/3dshapes/tacc/rtx_small/run_expected_residual_jacobian_pipeline_rtx_small.sh
```

### Noise-specific normalized 10x10 DAS

This variant keeps the predicted-noise gradient and residual separate. At each
of 10 timestamps it evaluates 10 noise-specific terms, jointly normalizes the
10 gradients and the 10 residuals, then averages each into one timestamp term.
It writes to the independent `das_mc_normalized10x10` namespace.

```bash
sbatch diffusion_jax_refined/3dshapes/tacc/rtx_small/run_das_mc_normalized10x10_train_rtx_small.sh
sbatch diffusion_jax_refined/3dshapes/tacc/rtx_small/run_das_mc_normalized10x10_query_score_rtx_small.sh
sbatch diffusion_jax_refined/3dshapes/tacc/rtx_small/run_das_mc_normalized10x10_lds_cached.sh
```

Submit each stage only after the previous stage completes successfully.
