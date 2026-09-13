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
