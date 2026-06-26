# Unprompted Diffusers Track

This is the separate no-prompt / unconditional `diffusers` training path.

It uses:

```text
training/train_diffusers_unconditional.py
```

and writes models under:

```text
<dataset>/result/<experiment>/model/<algorithm>/unprompted/
```

Example for CIFAR2:

```bash
cd "diffusion das refine/cifar2"

EXPERIMENT_TAG=experiment1 CUDA_VISIBLE_DEVICES=0 ALGORITHMS="das" bash scripts/00_train_unprompted.sh 0 0
```

This trains subset `0` for the default seeds `0, 1, 2`, producing:

```text
result/experiment1/model/das/unprompted/ddpm-sub-0-0/
result/experiment1/model/das/unprompted/ddpm-sub-0-1/
result/experiment1/model/das/unprompted/ddpm-sub-0-2/
```

To run one algorithm from its own folder:

```bash
cd "diffusion das refine/cifar2/data_attribution/traj_tracin"
EXPERIMENT_TAG=experiment1 CUDA_VISIBLE_DEVICES=0 bash script.sh train_unprompted 0 0
```

## Attribution And Eval

Unprompted data attribution and eval use a diffusers-compatible backend:

```text
common/unprompted_diffusers_attribution.py
common/unprompted_counterfactual_eval.py
common/unprompted_lds_eval.py
```

Run attribution:

```bash
EXPERIMENT_TAG=experiment1 CUDA_VISIBLE_DEVICES=0 ALGORITHMS="das" bash scripts/01_data_attribution_unprompted.sh
```

Outputs:

```text
result/experiment1/attribution_score/das_unprompted/
  scores.npy
  score_indices.npy
  score_indices.json
  result_topk.json
  run_config.json
```

Run counterfactual removal-set eval:

```bash
EXPERIMENT_TAG=experiment1 ALGORITHMS="das" TOPK=5000 bash scripts/02_metric_counterfactual_unprompted.sh
```

Run lightweight LDS proxy:

```bash
EXPERIMENT_TAG=experiment1 ALGORITHMS="das" LDS_M=100 LDS_SUBSET_SIZE=5000 bash scripts/03_metric_lds_unprompted.sh
```

Or run one algorithm end-to-end from its folder:

```bash
cd "diffusion das refine/cifar2/data_attribution/das"
EXPERIMENT_TAG=experiment1 CUDA_VISIBLE_DEVICES=0 bash script.sh all_unprompted 0 0
```

The current unprompted attribution score is `-MSE(noise_pred, noise)` averaged over selected timesteps and Monte Carlo noises for each training item. The algorithm name controls the default timestep schedule:

- `end_tracin`: endpoint timestep only.
- `traj_tracin` / `journey_trak`: evenly spaced trajectory timesteps.
- `das` / `dtrak`: a small fixed timestep grid.

You can override timesteps with:

```bash
UNPROMPTED_TIMESTEPS="0,200,400,600,800,999"
UNPROMPTED_SCORE_MC=4
UNPROMPTED_SCORE_BATCH_SIZE=64
```

For trajectory-style range splitting:

```bash
EXPERIMENT_TAG=experiment1 CUDA_VISIBLE_DEVICES=0 ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="1-2500,2501-5000" bash scripts/01_data_attribution_unprompted.sh
EXPERIMENT_TAG=experiment1 ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="1-2500,2501-5000" bash scripts/02_metric_counterfactual_unprompted.sh
EXPERIMENT_TAG=experiment1 ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="1-2500,2501-5000" bash scripts/03_metric_lds_unprompted.sh
```

The unprompted LDS script is a lightweight proxy over attribution score subset sums. Full LDS with retrained diffusers subset targets can be layered onto the same score files.
