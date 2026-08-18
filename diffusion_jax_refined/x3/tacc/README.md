# TACC launchers

Dataset-local TACC launchers live by machine family:

- `h100/script_0.sh`
- `h100/sample_for_attribution.sh`
- `h100/sample_query_gradient.sh`
- `h100/datapoint_gradients.sh`
- `vista/script_0.sh`
- `vista/sample_for_attribution.sh`
- `vista/sample_query_gradient.sh`
- `vista/datapoint_gradients.sh`

These wrappers live inside this dataset folder and delegate to this dataset's refined `scripts/script_0.sh`. The mode names are launcher
conventions: `prompted` means class-conditioned training, `unprompted` means
unconditional training, and `solo`/`multi` select the GPU launch style. Training
does not read `QUERY` or create prompt-specific model folders.

Examples:

```bash
TRAIN_MODE=prompted_solo TRAIN_SEED=42 GPU_IDS=0 \
  bash tacc/h100/script_0.sh

TRAIN_MODES="prompted_solo unprompted_solo" TRAIN_SEED=42 GPU_IDS=0 \
  bash tacc/h100/script_0.sh

TRAIN_MODE=prompted_multi TRAIN_SEED=42 GPU_IDS=0,1,2,3 \
  bash tacc/vista/script_0.sh

TRAIN_MODE=unprompted_multi TRAIN_SEED=42 GPU_IDS=0,1,2,3 \
  bash tacc/h100/script_0.sh

EXPERIMENT_TAG=experiment1 SAMPLE_MODEL_MODE=prompted_solo QUERY=horse SAMPLE_SEEDS=0 GPU_IDS=0 \
  bash tacc/h100/sample_for_attribution.sh

EXPERIMENT_TAG=experiment1 SAMPLE_MODEL_MODE=prompted_solo QUERY=horse SAMPLE_SEEDS=0 ALGORITHMS="das,dtrak" GPU_IDS=0 \
  bash tacc/h100/sample_query_gradient.sh

TRAIN_MODES="prompted_solo unprompted_solo" \
DATAPOINT_GRADIENT_MODES="both" \
ALGORITHMS="das,dtrak" \
GPU_IDS=0 \
  bash tacc/h100/datapoint_gradients.sh
```

The datapoint-gradient launcher writes under the model identity rather than the
query attribution folder. The score combiner expects the train-side artifact
file here:

```text
result/<experiment>/model/<train_mode>/seed_<TRAIN_SEED>_train_gradient/<algorithm>/
result/<experiment>/model/<train_mode>/seed_<TRAIN_SEED>_train_gradient/<algorithm>/train_datapoint_gradient_artifact.npz
```

The sample launcher writes trajectory samples under:

```text
result/<experiment>/sample/<adapter>/prompt_<query>/model_<SAMPLE_MODEL_MODE>__ckpt_<checkpoint>/seed_<sample_seed>/
```

The sample+query-gradient launcher writes query gradients beside the sample
seed. The score combiner expects the query-side artifact file here:

```text
result/<experiment>/sample/<adapter>/prompt_<query>/model_<SAMPLE_MODEL_MODE>__ckpt_<checkpoint>/seed_<sample_seed>_query_gradient/<algorithm>/
result/<experiment>/sample/<adapter>/prompt_<query>/model_<SAMPLE_MODEL_MODE>__ckpt_<checkpoint>/seed_<sample_seed>_query_gradient/<algorithm>/query_gradient_artifact.npz
```

`03_score.py` is pure now: it combines existing artifacts and writes
`scores.npy`, `score_indices.npy`, and `top_scores.json` under
`result/<experiment>/attribution_score/...`. It does not rerun the monolithic
legacy attribution engine; if either artifact is missing, it fails with the
missing path.

Checkpoints are grouped by dataset/experiment/model type, with the training seed
in the file name:

```text
result/<experiment>/model/prompted_jax/seed_<TRAIN_SEED>_epoch_<epoch>.ckpt
result/<experiment>/model/unprompted_jax/seed_<TRAIN_SEED>_epoch_<epoch>.ckpt
```
