# Torch DAS CIFAR2

CIFAR2 means CIFAR-10 filtered to:

- automobile
- horse

No download is needed when the repo has:

```text
../diffusion_jax_refined/dataset/cifar2/cifar-10-batches-py
```

Run from `diffusion_torch_das/`:

```bash
bash cifar2/scripts/01_train.sh
bash cifar2/scripts/02_gen.sh
bash cifar2/scripts/03_grad_train.sh
bash cifar2/scripts/04_grad_query.sh
bash cifar2/scripts/05_score.sh
```

Override defaults with environment variables, for example:

```bash
DEVICE=auto EPOCHS=1 BATCH_SIZE=8 bash cifar2/scripts/01_train.sh
```

## LDS Eval

The LDS flow follows the original DAS clone but fixes the hard-coded paths and
script typos.

```bash
# Create subset index files.
NUM_SUBSETS=64 SUBSET_SIZE=5000 bash cifar2/scripts/06_lds_make_subsets.sh

# Train subset models. Use START/END to shard across machines or GPUs.
START=0 END=63 SEEDS=0,1,2 DEVICE=cuda EPOCHS=200 BATCH_SIZE=128 \
  bash cifar2/scripts/07_lds_train.sh

# Evaluate subset models on generated queries.
START=0 END=63 SEEDS=0,1,2 EVAL_SEEDS=0 DEVICE=cuda \
  bash cifar2/scripts/08_lds_eval.sh

# LDS needs query x train scores, not query-averaged scores.
TRAIN_SHAPE=10000,4096 QUERY_SHAPE=1000,4096 bash cifar2/scripts/09_lds_score_matrix.sh

# Compute Spearman LDS.
NUM_SUBSETS=64 SEEDS=0,1,2 EVAL_SEEDS=0 bash cifar2/scripts/10_lds_score.sh
```

For a tiny sanity check:

```bash
NUM_SUBSETS=2 SUBSET_SIZE=8 bash cifar2/scripts/06_lds_make_subsets.sh
START=0 END=1 SEEDS=0 DEVICE=auto EPOCHS=1 BATCH_SIZE=4 bash cifar2/scripts/07_lds_train.sh
START=0 END=1 SEEDS=0 EVAL_SEEDS=0 DEVICE=auto TIMESTEPS=4 MAX_SAMPLES=4 bash cifar2/scripts/08_lds_eval.sh
```
