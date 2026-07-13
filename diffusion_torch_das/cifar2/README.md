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
