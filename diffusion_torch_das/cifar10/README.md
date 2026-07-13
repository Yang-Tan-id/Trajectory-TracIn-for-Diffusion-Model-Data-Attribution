# Torch DAS CIFAR10

CIFAR10 keeps all 10 CIFAR-10 classes.

No download is needed when the repo has:

```text
../diffusion_jax_refined/dataset/cifar10/cifar-10-batches-py
```

Run from `diffusion_torch_das/`:

```bash
bash cifar10/scripts/01_train.sh
bash cifar10/scripts/02_gen.sh
bash cifar10/scripts/03_grad_train.sh
bash cifar10/scripts/04_grad_query.sh
bash cifar10/scripts/05_score.sh
```

Override defaults with environment variables:

```bash
DEVICE=auto EPOCHS=1 BATCH_SIZE=8 bash cifar10/scripts/01_train.sh
```
