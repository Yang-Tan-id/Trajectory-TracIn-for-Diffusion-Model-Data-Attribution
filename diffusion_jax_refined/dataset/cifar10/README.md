# CIFAR10 Data

Expected files:

- `cifar-10-batches-py/batches.meta`
- `cifar-10-batches-py/data_batch_*`
- `hf_cifar10/train/` only for the optional legacy Diffusers wrapper
- `indices/lds-val/sub-idx-<index>.pkl` for LDS subset training

Prompted and unprompted JAX training both read `cifar-10-batches-py/`.
