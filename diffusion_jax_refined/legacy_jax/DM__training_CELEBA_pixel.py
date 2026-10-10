"""CelebA pixel-DDPM adapter.

Prepared CelebA images use the repository's generic ``dataset.npz`` contract.
The training entrypoint runs this adapter unconditionally, so the single
placeholder label is never supplied to the UNet.
"""

from DM__training_CIFAR5_MULTI_pixel import *  # noqa: F401,F403
