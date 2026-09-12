"""3D Shapes pixel-DDPM adapter.

The generated 3D Shapes dataset uses the same generic ``dataset.npz`` contract
as CIFAR5 multi: RGB images plus an unordered multi-hot condition vector.  The
model, optimizer, DDPM schedule, checkpoint format, and attribution API are
therefore intentionally shared with the battle-tested CIFAR5 implementation.
"""

from DM__training_CIFAR5_MULTI_pixel import *  # noqa: F401,F403

