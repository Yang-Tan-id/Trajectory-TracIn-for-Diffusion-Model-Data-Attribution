# Legacy JAX Snapshot

This directory is a historical snapshot of the original JAX diffusion
experiments.

Historical conda environment: `experiment_dm`.

New prompted and unprompted runs should use `diffusion_jax_refined/`. The
maintained unprompted path uses a JAX UNet with `class_cond=False`, a separate
sampling step, and the same five attribution engines as prompted runs.
