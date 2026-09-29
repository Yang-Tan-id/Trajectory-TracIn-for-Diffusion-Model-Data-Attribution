# Fully-unrolled trajectory-response DAS

For q00-q09, differentiate through the complete deterministic 1000-step DDIM
sampler at the final prompted EMA parameters. Four trajectory-wide Rademacher
probes estimate the Gauss--Newton trajectory response

```text
mean_t ||J_t delta_theta_i||^2.
```

The first cached state (the fixed initial noise) contributes zero parameter
gradient naturally. Query features use one global 4096-dimensional CountSketch
without L2 normalization, preserving trajectory-Jacobian magnitude. Training
features use the same projection and the normal DAS feature normalization.
Training DAS uses 100 timestamps x MC10 terms and train-gradient MC10. All
configured damping lambdas and all LDS targets are evaluated.

```bash
python -u 132_launch_unrolled_traj_das_10q_4gpu.py \
  --gpus 0,1,2,3 \
  --batch-size 128
python -u 133_eval_unrolled_traj_das_10q_lds.py
```
