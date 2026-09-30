# Null-model same-direction learning probe

This diagnostic selects ten deterministic training points and starts a separate
model copy for each point from the current null model: the first prompted raw
checkpoint, `epoch_0004.pt`. The checkpoint AdamW moments, step, parameter
groups, learning rate, weight decay, and global-norm clipping setting are
restored.

For point `i`, one Gaussian noise tensor `epsilon_i` is sampled. All 1000
diffusion timestamps reuse that exact tensor. Timestamps 999 through 0 are
split into four consecutive batches of 250, giving four optimizer updates.

After the updates, the model is evaluated over all 1000 timestamps against the
unchanged null model using:

1. the training direction `epsilon_i`;
2. the opposite direction `-epsilon_i`;
3. 100 random directions normalized to the same norm as `epsilon_i`.

The evaluation does not compare diffusion losses. It directly measures
`epsilon_after(x_t,t,c) - epsilon_before(x_t,t,c)` and reports MSE, RMSE, mean
L2, maximum absolute change, and the signed projection onto each evaluated
noise direction. Verify and launch on four GPUs:

```bash
python 147_verify_null_same_direction_learning.py
python -u 149_launch_null_same_direction_learning_4gpu.py --gpus 0,1,2,3
```

The combined summary is:

```text
x3_lds_exp_50k/null_same_direction_predicted_noise_difference_10points/summary.json
```

Each point directory also stores the 102 exact directions and their predicted-
noise difference metrics in `direction_prediction_differences.npz`, plus its
four-step updated model.
