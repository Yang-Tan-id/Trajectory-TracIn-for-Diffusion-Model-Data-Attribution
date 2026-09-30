# Null-gradient cross-direction prediction

For the same ten deterministic training points as the direction-learning
probe, this experiment computes the mean null-checkpoint diffusion-loss
gradient over all 1000 timestamps with one fixed positive noise direction:

```text
g_plus = grad_theta mean_t L(theta_null; x_t(epsilon), epsilon).
```

On the opposite-direction states it predicts a one-step gradient-descent
change in predicted noise without forming a Jacobian:

```text
predicted_delta = J_null(x_t(-epsilon)) @ (-g_plus).
```

The target is the actual raw-model change from the null checkpoint at epoch 4
to the next saved checkpoint at epoch 8. Direction quality is measured by a
global cosine over all 1000 timestamps and output pixels, plus per-timestamp
cosines. The same-direction input is retained as a control. A second control,
`J_null @ (theta_next - theta_null)`, checks the local linearization itself.

```bash
python 150_verify_null_gradient_cross_direction.py
python -u 152_launch_null_gradient_cross_direction_4gpu.py --gpus 0,1,2,3
```

Combined output:

```text
x3_lds_exp_50k/null_gradient_cross_direction_next_checkpoint_10points/summary.json
```

After the four-GPU run, compare whether changes on the same- and opposite-noise
inputs point in the same output-space direction without rerunning any model:

```bash
python -u 153_compare_same_opposite_predicted_noise_change.py
```

This reports the direct cosine for the actual null-to-next change, the
single-point `J(-g)` prediction, and the checkpoint-parameter-delta JVP control.
