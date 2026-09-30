# Null-gradient odd/even direction experiment

This experiment tests whether the direction-odd part of a single-point diffusion
loss gradient explains the opposite behavior observed between `+epsilon` and
`-epsilon` noised inputs.

For each of the same ten deterministic training points, at the raw null
checkpoint, it computes

```text
g+    = grad L(x_t(+epsilon), +epsilon)
g-    = grad L(x_t(-epsilon), -epsilon)
g_even = (g+ + g-) / 2
g_odd  = (g+ - g-) / 2
```

The four descent tangents `-g+`, `-g-`, `-g_even`, and `-g_odd` are propagated
with JVPs at both noised inputs. Their predicted-noise changes are compared with
the actual raw null-to-next-checkpoint change. A JVP using the full checkpoint
parameter delta remains the control. Loss gradients use the datapoint's original
prompt; the default evaluation uses a deterministic different prompt.

Run on the school server from this experiment directory:

```bash
python -u 150_verify_null_gradient_cross_direction.py
python -u 159_launch_null_gradient_odd_even_4gpu.py \
  --gpus 0,1,2,3 \
  --evaluation-prompt random
python -u 160_compare_null_gradient_odd_even_same_opposite.py \
  --evaluation-prompt random
```

Primary outputs:

```text
x3_lds_exp_50k/null_gradient_odd_even_random_prompt_next_checkpoint_10points/summary.json
x3_lds_exp_50k/null_gradient_odd_even_random_prompt_next_checkpoint_10points/odd_even_same_opposite_consistency.json
```
