# Null-gradient transfer to a different datapoint

For each of the ten existing source datapoints, this experiment deterministically
chooses a different target datapoint. The source supplies the loss gradients and
the target supplies both the evaluated image and its own prompt.

The same fixed noise direction is used for both datapoints. Source gradients are
computed for `+epsilon` and `-epsilon`; the tested source parameter tangents are
`-g+` and `-(g+ + g-)/2`. Each tangent is propagated at the target's positive and
negative noised states, then compared with the target's actual raw
null-to-next-checkpoint predicted-noise change. The full checkpoint parameter
delta JVP is retained as a control.

```bash
python -u 162_launch_null_gradient_even_cross_datapoint_4gpu.py \
  --gpus 0,1,2,3
python -u 163_compare_cross_datapoint_same_opposite.py
```

Outputs are under:

```text
x3_lds_exp_50k/null_gradient_even_cross_datapoint_next_checkpoint_10pairs/
```
