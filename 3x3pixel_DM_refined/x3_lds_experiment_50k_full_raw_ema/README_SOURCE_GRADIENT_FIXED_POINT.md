# Source-gradient prediction of one fixed target response

This experiment removes the exact saved parameter delta from the predictor.
For each existing fresh-SGD branch, it reconstructs the update using only the
baseline model, source datapoint loss gradient, clipping threshold, and learning
rate:

```text
Delta-theta-hat = -learning-rate * clip-scale * source-gradient.
```

It then predicts the complete 27-dimensional output change at each fixed target
endpoint, pollution direction, timestep, and prompt using one high-precision
JVP. The saved updated model is used only to construct the finite ground truth.

Two source-update representations are tested:

- the unquantized mathematical SGD step;
- the same step after float32 parameter-storage rounding.

The exact saved parameter delta is retained only as a query-JVP control.

Run:

```bash
python -u 191_launch_source_gradient_fixed_point_4gpu.py \
  --gpus 0,1,2,3 \
  --batch-size 512
```

If necessary, reduce the batch size to 256. The log and summary are:

```text
x3_lds_exp_50k/source_gradient_fixed_point_jvp_20dir_100t_float64/logs/source_gradient_fixed_point_4gpu.log
x3_lds_exp_50k/source_gradient_fixed_point_jvp_20dir_100t_float64/summary.json
```
