# Four datapoints/directions: output-vector composition

Each sequence performs four restored-AdamW updates. Update `b` uses a different
source datapoint, that datapoint's own prompt and fixed noise direction, and one
250-timestep block. Ten cyclic four-datapoint sequences are evaluated.

For every fixed target endpoint, pollution direction, timestep, and prompt, the
experiment predicts each update's complete 27-dimensional output response. It
then compares:

- adding the four response vectors before taking the norm;
- summing four squared response norms and discarding cross terms.

Run:

```bash
python -u 195_launch_multisource4_fixed_point_4gpu.py \
  --gpus 0,1,2,3 \
  --batch-size 256
```

Outputs:

```text
x3_lds_exp_50k/multisource4_restored_adamw_fixed_point_20dir_100t_float64/logs/multisource4_fixed_point_4gpu.log
x3_lds_exp_50k/multisource4_restored_adamw_fixed_point_20dir_100t_float64/summary.json
```
