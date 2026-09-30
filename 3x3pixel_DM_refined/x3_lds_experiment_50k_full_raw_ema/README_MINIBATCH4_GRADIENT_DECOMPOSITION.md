# One-minibatch per-datapoint gradient decomposition

This validation averages four distinct datapoint losses into one minibatch loss
and performs exactly one update. Each datapoint uses its own fixed noise
direction and one disjoint block of 250 diffusion timestamps.

It tests two updates from the same null checkpoint:

- fresh SGD: the four per-datapoint parameter contributions must sum to the
  batch update;
- restored AdamW: every datapoint uses the same realized batch clipping scalar
  and the same realized Adam second-moment denominator. Historical momentum and
  weight decay are retained as a separate optimizer-history baseline rather
  than assigned to the four current datapoints.

For every fixed target point, the four signed 27-dimensional predicted-noise
responses are added before taking the norm. A termwise-square control is also
saved to measure the missing cross terms.

Run on the school server:

```bash
python -u 197_launch_minibatch4_gradient_decomposition_4gpu.py \
  --gpus 0,1,2,3 \
  --batch-size 512
```

Outputs:

```text
x3_lds_exp_50k/minibatch4_gradient_decomposition_20dir_100t_float64/logs/minibatch4_gradient_decomposition_4gpu.log
x3_lds_exp_50k/minibatch4_gradient_decomposition_20dir_100t_float64/summary.json
```
