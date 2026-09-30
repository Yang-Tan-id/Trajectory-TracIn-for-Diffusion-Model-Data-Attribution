# Two-checkpoint AdamW approximation

This diagnostic asks whether two saved checkpoints are sufficient to predict
their predicted-noise transition without replaying gradients along the moving
training model.

For each selected checkpoint pair, gradients are evaluated with parameters
fixed at:

- the start checkpoint;
- the target checkpoint;
- the average of the two clipped endpoint gradients.

Each gradient stream is fed into a shadow optimizer initialized from the start
checkpoint's AdamW state. The shadow optimizer follows the original batch
order and LR schedule, but its gradients never follow the changing shadow
parameters. This is therefore an endpoint approximation, not full training
replay.

Run four representative intervals on four GPUs:

```bash
python -u 207_launch_checkpoint_endpoint_adam_diagnostic_4gpu.py \
  --gpus 0,1,2,3 \
  --pair-indices 0,16,32,48 \
  --family prompted \
  --query-ids 0-9 \
  --query-batch-size 256
```

The report compares the exact parameter-delta JVP with frozen-start,
frozen-target, and endpoint-trapezoid AdamW approximations. Results are saved
under:

```text
x3_lds_exp_50k/checkpoint_endpoint_adam_diagnostic_10q_100t/
```

To test whether four realized events per datapoint are too noisy, repeat the
same two-checkpoint experiment with 20 independent `(t, epsilon)` samples per
datapoint loss. Start and target endpoint models share the same MC samples:

```bash
python -u 207_launch_checkpoint_endpoint_adam_diagnostic_4gpu.py \
  --gpus 0,1,2,3 \
  --pair-indices 0,16,32,48 \
  --family prompted \
  --query-ids 0-9 \
  --query-batch-size 256 \
  --loss-mc 20
```

This performs no moving-model gradient replay. Its output is isolated under:

```text
x3_lds_exp_50k/checkpoint_endpoint_adam_diagnostic_mc20_10q_100t/
```

To test every adjacent checkpoint pair, the same launcher maintains a dynamic
four-GPU queue (one pair per GPU at a time):

```bash
python -u 207_launch_checkpoint_endpoint_adam_diagnostic_4gpu.py \
  --gpus 0,1,2,3 \
  --pair-indices 0-48 \
  --family prompted \
  --query-ids 0-9 \
  --loss-mc 1
```

Once the pair files exist, `--analyze-only` recomputes the 49-pair aggregate
without launching GPU workers.

After either run, fit one parameter-space scalar per checkpoint pair and
frozen-gradient method without rerunning the GPU computation:

```bash
python -u 208_eval_checkpoint_endpoint_scalar_calibration.py \
  --pair-indices 0,16,32,48 \
  --family prompted \
  --loss-mc 1
```

Use `--loss-mc 20` for the MC20 outputs.  The fitted scalar is

```text
alpha_c = <d_c, Delta-theta_c> / ||d_c||^2,
```

where `d_c` is the frozen-gradient AdamW displacement and `Delta-theta_c` is
the observed displacement between the saved checkpoints.  This calibration
can correct response scale but cannot improve directional cosine.
