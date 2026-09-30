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
