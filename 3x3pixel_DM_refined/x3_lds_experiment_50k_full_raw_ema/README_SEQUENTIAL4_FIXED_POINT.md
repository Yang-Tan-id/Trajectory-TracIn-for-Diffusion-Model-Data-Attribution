# Four sequential updates: fixed-point response prediction

This experiment reproduces the original four-update source process:

```text
theta0 --block 1 (250 t)--> theta1
       --block 2 (250 t)--> theta2
       --block 3 (250 t)--> theta3
       --block 4 (250 t)--> theta4
```

Each loss gradient is evaluated at its current parameters. The optimizer is the
restored AdamW optimizer from the null checkpoint, matching the original saved
four-update model. The replayed final parameters are checked against that saved
model before response results are interpreted.

At each fixed target endpoint, pollution direction, timestep, and prompt, the
experiment compares a single initial Jacobian, a sum of four step-local
Jacobians, stepwise trapezoidal integration, and stepwise two-point
Gauss-Legendre integration.

Run:

```bash
python -u 193_launch_sequential4_fixed_point_4gpu.py \
  --gpus 0,1,2,3 \
  --batch-size 256
```

Outputs:

```text
x3_lds_exp_50k/sequential4_restored_adamw_fixed_point_20dir_100t_float64/logs/sequential4_fixed_point_4gpu.log
x3_lds_exp_50k/sequential4_restored_adamw_fixed_point_20dir_100t_float64/summary.json
```
