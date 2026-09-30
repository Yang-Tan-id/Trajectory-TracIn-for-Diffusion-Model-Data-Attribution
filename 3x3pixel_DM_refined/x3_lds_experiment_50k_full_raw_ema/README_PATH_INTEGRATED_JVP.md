# Fixed-point path-integrated JVP test

This test predicts the finite predicted-noise change at one fixed target
endpoint, pollution direction, timestep, and prompt. It does not average the
target before evaluating prediction accuracy.

It reuses the 40 existing one-step fresh-SGD branches and compares:

- the Jacobian at the starting parameters;
- second-order directional Taylor expansion;
- endpoint trapezoidal integration;
- two-point Gauss-Legendre parameter-path integration;
- four-point Gauss-Legendre parameter-path integration.

The diagnostic disables TF32 and evaluates the model, diffusion schedule,
finite difference, and JVPs in float64. This is necessary because the fresh-SGD
parameter delta is only about `2e-5`; ordinary TF32/float32 finite subtraction
is not reliable enough to validate the parameter-path identity.

It uses 20 fixed pollution directions and 100 evenly spaced timesteps for every
branch. Run it on four GPUs:

```bash
python -u 189_launch_path_integrated_jvp_4gpu.py \
  --gpus 0,1,2,3 \
  --batch-size 256
```

If nested JVP memory is too high, reduce `--batch-size` to 256 or 128. Resume is
automatic at the completed block level. The shared log is:

```text
x3_lds_exp_50k/path_integrated_jvp_fresh_sgd_20dir_100t_float64/logs/path_integrated_jvp_4gpu.log
```

The final summary is:

```text
x3_lds_exp_50k/path_integrated_jvp_fresh_sgd_20dir_100t_float64/summary.json
```

After completion, analysis can be repeated without GPUs:

```bash
python -u 189_launch_path_integrated_jvp_4gpu.py --analyze-only
```
