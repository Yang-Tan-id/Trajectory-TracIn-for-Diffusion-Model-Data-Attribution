# Query-dependent trajectory inverse-noise DAS

This is DAS itself, not TracIn-DAS. It uses only the final EMA model and has no
checkpoint sum or learning-rate weight.

For every q00-q09 reference trajectory state `x_t^q` and training point `x_i`,
the deterministic noise target is inferred from the forward equation:

```text
epsilon*_(i,q,t) =
    (x_t^q - sqrt(alpha_bar_t) x_i) / sqrt(1 - alpha_bar_t).
```

For each Gaussian output probe, DAS uses the same query state on both sides:

```text
phi_(i,q,t) = grad_theta <epsilon_theta(x_t^q,t,c_i), probe>
r_(i,q,t)   = <epsilon_theta(x_t^q,t,c_i)-epsilon*_(i,q,t), probe>
phi_(q,t)   = grad_theta <epsilon_theta(x_t^q,t,c_q), probe>
```

The feature Gram matrix is query/timestamp/probe dependent. The score is the
squared damped DAS response and is averaged over 99 trajectory states and 10
Gaussian probes. Endpoint index 99 (`t=0`) is excluded. Features use the normal
DAS L2 normalization and a shared CountSketch4096 projection.

Training feature gradients depend on `c_i` but not otherwise on `x_i`, so the
implementation computes them once per unique training condition, weights the
Gram matrix by condition frequency, and maps residuals back to all 50k points.
This is exactly equivalent to expanding all duplicate condition features.

```bash
python -u 142_launch_trajectory_inverse_noise_das_10q_4gpu.py \
  --gpus 0,1,2,3 \
  --condition-batch-size 64
```

The launcher merges all lambdas and evaluates all LDS targets. Output:

```text
x3_lds_exp_50k/lds/trajectory_inverse_noise_das_q00_q09_lambda_sweep.json
```
