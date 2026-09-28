# Reference forward-10 predicted-noise TracIn

For each of the 100 cached final-EMA reference-trajectory states `x_t`, draw
one deterministic Gaussian direction and forward-noise the state by ten
diffusion indices:

```text
x_{t->t+10} = sqrt(alpha_bar[t+10] / alpha_bar[t]) * x_t
              + sqrt(1 - alpha_bar[t+10] / alpha_bar[t]) * noise
```

The query side is predicted noise projected onto the next-checkpoint delta
direction at exactly the same forward-noised reference state; it is not a query
loss:

```text
direction = unit(model_{c+1}(x_{t->t+10}, t+10, query_cond)
                 - model_c(x_{t->t+10}, t+10, query_cond))
g_query = grad_theta dot(model_c(x_{t->t+10}, t+10, query_cond), direction)
```

The train side is the current-checkpoint diffusion-loss gradient at `t+10`,
using exactly the same noise that constructs the query state. The first
zero-based reference timestep is 999, so its target is
1009 (one-based timestep 1010). The original linear beta formula is extended
for exactly ten indices. The experiment uses all 49 next pairs formed by the
50 raw checkpoints (including the final checkpoint as the last target), their
saved source-checkpoint learning rates, CountSketch4096, all 100 queries, and emits linear,
termwise-square, and timestamp-sum-square scores.

Run on four GPUs:

```bash
python -u 102_launch_reference_forward10_aligned_100q_4gpu.py \
  --batch-size 5120
```

Then evaluate all LDS targets and both signs:

```bash
python -u 103_eval_reference_forward10_aligned_100q_lds.py
```
