# Trajectory-state relative-forward aligned DAS

For q00-q99, take the cached reference-trajectory states at approximately
`t=888,777,...,111,0`. The initial `t=999` state is deliberately skipped.
For anchor state `x_t`, choose `10,20,...,90` target levels evenly over
`[t,999]` and forward-noise directly from `x_t` to each target `s`:

```text
x_s = sqrt(alpha_bar_s / alpha_bar_t) * x_t
      + sqrt(1 - alpha_bar_s / alpha_bar_t) * noise
```

Thus the `t≈888` anchor uses ten targets covering approximately `888..999`.
At every `(anchor,target,MC)` term, query relative-forward noise and training
loss noise are identical. Each anchor is averaged over its own target count and
MC10; the nine independently squared DAS scores are then summed. The final EMA
model, projection dimension 4096, and all configured damping lambdas are used.

```bash
python -u 127_launch_multiclean_aligned_das_100q_4gpu.py \
  --gpus 4,5,6,7 \
  --batch-size 2560
python -u 128_eval_multiclean_aligned_das_100q_lds.py
```
