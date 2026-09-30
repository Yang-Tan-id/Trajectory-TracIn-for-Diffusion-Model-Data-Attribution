# Target pollution/loss-noise alignment ablation

This is a controlled A/B test of target-side noise alignment. It reuses the
same endpoints, 100 pollution directions, 1000 noise levels, saved fresh-SGD
parameter updates, and finite-response ground truth from the aligned endpoint
MC experiment.

- Aligned: pollution direction `epsilon_r` is the target of the diffusion loss.
- Non-aligned: the input is still polluted with `epsilon_r`, but the loss target
  is `epsilon_(r+1 mod 100)`.

The cyclic shift guarantees a different target noise while preserving the exact
empirical noise bank and all other variables. Both loss projections receive the
same per-curve median multiplicative calibration before relative-error and
false-small comparisons.

```bash
python 181_verify_endpoint_direction_nonaligned.py
python -u 183_launch_endpoint_direction_nonaligned_4gpu.py \
  --gpus 0,1,2,3 \
  --batch-size 5120
```

Re-analyze saved outputs with:

```bash
python -u 183_launch_endpoint_direction_nonaligned_4gpu.py --analyze-only
```

Summary:

```text
x3_lds_exp_50k/endpoint_direction_mc_nonaligned_shift100dir_1000t/summary.json
```
