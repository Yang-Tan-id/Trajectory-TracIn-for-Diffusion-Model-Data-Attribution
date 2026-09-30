# Frozen-start AdamW tangent Bundle TracIn

This experiment propagates all 192 LDS-subset weight tangents through every
one of the 49 adjacent raw-checkpoint pairs.  Gradients are evaluated at the
pair's start checkpoint using the exact four cached training events.  AdamW
momentum, second moment, clipping, weight decay, batch order, and learning-rate
schedule are differentiated along the frozen-start optimizer path.

Each pair is independently scalar-calibrated against its observed parameter
delta.  Query predicted-noise JVPs use that pair's start checkpoint.  The
final score is `mean_t ||sum_c response_subset,c,t||^2`.

Run ten queries on four GPUs:

```bash
python -u 211_launch_adam_bundle_pairs_4gpu.py \
  --gpus 0,1,2,3 \
  --query-ids 0-9
```

The launcher resumes completed pair directories and evaluates all LDS targets
after all 49 pairs finish.
