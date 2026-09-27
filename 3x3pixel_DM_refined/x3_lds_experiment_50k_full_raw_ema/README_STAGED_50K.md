# Five-stage 50k experiment

This experiment writes only under `x3_staged_lds_exp_50k` and does not modify
the existing `x3_lds_exp_50k` results.

```bash
python 60_prepare_staged_50k.py
python 72_verify_staged_50k.py
python -u 62_launch_staged_training_4gpu.py
python -u 63_build_staged_10_queries.py
python -u 65_launch_staged_observed_4gpu.py

# Attribution may be run separately. DAS remains final-EMA/all-50k.
python -u 70_launch_staged_attribution_4gpu.py --experiment traj
python -u 70_launch_staged_attribution_4gpu.py --experiment das

python -u 71_eval_staged_lds.py
```

Training is continuous across the five 40-epoch stages: model parameters,
AdamW state, EMA state, global step, and the cosine learning-rate schedule are
not reset. Every LDS subset contains exactly 5,000 points from every stage.

For the checkpoint transition ending at epoch `e`, Traj training gradients use
the stage that trained epochs `e-3..e`. Thus the boundary transition 40 -> 44
uses stage 2. DAS deliberately ignores stage membership and uses the final EMA
checkpoint with all 50,000 training examples.
