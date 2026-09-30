# Can opposite-side gradients predict output-change magnitude elsewhere?

This experiment uses the clean `fresh_sgd` one-sided updates as ground truth.
For each source and timestamp block, the actual model was updated only with the
fixed `+epsilon` source loss. At the null model the experiment computes source
gradients from `+epsilon`, `-epsilon`, their even average, and their odd
difference. Every candidate is converted into a counterfactual SGD update using
the same checkpoint learning rate and global-norm clipping rule.

Exact JVPs predict output changes at a different target datapoint, its own
prompt, five independent directions, and all 1000 timestamps. Predictions are
compared with the finite updated-minus-null output from the saved fresh-SGD
model. Reported metrics include magnitude Pearson/Spearman correlation,
predicted/actual L2 ratio, relative magnitude error, and vector cosine. A JVP
using the actual saved parameter delta is the linearization control.

The run also reconstructs the saved fresh-SGD parameter step from the plus-side
gradient and requires their cosine to be at least 0.999. Thus
`plus_gradient_sgd` is a faithful first-order prediction of the actual update,
while `actual_parameter_delta_jvp` is the best local-linearization control.

```bash
python 172_verify_opposite_gradient_magnitude.py
python -u 174_launch_opposite_gradient_magnitude_4gpu.py --gpus 0,1,2,3
```

Summary:

```text
x3_lds_exp_50k/null_tblock_opposite_gradient_magnitude_fresh_sgd/summary.json
```
