# Null timestamp-block updates evaluated across target directions

Ten deterministic source datapoints each receive one fixed training-noise axis.
The 1000 diffusion timestamps are split from clean/near to noisy/far into four
blocks: 0-249, 250-499, 500-749, and 750-999. Every block starts independently
from the same raw null checkpoint, restores the checkpoint AdamW state, averages
the 250 losses, and takes exactly one clipped AdamW step. This creates 40 updated
models rather than one four-step model per source.

Each source is paired with a different target datapoint. Five deterministic,
equal-norm random noise directions pollute the target, which uses its own prompt.
For every updated model and all 1000 target timestamps, the experiment saves
`predicted_noise(updated) - predicted_noise(null)`, its L2/MSE/max-absolute
magnitude, and pairwise cosine across the five target directions.

```bash
python 166_verify_null_tblock_cross_direction.py
python -u 168_launch_null_tblock_cross_direction_4gpu.py --gpus 0,1,2,3
python -u 169_print_null_tblock_cross_direction.py
```

Outputs are written under:

```text
x3_lds_exp_50k/null_tblock_cross_direction_10source_40models_5targetdirs/
```
