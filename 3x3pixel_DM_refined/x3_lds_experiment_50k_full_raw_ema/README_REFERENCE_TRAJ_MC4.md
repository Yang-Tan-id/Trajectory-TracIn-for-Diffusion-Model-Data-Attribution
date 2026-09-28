# Reference-trajectory state-perturbation MC4

This experiment uses q00-q09 from the original non-staged experiment. At each
of the 100 cached final-EMA reference-trajectory states, it creates four query
points

```text
x_ref(t) + epsilon * unit_gaussian_direction(q, t, m),  m=0..3.
```

The four deterministic directions are fixed across all 49 raw-checkpoint next
transitions. At every perturbed point, the current predicted noise is projected
onto the normalized next-minus-current predicted-noise direction. Each of the
four resulting query gradients is CountSketch-projected to 4096 dimensions and
paired separately with the same timestamp-aligned training-loss gradient.

The training gradient uses one independent random diffusion-loss noise per
datapoint by default (`--train-mc 1`). Query perturbations and training noises
are independent. Larger train-MC values remain available as an option.

Run with the default L2 perturbation radius 0.01:

```bash
python -u 79_launch_reference_traj_mc4_4gpu.py \
  --epsilon 0.01 --train-mc 1 --batch-size 128
```

Then evaluate linear, termwise-square, and timestamp-sum-square against all
eight LDS response metrics:

```bash
python -u 80_eval_reference_traj_mc4_lds.py --epsilon 0.01 --train-mc 1
```

Every epsilon value gets a separate shard, method, and LDS-summary namespace.
