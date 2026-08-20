# X3 5000 / LDS experiment

## Training seed

All full and subset models use the SAME training seed:

```python
TRAIN_SEED = 67
```

The LDS mask-bank seeds (0, 1, 2) only determine which 25% datapoints are selected.
They do NOT change model initialization / loader shuffle / diffusion timestep-noise seed.


## Intended experiment

- Base data: 5000 generated points, generator seed 67.
- Full models: prompted + unprompted.
- LDS masks: 3 seeds × 64 masks, each 25% = 1250 points.
- The 192 masks are shared across model families.
- Default config trains both prompted and unprompted final subset checkpoints per mask.
- 7 initial noise seeds:
  - seeds 0,1,2 are prompted; 4 random prompts each => 12 queries
  - seeds 3,4,5,6 are unprompted => 4 queries
  - total = 16
- Traj-TracIn: reference target + next-checkpoint target.
- DAS: projected dimension 4096, later-JAX logic, lambda sweep.
- LDS: Spearman correlation of subset attribution sums vs actual subset-model response.

## Run order

```bash
python 00_prepare_experiment.py
python 01_launch_train_4gpu.py
python 02_build_queries.py
python 03_cache_traj_query_gradients.py
```

Recommended 4-GPU attribution launch:

```bash
python 04_launch_attribution_4gpu.py
```

Or run methods manually:

```bash
python 04_run_attribution.py --gpu 0 --method traj_ref
python 04_run_attribution.py --gpu 1 --method traj_next
python 04_run_attribution.py --gpu 2 --method das
```

Then:

```bash
python 05_collect_subset_outputs.py

python 06_lds_eval.py --method traj_ref  --metric traj_ref
python 06_lds_eval.py --method traj_next --metric traj_ref

python 06_lds_eval.py --method das --metric traj_ref --lambda 0.1
python 06_lds_eval.py --method das --metric traj_ref --lambda 0.2
# ... sweep ...
```

You can also evaluate against simple-loss response:

```bash
python 06_lds_eval.py --method traj_ref --metric simple_loss
python 06_lds_eval.py --method das --metric simple_loss --lambda 2.0
```

## Important count

`3 × 64 = 192` refers to subset MASKS / subset JOB SLOTS.

The default:
```python
SUBSET_TRAIN_FAMILIES = ("prompted", "unprompted")
```
trains two final models for each mask, because a prompted LDS query should be compared
against prompted subset models and an unprompted LDS query against unprompted subset models.

If you truly want 192 trained subset models total, change it to one family.


## Eta convention

There are two unrelated eta symbols in the code:

1. **Traj-TracIn checkpoint eta**
   - `eta_c = learning rate at checkpoint c`
   - saved directly inside every newly trained checkpoint as `ckpt["eta"]`
   - both Traj-Ref and Traj-Next multiply each checkpoint contribution by `eta_c`
   - query gradients themselves are NOT multiplied when cached; eta is applied when aggregating TracIn scores.

2. **DDIM sampling eta**
   - remains `0.0`
   - this controls DDIM stochasticity and is unrelated to TracIn checkpoint weighting.

DAS does not use the TracIn checkpoint eta weighting.



## Progress / ETA

The updated scripts print approximate ETA at multiple levels:

- `train_worker.py`: epoch-level elapsed/ETA for each model.
- `01_launch_train_4gpu.py`: overall completed-model count and queue ETA.
- Traj attribution: every 100 training points plus checkpoint/global ETA.
- DAS attribution: feature-build percentage and ETA.
- `04_launch_attribution_4gpu.py`: overall 48-job attribution queue ETA.
- `05_collect_subset_outputs.py`: LDS query/subset progress and ETA.

Early ETA values can be noisy; they stabilize after several completed units.


## Fast attribution backend

This package replaces the low-GPU-utilization attribution loop with:

- Traj-TracIn: batched MC + batched datapoints + `torch.func.jvp`.
  The JVP returns the exact per-example dot product
  `<query_gradient, train_loss_gradient>` without forming one train gradient
  with one backward per datapoint.
- DAS: `torch.func.vmap(grad)` across a train batch, batched CountSketch,
  GPU-resident 4096-D Gram, GPU lambda solve.

Main tuning knobs:

```python
TRACIN_SCORE_BATCH_SIZE = 512
DAS_FEATURE_BATCH_SIZE = 64
```

If memory allows, increase them. If OOM occurs, reduce them.
