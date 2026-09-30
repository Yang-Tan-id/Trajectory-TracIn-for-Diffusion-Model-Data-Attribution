# Checkpoint-transition diagnostic

This experiment tests whether the response vectors used by Bundle TracIn can
predict the actual predicted-noise change between adjacent saved checkpoints.
It covers the 49 available transitions from epoch 4->8 through epoch 196->200;
there is no saved epoch-0 checkpoint.

For each query trajectory state it compares the finite response

```text
epsilon_theta_target(x_t) - epsilon_theta_start(x_t)
```

against:

1. an exact parameter-delta JVP at the start checkpoint;
2. a restored-AdamW replay delta JVP;
3. the current Bundle gradient-sum direction at the target checkpoint;
4. the same gradient-sum direction scaled by `4 / training_batch_size`,
   evaluated at both the start and target checkpoints.

The AdamW replay restores the start checkpoint optimizer state and replays the
original four epochs using the original shuffled minibatch order, cached
training `t/noise`, clipping, and per-step LR schedule.

Verify prerequisites:

```bash
python 203_verify_checkpoint_transition_diagnostic.py
```

Recommended four-transition pilot, one transition per GPU:

```bash
python -u 205_launch_checkpoint_transition_diagnostic_4gpu.py \
  --gpus 0,1,2,3 \
  --pair-indices 0,16,32,48 \
  --query-ids 0-9 \
  --query-batch-size 256
```

Run all 49 transitions after the pilot succeeds:

```bash
python -u 205_launch_checkpoint_transition_diagnostic_4gpu.py \
  --gpus 0,1,2,3 \
  --pair-indices 0-48 \
  --query-ids 0-9 \
  --query-batch-size 256
```

The run is restartable per transition. To summarize completed outputs without
launching workers, add `--analyze-only` with the same pair/query selections.
The merged report is saved to:

```text
x3_lds_exp_50k/checkpoint_transition_diagnostic_49pair_10q_100t/summary.json
```
