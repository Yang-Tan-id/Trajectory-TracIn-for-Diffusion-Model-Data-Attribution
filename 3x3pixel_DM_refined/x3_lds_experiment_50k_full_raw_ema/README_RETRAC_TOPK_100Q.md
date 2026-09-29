# ReTrac / End-TracIn / TracIn-DAS top-k removal

This experiment retrains 900 models:

- 100 queries (`q00` through `q99`);
- top-positive removal at 2%, 5%, and 10% (1,000 / 2,500 / 5,000 points);
- normalized Diffusion-ReTrac;
- unnormalized endpoint Diffusion-TracIn;
- aligned checkpoint-noise CountSketch4096 TracIn-DAS using the
  timestamp-sum-squared contraction.

Scores are ranked directly in descending order without applying an LDS sign.
Each retrained model uses the query's prompted/unprompted family and is evaluated
with EMA parameters, the same prompt, and the same initial diffusion noise as
the saved reference query.

After the three 100-query attribution banks exist, run:

```bash
python -u 118_launch_retrac_tracindas_topk_100q_4gpu.py \
  --gpus 0,1,2,3
```

The run is resumable: completed final checkpoints/evaluations are skipped and
incomplete training jobs resume from their latest ten-epoch checkpoint.

Outputs are under:

```text
x3_lds_exp_50k/topk_removal_retrac_endtracin_tracindas_100q/
```

Important reports:

- `per_query_results.csv`: raw endpoint/trajectory changes for every job;
- `per_query_method_differences.csv`: every pairwise method difference for
  every query and removal fraction;
- `per_query_endpoint_trajectory_differences.txt`: readable per-query report;
- `method_overlap.json`: pairwise overlap of removed sets;
- `summary.json`: aggregate means, standard deviations, medians, ranges, and
  pairwise differences.
