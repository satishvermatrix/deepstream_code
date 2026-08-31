# Learn: build multisrc tracker + metrics from scratch

Step-by-step tutorials live in `learn/tutorial/`. Work **one file at a time**, **one function at a time**.

```text
learn/
  tutorial/
    README.md                 ← start here (ordered index)
    00-prerequisites.md
    01-stage_metrics.md
    02-async_logger.md
    03-gpu_sampler.md
    04-ds_latency.md
    05-pgie_batch.md
    06-mlflow_run.md
    07-pipeline_tracker.md
    08-pipeline_tracker_multisrc.md
```

Your implementation target (parallel tree):

```text
learn/src/                   ← you write code here while learning
  metrics/stage_metrics.py
  logging/async_logger.py
  ...
```

Reference implementation: `common/` and `pipeline_tracker*.py` in the repo root.
