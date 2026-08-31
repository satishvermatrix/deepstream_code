# Tutorial index

Build `pipeline_tracker_multisrc.py` and all metrics code **modularly**. Each doc = **one source file**. Inside each doc, substeps = **one function** (or class method).

## Order

| Step | Tutorial | Target file you write |
|------|----------|------------------------|
| 0 | [00-prerequisites.md](00-prerequisites.md) | Environment only |
| 1 | [01-stage_metrics.md](01-stage_metrics.md) | `learn/src/metrics/stage_metrics.py` |
| 2 | [02-async_logger.md](02-async_logger.md) | `learn/src/logging/async_logger.py` |
| 3 | [03-gpu_sampler.md](03-gpu_sampler.md) | `learn/src/gpu/gpu_sampler.py` |
| 4 | [04-ds_latency.md](04-ds_latency.md) | `learn/src/metrics/ds_latency.py` |
| 5 | [05-pgie_batch.md](05-pgie_batch.md) | `learn/src/config/pgie_batch.py` |
| 6 | [06-mlflow_run.md](06-mlflow_run.md) | `learn/src/mlflow_run.py` |
| 7 | [07-pipeline_tracker.md](07-pipeline_tracker.md) | `learn/src/pipeline/tracker_single.py` |
| 8 | [08-pipeline_tracker_multisrc.md](08-pipeline_tracker_multisrc.md) | `learn/src/app/run_multisrc.py` |

## Rules

1. **No GStreamer imports** in metrics/logging/gpu modules until step 4 (`ds_latency` needs Gst/pyds).
2. **Test after every substep** — each doc ends with a checkpoint.
3. **Fakesink before filesink** — capacity runs without encode.
4. Compare finished modules to `common/` when curious, but type your own version first.

## Quick test commands (inside `ds8-dev`)

```sh
docker exec -it ds8-dev bash
cd /workspace/learn
python3 -m pytest tests/ -v          # after you add tests
python3 src/app/run_multisrc.py -n 2 --sink fake --run-name learn_n2
```
