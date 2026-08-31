# File 3 — `gpu_sampler.py`

**Write:** `learn/src/gpu/gpu_sampler.py`  
**Reference:** `common/gpu_sampler.py`  
**Dependencies:** none (subprocess only)  
**Run tests inside `ds8-dev`** (needs `nvidia-smi`)

---

## Substep 3.1 — Constants `QUERY` and `FIELDS`

Match field order from nvidia-smi CSV output.

---

## Substep 3.2 — `_to_float(raw)`

Strip; return `None` for `N/A`, empty; else `float(raw)` or `None` on error.

**Test:**

```python
assert _to_float("42") == 42.0
assert _to_float("N/A") is None
```

---

## Substep 3.3 — `query_gpu()`

Run `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` with 3s timeout.

Parse CSV into dict: `{"t": time.time(), **FIELDS}`.

**Test (in container):**

```sh
python3 -c "from gpu.gpu_sampler import query_gpu; print(query_gpu())"
```

Expect `gpu_util` numeric or None.

---

## Substep 3.4 — `GpuSampler.__init__(emit, interval=1.0)`

Store emit callback, interval, `samples` list, `threading.Event`, create thread (do not start yet).

---

## Substep 3.5 — `GpuSampler.start()`

One sync `query_gpu()` into `samples` (ignore errors); start thread.

---

## Substep 3.6 — `GpuSampler._run`

While not stopped: sample, append, optional `emit({"type":"gpu", **row})`, wait `interval`.

---

## Substep 3.7 — `GpuSampler.stop()`

Set event; join thread (5s timeout).

---

## Substep 3.8 — `GpuSampler.summary()`

Aggregate: `gpu_util_avg`, `mem_used_mb_max`, `nvdec_util_avg`, `nvenc_util_avg`, `power_w_avg`, `temp_c_max`, `gpu_samples`.

**Test:** 3s run → `gpu_samples >= 2`.

---

## Checkpoint — `learn/steps/step03_gpu.py`

```python
import time
from logging.async_logger import AsyncLogger
from gpu.gpu_sampler import GpuSampler

log = AsyncLogger("/tmp/learn_gpu.jsonl")
gpu = GpuSampler(emit=log.emit, interval=1.0)
gpu.start()
time.sleep(3.5)
gpu.stop()
log.close()
print(gpu.summary())
```

Expected: JSONL has `type: gpu` lines; summary has `gpu_samples >= 2`.

Next: [04-ds_latency.md](04-ds_latency.md)
