# File 1 — `stage_metrics.py`

**Write:** `learn/src/metrics/stage_metrics.py` (import as `metrics.stage_metrics` with PYTHONPATH)  
**Reference:** `common/stage_metrics.py`  
**Dependencies:** none (pure Python)

---

## Substep 1.1 — Module header and constants

Add:

- `from __future__ import annotations`
- imports: `math`, `time`, `defaultdict`, `Any`
- `_HOP_PREV` dict (`mux→pgie→trk→osd→enc` chain)
- `_TS_CAP = 4096`, `_SAMPLE_CAP = 4096`

**Test:** file imports without error.

```sh
python3 -c "from metrics.stage_metrics import _HOP_PREV; print(_HOP_PREV)"
```

---

## Substep 1.2 — `frame_key(stream_id, frame_num)`

Pack `(stream_id, frame_num)` into one int for dict keys.

```python
def frame_key(stream_id: int, frame_num: int) -> int:
    return (int(stream_id) << 32) | (int(frame_num) & 0xFFFFFFFF)
```

**Test:**

```python
assert frame_key(0, 100) != frame_key(1, 100)
assert frame_key(0, 100) != frame_key(0, 101)
```

---

## Substep 1.3 — `_percentile(samples, p)`

Return p-th percentile of a list; empty → `0.0`.

**Test:** `[1,2,3,4,5]` p95 > 4; `[]` → 0.

---

## Substep 1.4 — `_stats(samples)`

Return `avg`, `min`, `max`, `p95`, `n`. Empty list → all zeros.

**Test:** `[10, 20, 30]` avg=20, n=3.

---

## Substep 1.5 — `sanitize_metric_name(name)`

Replace non-alphanumeric with `_`, collapse `__`, default `"unknown"`.

**Test:**

```python
assert sanitize_metric_name("nvstreammux-Stream-muxer") == "nvstreammux_Stream_muxer"
```

---

## Substep 1.6 — `StageMetrics.__init__`

Initialize:

- `counts`, `objects`, `by_class`, `dropped_frames`
- `_last_frame_num`, `_stage_ts`, `_hop_samples`, `_ds_samples`, `_frame_hops`
- `_probe_log_sum_us`, `_probe_log_n`, `t_start`, `t_end`

**Test:** `m = StageMetrics()` — all counters zero.

---

## Substep 1.7 — `StageMetrics.bump(stage)`

Increment `counts[stage]`; set `t_start` on first call.

**Test:**

```python
m = StageMetrics()
m.bump("src")
assert m.counts["src"] == 1
assert m.t_start is not None
```

---

## Substep 1.8 — `StageMetrics.mark_mux_frame`

Delegate to `mark_stage("mux", ...)`. Implement **after** `mark_stage` or stub `mark_stage` first.

---

## Substep 1.9 — `StageMetrics.mark_stage(stage, frame_num, stream_id=0)`

Core hop logic:

1. `now = time.perf_counter()`; bump `counts[stage]`
2. Store `now` in `_stage_ts[stage][frame_key(...)]`; cap at `_TS_CAP`
3. If `_HOP_PREV[stage]` exists, compute `{prev}_{stage}` hop ms
4. If `stage == "osd"`, also compute `mux_osd`
5. Append to `_hop_samples` and `_frame_hops`; return `hops` dict

**Test (synthetic, no sleep required if you mock times):**

```python
m = StageMetrics()
# Manually set _stage_ts["mux"][key] = t0, then mark_stage("pgie", ...)
# Or use small sleep between mark_stage calls
```

---

## Substep 1.10 — `StageMetrics.hops_for_frame(frame_num, stream_id=0)`

Return copy of `_frame_hops[frame_key(...)]` or `{}`.

**Test:** after marking mux→pgie for frame 0, lookup returns `mux_pgie`.

---

## Substep 1.11 — `StageMetrics.note_osd_frame(...)`

1. `hops = mark_stage("osd", ...)`
2. Update `t_end`, `objects`, `by_class`
3. Detect dropped frames via `frame_num` gap per `stream_id`
4. Return `(hops.get("mux_osd", 0.0), hops)`

**Test:** 3 consecutive frames → `objects` = sum of `n_obj`; `dropped_frames` = 0.

---

## Substep 1.12 — `StageMetrics.note_ds_latencies(by_component)`

Append each component ms to `_ds_samples[sanitize_metric_name(name)]`.

**Test:** two calls with `{"pgie": 5.0}` → list length 2.

---

## Substep 1.13 — `StageMetrics.note_probe_log_us(elapsed_us)`

Accumulate sum and count.

**Test:** two calls 10.0 and 20.0 → `probe_log_us_avg()` = 15.0.

---

## Substep 1.14 — `StageMetrics.fps_e2e`

`counts["osd"] / (t_end - t_start)` or 0 if invalid.

**Test:** after `note_osd_frame` on 30 frames with known `t_start`/`t_end`.

---

## Substep 1.15 — `StageMetrics.latency_e2e_ms_avg`

Mean of `_hop_samples["mux_osd"]`.

---

## Substep 1.16 — `StageMetrics.probe_log_us_avg`

`_probe_log_sum_us / _probe_log_n`.

---

## Substep 1.17 — `StageMetrics.summary`

Build full dict: frame counts, FPS, hops/ds stats via `_stats`, flatten keys like `latency_hop_mux_osd_ms_p95`.

**Test:** run synthetic loop (10 frames), `summary()` has `frames_osd==10`, `latency_hops_ms` keys present.

---

## Checkpoint — `learn/steps/step01_metrics.py`

```python
"""Simulate probes without GStreamer."""
import time
from metrics.stage_metrics import StageMetrics

m = StageMetrics()
for fn in range(50):
    m.mark_stage("mux", fn, stream_id=0)
    time.sleep(0.001)
    m.mark_stage("pgie", fn, stream_id=0)
    time.sleep(0.002)
    m.mark_stage("trk", fn, stream_id=0)
    time.sleep(0.001)
    m.note_osd_frame(fn, 2, {"person": 2}, stream_id=0)

import json
print(json.dumps(m.summary(), indent=2))
```

Expected: `frames_osd=50`, `fps_e2e>0`, `mux_osd` avg ≈ sum of hop parts.

Next: [02-async_logger.md](02-async_logger.md)
