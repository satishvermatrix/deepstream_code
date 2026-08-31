# File 4 — `ds_latency.py`

**Write:** `learn/src/metrics/ds_latency.py`  
**Reference:** `common/ds_latency.py`  
**Dependencies:** `stage_metrics`, Gst, pyds

---

## Substep 4.1 — Gst / pyds imports

```python
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402
import pyds  # noqa: E402
from metrics.stage_metrics import StageMetrics
```

Set env before import in app (not this file):

```python
os.environ.setdefault("NVDS_ENABLE_LATENCY_MEASUREMENT", "1")
os.environ.setdefault("NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT", "1")
```

---

## Substep 4.2 — `MAX_COMPONENT_LEN` and ctypes structs

Define `NvDsMetaSubCompLatency` and `NvDsMetaCompLatency` matching DeepStream C layout.

**Test:** struct size reasonable; no import error.

---

## Substep 4.3 — `_glist_foreach(lst, caster)`

Walk GList: yield `caster(lst.data)`, advance `lst.next`; handle `StopIteration`.

**Test:** only possible with real list or skip until pipeline step.

---

## Substep 4.4 — `_as_ptr(obj)`

Try `pyds.get_ptr`, else int, else `hash(obj)`.

---

## Substep 4.5 — `_decode_name(raw: bytes)`

Split on `\0`, utf-8 decode, strip.

**Test:** `b"pgie\x00garbage"` → `"pgie"`.

---

## Substep 4.6 — `_comp_from_user_meta(user_meta)`

1. Check `meta_type == NVDS_LATENCY_MEASUREMENT_META`
2. `NvDsMetaCompLatency.from_address(_as_ptr(user_meta.user_meta_data))`
3. Validate name chars; `ms = out - in`; reject if ms < 0 or > 60000
4. Return `(name, ms)` or None

**Test:** defer to pipeline with real buffer.

---

## Substep 4.7 — `read_component_latencies_ms(batch_meta)`

Walk `batch_user_meta_list` and per-frame `frame_user_meta_list`; max-merge by component name.

**Test:** after tracker pipeline runs, OSD probe sees non-empty dict when env flags on.

---

## Substep 4.8 — `stage_frame_probe(stage, metrics)` factory

Inner `_probe(pad, info, user_data)`:

1. Get buffer; if none → `metrics.bump(stage)`; return OK
2. `batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))`
3. For each frame: `metrics.mark_stage(stage, frame_num, stream_id=pad_index)`
4. If no frames → `bump(stage)`

Return `_probe`.

**Test:** mux probe increments `frames_mux` per frame in batch.

---

## Checkpoint

Integrate in step 7 pipeline: after EOS, `frames_mux == frames_pgie` for single source.

Next: [05-pgie_batch.md](05-pgie_batch.md)
