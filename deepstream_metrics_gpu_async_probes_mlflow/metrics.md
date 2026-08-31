# Metrics architecture

How `pipeline_tracker.py` and `pipeline_tracker_multisrc.py` measure performance, where data is collected, and where it lands after a run.

Shared instrumentation lives in `common/`:

| Module | Role |
|--------|------|
| `stage_metrics.py` | In-memory frame counts, hop latencies, detection totals (no I/O) |
| `ds_latency.py` | Read DeepStream `NvDsMetaCompLatency`; lightweight `stage_frame_probe` factory |
| `async_logger.py` | Bounded-queue JSONL (`AsyncLogger`) or sync JSONL (`ProbeSyncLogger`) |
| `gpu_sampler.py` | Background `nvidia-smi` sampler (must not run inside probes) |
| `mlflow_run.py` | Local-file MLflow params, metrics, artifacts |
| `pgie_batch.py` | Per-batch nvinfer config (multisrc only) |

---

## Design principles

1. **Pad probes collect samples on the GStreamer streaming thread** — `StageMetrics` updates are in-memory only.
2. **No blocking I/O in probes** — per-frame logs use `AsyncLogger.emit()` (`put_nowait`) by default; GPU sampling runs on a daemon thread.
3. **Aggregation runs after EOS** — `metrics.summary()` and `gpu.summary()` on the main thread; then JSON file + MLflow.
4. **Two independent latency views** — pad hop wall time vs DeepStream plugin internal time (see below).
5. **Frame correlation** — `(stream_id, frame_num)` via `frame_key()`; multisrc uses `pad_index` as `stream_id`.

**Fair summary:** probes do hot-path **sampling**; end-of-run code does **aggregation**, GPU polling, and persistence.

---

## Metric types

### 1. Frame counts (`frames_*`)

Per-stage buffer/frame counts stored in `StageMetrics.counts`.

| Key | Stage | Typical probe |
|-----|-------|----------------|
| `frames_src` | Decoded frames leaving each `nvurisrcbin` | `count_probe("src")` on `vsrc_*` |
| `frames_mux` | Batched frames leaving `nvstreammux` | `stage_frame_probe("mux")` |
| `frames_pgie` | Frames leaving `nvinfer` | `stage_frame_probe("pgie")` |
| `frames_trk` | Frames leaving `nvtracker` | `stage_frame_probe("trk")` |
| `frames_osd` | Frames at OSD input | `note_osd_frame()` / OSD probe |
| `frames_enc` | Frames entering encoder | `stage_frame_probe("enc")` (when present) |

`stage_frame_probe` increments by **NvDs `frame_num`** per frame in the batch (not one bump per buffer when batch meta exists).

### 2. Pad hop latency (`hops_ms`, `latency_hops_ms`)

Wall-clock time between pad probes for the **same** `(stream_id, frame_num)`, using `time.perf_counter()`.

| Hop | From → To | Recorded when |
|-----|-----------|----------------|
| `mux_pgie` | mux src → pgie src | `mark_stage("pgie", …)` |
| `pgie_trk` | pgie src → tracker src | `mark_stage("trk", …)` |
| `trk_osd` | tracker src → OSD sink | `mark_stage("osd", …)` |
| `mux_osd` | mux src → OSD sink | `mark_stage("osd", …)` — end-to-end hop |
| `osd_enc` | OSD → encoder sink | `mark_stage("enc", …)` (summary only if enc probe exists) |

Special case: if `trk` timestamp is missing for a frame, `pgie_trk` hop falls back to `pgie → osd` path logic uses `pgie` as previous stage for `trk`.

Run summary includes `latency_hops_ms` with **avg, min, max, p95, n** per hop, plus flat keys like `latency_hop_mux_osd_ms_p95`.

Per-frame JSONL field: `hops_ms` (subset for that frame).

`e2e_ms` in JSONL = `mux_osd` hop for that frame.

**Includes:** queueing and `nvvideoconvert` gaps between probed elements.  
**Does not include:** time inside `nvurisrcbin` before `vsrc_0` (decode is before the src probe).

### 3. DeepStream plugin latency (`ds_ms`, `latency_ds_ms`)

Official per-component latency from `NvDsMetaCompLatency` when:

```text
NVDS_ENABLE_LATENCY_MEASUREMENT=1
NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1
```

Each plugin records `in_system_timestamp` at buffer input and `out_system_timestamp` at output:

```text
component_latency_ms = out_system_timestamp - in_system_timestamp
```

Read in the OSD probe via `read_component_latencies_ms(batch_meta)` (`common/ds_latency.py`). Examples: `nvv4l2decoder0`, `nvstreammux_streammux`, `pgie`, `tracker`.

Run summary: `latency_ds_ms` with avg/min/max/p95/n per component.

**Different from hop latency** — measures time **inside** each GStreamer element’s processing, not wall time between your probes.

### 4. Throughput (`fps_e2e`, `fps_per_stream`)

| Metric | Formula | Notes |
|--------|---------|-------|
| `fps_e2e` | `frames_osd / (t_end - t_start)` | `t_end` updated on each OSD frame |
| `fps_per_stream` | `fps_e2e / num_streams` | **multisrc only** |

Multisrc with `--sink file` may override `fps_e2e` using `frames_trk / dt` (encoder path can skew OSD timing).

### 5. Detection / tracking totals

Collected in the OSD probe while walking `obj_meta_list`:

| Metric | Meaning |
|--------|---------|
| `objects_total` | Sum of `n_obj` per frame |
| `by_class` | Histogram of `obj_label` / `class_id` |
| `dropped_frames` | Gap detection: `frame_num > last + 1` per `stream_id` |
| `ids` | Tracker object IDs per frame (JSONL only) |

### 6. GPU metrics (`gpu_*`)

Background thread samples `nvidia-smi` every ~1 s (`GpuSampler`). **Not** collected in probes.

| Summary key | Meaning |
|-------------|---------|
| `gpu_util_avg` | GPU utilization % |
| `mem_used_mb_max` | Peak VRAM used |
| `nvdec_util_avg` / `nvenc_util_avg` | Decoder / encoder utilization |
| `power_w_avg` / `temp_c_max` | Power draw, temperature |
| `gpu_samples` | Number of samples |

Async log mode: JSONL lines with `type: "gpu"`.

### 7. Logging overhead (`probe_log_us_avg`, `log_dropped`)

| Metric | Meaning |
|--------|---------|
| `probe_log_us_avg` | Average microseconds for `log_sink.emit()` in OSD probe |
| `log_dropped` | Records dropped when async queue is full |

`--log-mode probe` uses synchronous disk writes in the streaming thread (A/B comparison).

---

## Where data is collected vs aggregated vs stored

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GStreamer streaming thread (pad probes)                                │
│  • StageMetrics: counts, hop samples, ds samples, objects               │
│  • osd probe: annotate, build frame record, emit() to logger queue      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────┼───────────────────────────────────┐
│  gpu-sampler thread               │  async-logger thread (optional)     │
│  • nvidia-smi → emit(gpu)         │  • JSONL append from queue         │
└───────────────────────────────────┴───────────────────────────────────┘
                                    │
                                    ▼  EOS / shutdown
┌─────────────────────────────────────────────────────────────────────────┐
│  Main thread                                                            │
│  • metrics.summary() + gpu.summary()                                    │
│  • write <run_name>_metrics.json                                        │
│  • MLflow log_metrics + artifacts                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Output files

Replace `<run_name>` with `--run-name` or the default (`tracker_async_<unix>`, `tracker_n4_fake_<unix>`, etc.).

| Path | Contents |
|------|----------|
| `output/logs/<run_name>.jsonl` | Per-frame records + optional GPU lines |
| `output/logs/<run_name>_metrics.json` | Full run summary |
| `mlruns/` | MLflow file store (`tracker` or `tracker_n` experiment) |

### JSONL frame record (example)

```json
{
  "type": "frame",
  "source_id": 0,
  "frame": 200,
  "n_obj": 14,
  "ids": [29, 31, 34],
  "e2e_ms": 11.5,
  "hops_ms": {
    "mux_pgie": 3.9,
    "pgie_trk": 6.5,
    "trk_osd": 0.3,
    "mux_osd": 10.7
  },
  "ds_ms": {
    "nvv4l2decoder0": 55.0,
    "pgie": 3.9,
    "tracker": 6.8
  },
  "log_mode": "async"
}
```

---

## `StageMetrics` internals

- **Correlation key:** `frame_key(stream_id, frame_num) = (stream_id << 32) | frame_num`
- **Caps:** last 4096 timestamps per stage; last 4096 hop/ds samples (rolling)
- **No I/O** — safe to call from any pad probe

Hop predecessor map (`_HOP_PREV`):

```text
mux → pgie → trk → osd → enc
```

---

## `pipeline_tracker.py`

Single-source tracker: one URI, batch size 1, annotated MP4 output.

### Pipeline graph

```text
nvurisrcbin --vsrc_0--> nvstreammux (1920×1080, batch=1)
    --> nvinfer (YOLO26m)
    --> nvtracker (NvDCF accuracy + ResNet50 ReID)
    --> nvvideoconvert --> nvdsosd
    --> nvvideoconvert --> NV12 --> nvv4l2h264enc
    --> h264parse --> qtmux --> filesink
```

### Probe map

| Stage | Element | Pad | Probe | Work |
|-------|---------|-----|-------|------|
| src | `nvurisrcbin` | `vsrc_*` (dynamic) | `count_probe("src")` | `bump("src")` only |
| mux | `nvstreammux` | src | `stage_frame_probe("mux")` | `mark_stage("mux", frame_num)` |
| pgie | `nvinfer` | src | `stage_frame_probe("pgie")` | `mark_stage("pgie", …)` |
| trk | `nvtracker` | src | `stage_frame_probe("trk")` | `mark_stage("trk", …)` |
| osd | `nvdsosd` | **sink** | `osd_sink_pad_probe` | Full: `ds_ms`, objects, `note_osd_frame`, JSONL |
| enc | `nvv4l2h264enc` | sink | `stage_frame_probe("enc")` | `mark_stage("enc", …)` |

OSD probe is registered in `main()` on `osd.get_static_pad("sink")` — runs **before** OSD draws overlays; sets `text_params` for labels (`class conf id:<track_id>`).

`vsrc_*` probe sees **decoded** NVMM frames (after NVDEC inside `nvurisrcbin`). `NvDsBatchMeta` appears from `nvstreammux` onward.

### CLI flags (metrics-related)

| Flag | Effect |
|------|--------|
| `--log-mode async` | `AsyncLogger` + GPU lines to JSONL (default) |
| `--log-mode probe` | `ProbeSyncLogger` — blocking JSONL on streaming thread |
| `--run-name` | Base name for JSONL and `_metrics.json` |
| `--log-video` | Attach output MP4 to MLflow artifacts |

### MLflow

- Experiment: **`tracker`**
- Params: pipeline config, mux size, tracker size, nvinfer ini properties, `log_mode`, paths
- Metrics: numeric fields from merged summary (hops, ds, GPU, FPS, counts)

### Run

```sh
./run_tracker.sh
./run_tracker.sh --run-name yolo26m_nvdcf --log-mode async
```

---

## `pipeline_tracker_multisrc.py`

Same detector + tracker stack, but **N copies** of the same URI for batch scaling experiments.

### Pipeline graph (default: `--sink fake`)

```text
N × nvurisrcbin --vsrc_0--> nvstreammux (batch=N, 1920×1080)
    --> nvinfer (YOLO26, batch=N, dynamic ONNX if N>1)
    --> nvtracker (enable-batch-process=1)
    --> nvvideoconvert --> nvdsosd --> fakesink
```

### Pipeline graph (`--sink file`)

```text
... --> nvtracker
    --> nvmultistreamtiler (grid → 1920×1080)
    --> nvvideoconvert --> nvdsosd
    --> nvvideoconvert --> NV12 --> nvv4l2h264enc
    --> h264parse --> qtmux --> filesink
```

### Probe map

| Stage | Element | Pad | When | Probe |
|-------|---------|-----|------|-------|
| src | `source_{i}` × N | `vsrc_*` | always | `count_probe("src")` → `sink_{i}` on mux |
| mux | `nvstreammux` | src | always | `stage_frame_probe("mux")` |
| pgie | `nvinfer` | src | always | `stage_frame_probe("pgie")` |
| trk | `nvtracker` | src | always | `stage_frame_probe("trk")` |
| osd | `nvdsosd` | sink | always | `osd_sink_pad_probe` (shared with single-src) |
| enc | `nvv4l2h264enc` | sink | **`--sink file` only** | `stage_frame_probe("enc")` |

No enc probe when `--sink fake` (default capacity benchmark).

### Multisrc-specific behavior

| Topic | Behavior |
|-------|----------|
| `stream_id` | `frame_meta.pad_index` (which mux sink / source) |
| PGIE config | `write_pgie_config()` → `output/logs/pgie_yolo26_bN.txt` |
| ONNX | Static batch-1 or dynamic `yolo26m_dyn.onnx` for N>1 |
| Summary extras | `num_streams`, `fps_per_stream`, `sink` |
| `--no-log-frames` | `NullLogger` — metrics still collected; no JSONL frames / no OSD annotate |
| `--no-log-frames` + async | Separate `AsyncLogger` for GPU lines only |

Reuses from `pipeline_tracker.py`: `osd_sink_pad_probe`, `count_probe`, `bus_call`, `make_element`, `to_uri`.

### CLI flags (metrics-related)

| Flag | Default | Effect |
|------|---------|--------|
| `-n` / `--num-streams` | 2 | Mux + PGIE batch size |
| `--sink fake` | yes | `fakesink` — no encode, no enc probe |
| `--sink file` | | Tiled MP4 + enc probe |
| `--no-log-frames` | | Skip per-frame JSONL and OSD text (metrics only) |
| `--log-mode` | async | Same as single-src |
| `--run-name` | | e.g. `tracker_n8_fake_<unix>` |

### MLflow

- Experiment: **`tracker_n`**

### Run

```sh
./run_tracker_multisrc.sh -n 4
./run_tracker_multisrc.sh -n 8 --sink file -o /workspace/output/out.mp4
./run_tracker_multisrc.sh -n 4 --no-log-frames   # capacity test, minimal logging
```

---

## Comparison: single vs multisrc

| | `pipeline_tracker.py` | `pipeline_tracker_multisrc.py` |
|--|----------------------|--------------------------------|
| Sources | 1 | N (`source_0` … `source_{N-1}`) |
| Mux batch | 1 | N |
| After tracker | OSD → encode → MP4 | fake: OSD → fakesink; file: tiler → OSD → encode |
| Enc probe | always | file sink only |
| FPS extras | — | `fps_per_stream`, optional trk-based FPS for file sink |
| Frame logging | always (unless custom) | optional `--no-log-frames` |
| MLflow experiment | `tracker` | `tracker_n` |

---

## Buffer lifecycle (relevant to metrics)

```text
[inside nvurisrcbin]
  encoded H.264 buffers → NVDEC → decoded NVMM NV12
       │
       ▼  vsrc_0  ← PROBE src (decoded frame; no NvDsBatchMeta)
nvstreammux.sink_i
       │  batch, scale 1920×1080, attach NvDsBatchMeta
       ▼  mux src  ← PROBE mux
nvinfer → nvtracker → … → osd sink  ← PROBE osd (full metadata)
```

First buffers your metrics code sees are **after decode**. Object metadata and `frame_num` are reliable from mux output onward.

---

## Related docs

- [Learn tutorial](learn/tutorial/README.md) — step-by-step build from scratch (one file, one function at a time)
- `common/stage_metrics.py` — hop logic and `summary()`
