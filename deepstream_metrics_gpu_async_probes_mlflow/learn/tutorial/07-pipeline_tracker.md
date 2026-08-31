# File 7 — `pipeline_tracker.py` (single source)

**Write:** `learn/src/pipeline/tracker_single.py` (or split `gst_utils.py` + `probes.py`)  
**Reference:** `pipeline_tracker.py`  
**Dependencies:** all modules from steps 1–6

Build **N=1** first with **fakesink** before MP4 encode.

---

## Substep 7.1 — `make_element(factory, name)`

`Gst.ElementFactory.make`; raise if None.

---

## Substep 7.2 — `to_uri(path_or_uri)`

If already `file://`, `http://`, `rtsp://` → return; else `Path.resolve().as_uri()`.

**Test:** `to_uri("output/foo.mp4")` starts with `file://`.

---

## Substep 7.3 — `bus_call(bus, message, loop)`

On EOS → print, `loop.quit()`; on ERROR → print, quit.

---

## Substep 7.4 — `count_probe(stage, metrics)`

Return inner probe that `metrics.bump(stage)` and returns OK.

---

## Substep 7.5 — `on_src_pad_added(src, pad, data)` (single source)

1. Ignore non-`vsrc` pads
2. `request_pad_simple("sink_0")` on mux
3. `pad.link(sinkpad)`
4. `pad.add_probe(BUFFER, count_probe("src", metrics))`

**Test:** step with mux+fakesink only — see "Linked vsrc_0".

---

## Substep 7.6 — `_annotate_object(obj_meta)`

Set `text_params.display_text` = `label conf id:oid`; position above bbox; border width.

Uses `UNTRACKED = getattr(pyds, "UNTRACKED_OBJECT_ID", 0xFFFFFFFFFFFFFFFF)`.

---

## Substep 7.7 — `osd_sink_pad_probe` (build in layers)

| Layer | Add |
|-------|-----|
| A | Get batch_meta; early return if missing |
| B | `read_component_latencies_ms` → `note_ds_latencies` |
| C | Walk objects: `n_obj`, `by_class`, `ids` |
| D | `note_osd_frame`, `hops_for_frame` |
| E | Build record dict; `log_sink.emit`; `note_probe_log_us` |

**Test each layer** with `--run-name learn_osd_A` etc.

---

## Substep 7.8 — `build_pipeline(...)` graph only (fakesink)

```text
nvurisrcbin → nvstreammux (batch=1) → nvinfer → nvtracker
  → nvvideoconvert → nvdsosd → fakesink
```

Properties: mux 1920×1080, tracker 960×544, tracker lib paths.

**No probes yet** — verify EOS.

---

## Substep 7.9 — Add stage probes in `build_pipeline`

| Pad | Probe |
|-----|-------|
| mux src | `stage_frame_probe("mux")` |
| pgie src | `stage_frame_probe("pgie")` |
| trk src | `stage_frame_probe("trk")` |

Return `(pipeline, osd_sink_pad)` for OSD probe in main.

---

## Substep 7.10 — `main()` wiring

1. Parse args: input, `--log-mode`, `--run-name`
2. `AsyncLogger` or `ProbeSyncLogger`
3. `StageMetrics()`, `GpuSampler(emit=...)`
4. `build_pipeline`; `osd_sink.add_probe(osd_sink_pad_probe, ...)`
5. `gpu.start()`, `MainLoop`, PLAYING
6. On shutdown: `gpu.stop()`, `log_sink.close()`, `summary()` → JSON file

---

## Substep 7.11 — Swap fakesink → filesink encode path

Add: conv_enc, caps NV12, nvv4l2h264enc, h264parse, qtmux, filesink.  
Add enc sink probe `stage_frame_probe("enc")`.

**Test:**

```sh
python3 learn/src/pipeline/tracker_single.py --run-name learn_n1_mp4
```

---

## Checkpoint

```sh
python3 learn/src/pipeline/tracker_single.py --run-name learn_single
cat output/logs/learn_single_metrics.json | head -30
```

Expect: `frames_mux == frames_osd`, JSONL with `type: frame`, `hops_ms`, optional `ds_ms`.

Next: [08-pipeline_tracker_multisrc.md](08-pipeline_tracker_multisrc.md)
