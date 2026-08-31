# File 8 — `pipeline_tracker_multisrc.py`

**Write:** `learn/src/app/run_multisrc.py`  
**Reference:** `pipeline_tracker_multisrc.py`  
**Dependencies:** step 7 + `pgie_batch`

---

## Substep 8.1 — `NullLogger` class

`silent = True`, `emit` no-op, `close` no-op, `dropped = 0`.

For `--no-log-frames` capacity runs.

---

## Substep 8.2 — `on_src_pad_added` (multisource)

Data: `(streammux, metrics, index)`.

Link `vsrc_*` → `sink_{index}` (request pad by index).  
Probe: `count_probe("src", metrics)` on each source pad.

**Test:** N=2 prints two link lines.

---

## Substep 8.3 — `tile_layout(n)`

`cols = ceil(sqrt(n))`, `rows = ceil(n/cols)` for tiler.

**Test:** n=4 → 2×2; n=8 → 3×3 grid math.

---

## Substep 8.4 — `build_pipeline` core (fake sink)

Elements:

- `N × nvurisrcbin` same URI
- `nvstreammux` batch=N
- `nvinfer` batch=N
- `nvtracker` + `enable-batch-process=1`
- `conv_osd → nvdsosd → fakesink`

**Test:** `-n 2 --sink fake` EOS, no error.

---

## Substep 8.5 — PGIE config for batch N

```python
pgie_config = write_pgie_config(src, n, dest_path)
```

Before PLAYING.

---

## Substep 8.6 — Stage probes (same as single + optional enc)

mux, pgie, trk src probes; enc only if file sink.

---

## Substep 8.7 — `build_pipeline` file sink branch

When `sink_mode == "file"`:

```text
tracker → nvmultistreamtiler → conv_osd → osd
  → conv_enc → caps → encoder → parser → qtmux → filesink
```

Tiler rows/cols from `tile_layout`.

---

## Substep 8.8 — `required_files(config, tracker_config, num_streams)`

Check pgie config, tracker yaml, onnx/dynamic onnx, parser .so, reid etlt.

---

## Substep 8.9 — `main()` argparse

Flags: `-n`, `--sink fake|file`, `--log-mode`, `--no-log-frames`, `--run-name`, `-o`.

Logger selection logic (NullLogger + gpu-only AsyncLogger when no-log-frames).

---

## Substep 8.10 — `main()` metrics summary extras

```python
summary["num_streams"] = n
summary["fps_per_stream"] = fps_e2e / n
summary["sink"] = args.sink
```

File sink: optional FPS from `frames_trk`.

---

## Substep 8.11 — MLflow experiment `tracker_n`

`start_run(..., experiment="tracker_n")`, `log_finish` with pgie config artifact.

---

## Substep 8.12 — Failure check

Return 1 if bus ERROR or `frames_osd == 0`.

---

## Checkpoint matrix

Run inside `ds8-dev`:

```sh
cd /workspace
python3 learn/src/app/run_multisrc.py -n 1 --sink fake --run-name learn_n1
python3 learn/src/app/run_multisrc.py -n 2 --sink fake --run-name learn_n2
python3 learn/src/app/run_multisrc.py -n 4 --sink fake --no-log-frames --run-name learn_n4_cap
python3 learn/src/app/run_multisrc.py -n 2 --sink file -o /workspace/output/learn_n2.mp4 --run-name learn_n2_file
```

**Assertions:**

| Run | Check |
|-----|-------|
| n1 | `frames_osd > 0`, hops in JSONL |
| n2 | JSONL has `source_id` 0 and 1 |
| n4 cap | metrics JSON only, no frame JSONL |
| n2 file | MP4 exists, `frames_enc > 0` |

---

## Parity with production

```sh
./run_tracker_multisrc.sh -n 2 --sink fake --run-name ref_n2
diff output/logs/learn_n2_metrics.json output/logs/ref_n2_metrics.json
```

Structure should match; latencies will differ slightly.

---

## Done

You have modular metrics + full multisrc pipeline. See also `metrics.md` in repo root for metric semantics.
