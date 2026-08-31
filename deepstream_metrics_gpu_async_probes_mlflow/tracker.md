# Tracker pipeline (YOLO26m + NvDCF)

Gst-Python + pyds app: detect with **YOLO26m**, track with **NvDCF accuracy** and **ResNet50 Market-1501 ReID**, write an annotated MP4.

Edit on the host in this folder. Run inside `ds8-dev` (`nvcr.io/nvidia/deepstream:8.0-triton-dgx-spark`). This folder is `/workspace` in the container.

Do **not** run `pipeline_tracker.py` with host Python.

## How to run

One-time container (shared with the detector app):

```sh
chmod +x dev.sh setup_tracker.sh run_tracker.sh
./dev.sh up
./dev.sh setup      # pyds + MLflow
```

One-time tracker assets (parser, YOLO26m ONNX, Market-1501 `.etlt`):

```sh
./setup_tracker.sh
```

That wrapper starts `ds8-dev` and runs `setup_tracker.py` in the image. It:

1. Compiles `parsers/yolo26/libnvdsinfer_custom_impl_Yolo26.so`
2. Downloads `models/Tracker/resnet50_market1501.etlt` (skipped if present)
3. Downloads `models/yolo26m.pt` from Ultralytics assets
4. Exports `models/yolo26m.onnx` (`[1, 300, 6]` end-to-end)

It does **not** build TensorRT engines and does **not** run the pipeline.

Run:

```sh
./run_tracker.sh
./run_tracker.sh --run-name yolo26m_nvdcf
```

Default input is the DeepStream sample `sample_1080p_h264.mp4`.  
Default output on the host: `output/tracker_annotated.mp4`.

The first `./run_tracker.sh` after a model change builds TensorRT engines (YOLO26m FP16 + ReID batch-100 FP16) under `models/`. Later runs reuse them.

### Flags

```sh
./run_tracker.sh --run-name yolo26m_nvdcf
./run_tracker.sh -o /workspace/output/out.mp4
./run_tracker.sh --config /workspace/configs/pgie_yolo26.txt
./run_tracker.sh --tracker-config /workspace/configs/config_tracker_NvDCF_accuracy.yml
./run_tracker.sh /workspace/my.mp4          # file must live in this folder
./run_tracker.sh --log-mode async            # default; probe = I/O on streaming thread
./run_tracker.sh --log-video                 # attach MP4 to MLflow
```

`./run_tracker.sh` is `docker exec ds8-dev python3 /workspace/pipeline_tracker.py …`.

## Pipeline

```
nvurisrcbin --vsrc_0--> nvstreammux (1920x1080, batch=1)
       --> nvinfer (YOLO26m, FP16)
       --> nvtracker (NvDCF accuracy, 960x544, Market-1501 ReID)
       --> nvvideoconvert --> nvdsosd
       --> nvvideoconvert --> NV12 --> nvv4l2h264enc
       --> h264parse --> qtmux --> filesink
```

`nvurisrcbin` has no static src pad. Video is linked in `pad-added` when `vsrc_0` appears.

OSD overlay per object: `class confidence id:<track_id>` plus the bbox. Example: `person 0.87 id:12`.

## Implementation

### App

`pipeline_tracker.py` builds the graph with Gst-Python, walks metadata with `pyds`, and (after EOS) writes metrics + MLflow. MLflow param `detector` is `yolo26m`.

| Piece | Role |
|---|---|
| `nvurisrcbin` | File/RTSP/HTTP; decoder is inside the bin (NVDEC) |
| `nvstreammux` | Batch of 1, 1080p, `live-source=0` |
| `nvinfer` | Primary detector (YOLO26m) |
| `nvtracker` | IDs + ReID re-association |
| `nvdsosd` | Draw bbox + text |
| `nvv4l2h264enc` | NVENC → MP4 (`sync=0`) |

### Detector (YOLO26m)

Config: `configs/pgie_yolo26.txt`.

| File | Role |
|---|---|
| `models/yolo26m.pt` | Ultralytics checkpoint (setup only) |
| `models/yolo26m.onnx` | End-to-end ONNX, output `[1, 300, 6]` = `x1,y1,x2,y2,conf,cls` at 640×640 |
| `models/yolo26m.onnx_b1_gpu0_fp16.engine` | TensorRT FP16, batch 1 |

YOLO26 is already post-NMS, so nvinfer uses `cluster-mode=4` (no extra NMS). Letterbox: `maintain-aspect-ratio=1`, `symmetric-padding=1`, `infer-dims=3;640;640`. Same custom parser as nano: `NvDsInferParseYolo26` in `parsers/yolo26/libnvdsinfer_custom_impl_Yolo26.so`. nvinfer scales boxes from 640 to the mux frame.

COCO 80 labels: `configs/labels_coco.txt`.

### Tracker (NvDCF accuracy + ReID)

Library: `/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so`.  
YAML: `configs/config_tracker_NvDCF_accuracy.yml` (DeepStream accuracy config; ReID paths remapped to `/workspace`).

| Setting | Value |
|---|---|
| Visual tracker | NvDCF VPI (ColorNames + HOG) |
| Re-association | on (`reidType: 2`) |
| ReID model | TAO `resnet50_market1501.etlt` |
| Engine | `models/Tracker/resnet50_market1501.etlt_b100_gpu0_fp16.engine` |
| Tracker size | 960×544 (multiple of 32) |
| `display-tracking-id` | 1 |

`object_id` is assigned here. The OSD probe formats that id into `text_params.display_text`.

### Metrics and logs

Pad probes stay cheap: count + timestamp + `put_nowait`. Disk and MLflow are off the streaming thread (`--log-mode async`).

Probes: `vsrc_0` (src count only), mux src, pgie src, tracker src, osd sink, encoder sink.

Two latency views, stored every frame:

1. **Pad hops** (`hops_ms`) — wall time between probes for the same `frame_num`: `mux_pgie`, `pgie_trk`, `trk_osd`, `mux_osd`. `osd_enc` is in the run summary only.
2. **DeepStream plugin times** (`ds_ms`) — `NvDsMetaCompLatency` on the buffer (decoder, mux, pgie, tracker). Env: `NVDS_ENABLE_LATENCY_MEASUREMENT` and `NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT`.

Averages (avg/min/max/p95) go to `<run_name>_metrics.json` and MLflow experiment **`tracker`**.

GPU sampler (~1 Hz `nvidia-smi`) writes `type: gpu` lines on the same JSONL in async mode.

## Outputs

Replace `<run_name>` with `--run-name` (default `tracker_async_<unix>`).

| Path | Contents |
|---|---|
| `output/tracker_annotated.mp4` | Overlay video |
| `output/logs/<run_name>.jsonl` | Per-frame `n_obj`, `ids`, `hops_ms`, `ds_ms`; GPU lines |
| `output/logs/<run_name>_metrics.json` | Run summary (FPS, hops, DS latency, GPU) |
| `models/yolo26m.onnx` | Detector ONNX |
| `models/yolo26m.onnx_b1_gpu0_fp16.engine` | YOLO TensorRT cache |
| `models/Tracker/*.engine` | ReID TensorRT cache |
| `mlruns/` | MLflow file store |

There is no file named `metrics.json`. The summary is always `<run_name>_metrics.json`. Latest YOLO26m run: `output/logs/yolo26m_nvdcf_metrics.json`.

JSONL frame record (trimmed):

```json
{
  "type": "frame",
  "frame": 200,
  "n_obj": 14,
  "ids": [29, 31, 34],
  "e2e_ms": 11.5,
  "hops_ms": {"mux_pgie": 3.9, "pgie_trk": 6.5, "trk_osd": 0.3, "mux_osd": 10.7},
  "ds_ms": {"nvv4l2decoder0": 55.0, "pgie": 3.9, "tracker": 6.8}
}
```

Frame 0 is usually a warmup spike (first YOLO / ReID batch). Use avg/p95, not max, to judge steady state.

### Sample 1080p (file, `sync=0`)

Same clip as the nano runs. YOLO26m finds more objects; PGIE is slower.

| | YOLO26n (earlier) | YOLO26m (`yolo26m_nvdcf`) |
|---|---|---|
| FPS (not live-stream) | ~107 | ~89 |
| Objects | ~11k | ~18k |
| `latency_ds_pgie_ms_avg` | ~2.0 ms | ~3.9 ms |
| `latency_ds_tracker_ms_avg` | ~6.4 ms | ~6.8 ms |

## Multi-stream scale (`pipeline_tracker_multisrc`)

Same YOLO26m + NvDCF graph, but **N copies of one URI** so you can see how FPS / GPU util / tracker latency move with N.

```sh
./setup_tracker.sh                 # also exports models/yolo26m_dyn.onnx
./run_tracker_multisrc.sh -n 4     # fakesink (capacity; no MP4)
./run_tracker_multisrc.sh -n 4 --sink file
./sweep_tracker_n.sh               # default N=1,2,4,8
./sweep_tracker_n.sh --streams 1,2,4
```

Default sink is `fakesink` so NVENC is not in the way. Mux + nvinfer `batch-size` equal N. First run at each N builds `yolo26m_dyn.onnx_bN_gpu0_fp16.engine` (slow). Metrics: `output/logs/tracker_n{N}_metrics.json`. Sweep JSON: `output/logs/tracker_n_sweep.json`.

`fps_e2e` is **total** frames/s across streams. `fps_per_stream = fps_e2e / N`. For live 30 fps cameras you want `fps_per_stream` well above 30 with `sync=0` (this file test).

## Layout

```
pipeline_tracker.py
setup_tracker.sh / setup_tracker.py   # parser + yolo26m.pt + ONNX export
run_tracker.sh
configs/pgie_yolo26.txt               # points at yolo26m.onnx
configs/config_tracker_NvDCF_accuracy.yml
configs/labels_coco.txt
parsers/yolo26/                       # NvDsInferParseYolo26 (shared n/s/m/l/x)
common/stage_metrics.py
common/ds_latency.py
common/async_logger.py
common/gpu_sampler.py
common/mlflow_run.py
models/yolo26m.pt .onnx .engine
models/Tracker/
output/logs/
```

## Swap model or tracker

- Different YOLO26 size (`n` / `s` / `l` / `x`): change the `.pt` URL and ONNX names in `setup_tracker.py`, then `onnx-file` / `model-engine-file` in `pgie_yolo26.txt`. Keep `cluster-mode=4` and `NvDsInferParseYolo26` if the output is still `[N,6]`.
- Different tracker YAML: `--tracker-config` (perf / max_perf / NvSORT). Accuracy + Market-1501 is the default described here.
- More streams: `pipeline_tracker_multisrc.py` / `./run_tracker_multisrc.sh -n N` (same file N times). Sweep: `./sweep_tracker_n.sh`.
