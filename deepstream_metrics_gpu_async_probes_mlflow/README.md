# DeepStream 8 detector (Cursor on host, run in image)

Edit files in this folder in Cursor. Run them inside the `ds8-dev` container (DeepStream 8.0 on GPU). This folder is bind-mounted at `/workspace` in the container — saves are visible immediately.

Do **not** run `pipeline_detector.py` with host Python.

## One-time setup

From this directory:

```sh
chmod +x dev.sh run.sh compare_log_modes.sh
./dev.sh up      # create/start container ds8-dev
./dev.sh setup   # install NVIDIA pyds + MLflow inside the image
```

Image: `nvcr.io/nvidia/deepstream:8.0-triton-dgx-spark`  
GPU: NVIDIA GB10 (this machine)

## Run the detector

```sh
./run.sh
```

Default input is the sample file already in the image:

`/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4`

Default output (on the host):

`output/detector_annotated.mp4`

First run can take several minutes while TensorRT builds the FP16 engine under `models/`. Later runs reuse that engine.

### Useful flags

```sh
# Custom video (must be under this folder so the container can see it)
./run.sh /workspace/my.mp4

# Output path
./run.sh -o /workspace/output/out.mp4

# Different nvinfer config
./run.sh --config /workspace/configs/pgie_config.txt

# Name the MLflow run
./run.sh --run-name trafficcam_fp16

# Attach the annotated MP4 as an MLflow artifact (large)
./run.sh --log-video
```

Use `./run.sh` or `sh run.sh` (not host `python3`).

## Probe vs async logging (latency A/B)

Pad probes run on the GStreamer streaming thread. Writing files there adds latency.

| Mode | What happens | Typical effect |
|---|---|---|
| `--log-mode probe` | JSONL write + flush **inside** the OSD probe | Higher `probe_log_us_avg` and `latency_e2e_ms_avg` |
| `--log-mode async` | Probe only enqueues; a background thread writes | Lower probe time / E2E latency |

Same video, same model:

```sh
./run.sh --log-mode probe --run-name log_probe
./run.sh --log-mode async --run-name log_async
./compare_log_modes.sh
```

Or one command that runs both then prints the table:

```sh
./compare_log_modes.sh --run
```

Compare **`probe_log_us_avg`** and **`latency_e2e_ms_avg`**. File playback with `sync=0` often keeps FPS similar; those two metrics are the fair comparison.

## Nsight Systems (optional)

Heavier; skip unless you want per-plugin NVTX:

```sh
./run.sh --profile --log-mode async
```

Reports land under `output/nsys/`.

## Outputs

| Path | Contents |
|---|---|
| `output/detector_annotated.mp4` | Overlay video |
| `output/logs/<run_name>.jsonl` | Per-frame (and GPU) records |
| `output/logs/<run_name>_metrics.json` | Run summary (FPS, stage counts, GPU, latency) |
| `models/*.engine` | TensorRT engine cache |
| `mlruns/` | Local MLflow file store |

## MLflow

Runs are stored in `mlruns/` on the host (`file:/workspace/mlruns` in the container). MLflow 3 needs `MLFLOW_ALLOW_FILE_STORE=true` (already set in the Python apps).

Browse UI from the container (bind a port if you want it in a browser):

```sh
./dev.sh shell
# inside:
mlflow ui --backend-store-uri file:/workspace/mlruns --host 0.0.0.0 --port 5000
```

To add a port when creating the container, stop and recreate with `-p 5000:5000`, or run `mlflow ui` on the host against `./mlruns` if you install MLflow locally.

Logged **params** include `log_mode`, nvinfer keys from the config, mux size, encoder bitrate.  
Logged **metrics** include `fps_e2e`, per-stage frame counts, `dropped_frames`, pad-to-pad hop latencies (`latency_hop_*_ms_avg`), DeepStream plugin latencies (`latency_ds_*_ms_avg`), `probe_log_us_avg`, GPU util / NVDEC / NVENC / memory.

Per-frame `hops_ms` and `ds_ms` are in the JSONL. Run averages also land in `output/logs/<run_name>_metrics.json`.

## Container helper

```sh
./dev.sh up       # start
./dev.sh status   # running? pyds?
./dev.sh shell    # bash in the image
./dev.sh down     # stop (keep container)
./dev.sh rm       # delete container
```

## Tracker pipeline (YOLO26 + NvDCF accuracy)

See **[tracker.md](tracker.md)** for how to run it and how it is implemented.

Short path:

```sh
./setup_tracker.sh    # first time
./run_tracker.sh
```

Output: `output/tracker_annotated.mp4`. Do **not** run `pipeline_tracker.py` with host Python.

## Layout

```
deepstream8/
  pipeline_detector.py    # Gst-Python + pyds detector (TrafficCamNet)
  pipeline_tracker.py     # YOLO26m + NvDCF accuracy + Market-1501 ReID
  pipeline_tracker_multisrc.py  # N copies of the same URI
  configs/pgie_config.txt
  configs/pgie_yolo26.txt
  configs/config_tracker_NvDCF_accuracy.yml
  parsers/yolo26/         # custom post-NMS bbox parser
  common/                 # async logger, stage metrics, GPU sampler, MLflow
  run.sh / run_tracker.sh / run_tracker_multisrc.sh / sweep_tracker_n.sh
  setup_tracker.sh        # models + parser
  compare_log_modes.sh
  dev.sh
```
