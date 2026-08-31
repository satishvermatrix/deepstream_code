#!/usr/bin/env python3
"""
DeepStream 8 multi-source tracker (Gst-Python + pyds).

Repeats the same URI N times to measure how YOLO26m + NvDCF scales on one GPU.

  N x nvurisrcbin --vsrc_0--> nvstreammux (batch=N)
       --> nvinfer (YOLO26, batch=N)
       --> nvtracker (NvDCF accuracy + Market-1501 ReID)
       --> nvvideoconvert --> nvdsosd --> fakesink   (default, capacity)
  optional: nvmultistreamtiler --> NVENC --> MP4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NVDS_ENABLE_LATENCY_MEASUREMENT", "1")
os.environ.setdefault("NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT", "1")
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:/workspace/mlruns")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from common.async_logger import AsyncLogger, ProbeSyncLogger
from common.ds_latency import stage_frame_probe
from common.gpu_sampler import GpuSampler
from common.mlflow_run import log_finish, parse_nvinfer_ini, start_run
from common.pgie_batch import ONNX_DYNAMIC, write_pgie_config
from common.stage_metrics import StageMetrics
from pipeline_tracker import (
    DEFAULT_PGIE,
    DEFAULT_TRACKER,
    SAMPLE_URI,
    TRACKER_LIB,
    bus_call,
    count_probe,
    make_element,
    maybe_reexec_nsys,
    osd_sink_pad_probe,
    to_uri,
)

DEFAULT_OUTPUT = "/workspace/output/tracker_multisrc.mp4"


class NullLogger:
    dropped = 0
    silent = True

    def emit(self, _record) -> None:
        return

    def close(self) -> None:
        return


def on_src_pad_added(src, pad, data):
    streammux, metrics, index = data
    pad_name = pad.get_name()
    if not pad_name.startswith("vsrc"):
        print(f"Ignoring non-video pad {pad_name}")
        return
    sink_name = f"sink_{index}"
    sinkpad = streammux.request_pad_simple(sink_name)
    if sinkpad is None:
        templ = streammux.get_pad_template("sink_%u")
        if templ is not None:
            sinkpad = streammux.request_pad(templ, sink_name, None)
    if sinkpad is None:
        print(f"Could not request nvstreammux {sink_name}")
        return
    if pad.link(sinkpad) != Gst.PadLinkReturn.OK:
        print(f"Failed to link {pad_name} -> {sinkpad.get_name()}")
        return
    pad.add_probe(Gst.PadProbeType.BUFFER, count_probe("src", metrics), None)
    print(f"Linked {src.get_name()}.{pad_name} -> streammux.{sinkpad.get_name()}")


def tile_layout(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return int(rows), int(cols)


def build_pipeline(
    uri: str,
    num_streams: int,
    pgie_config: str,
    tracker_config: str,
    metrics: StageMetrics,
    sink_mode: str,
    output_path: str,
):
    pipeline = Gst.Pipeline.new("ds8-tracker-multisrc")

    streammux = make_element("nvstreammux", "streammux")
    pgie = make_element("nvinfer", "pgie")
    tracker = make_element("nvtracker", "tracker")
    conv_osd = make_element("nvvideoconvert", "conv_osd")
    osd = make_element("nvdsosd", "osd")

    streammux.set_property("batch-size", num_streams)
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batched-push-timeout", 33000)
    streammux.set_property("live-source", 0)
    pgie.set_property("config-file-path", pgie_config)
    pgie.set_property("batch-size", num_streams)

    tracker.set_property("ll-lib-file", TRACKER_LIB)
    tracker.set_property("ll-config-file", tracker_config)
    tracker.set_property("tracker-width", 960)
    tracker.set_property("tracker-height", 544)
    tracker.set_property("gpu-id", 0)
    tracker.set_property("display-tracking-id", 1)
    if tracker.find_property("enable-batch-process"):
        tracker.set_property("enable-batch-process", 1)

    osd.set_property("display-text", 1)
    osd.set_property("display-bbox", 1)

    sources = []
    extra = min(max(num_streams, 1), 5)
    for i in range(num_streams):
        src = make_element("nvurisrcbin", f"source_{i}")
        src.set_property("uri", uri)
        if src.find_property("num-extra-surfaces"):
            src.set_property("num-extra-surfaces", extra)
        if src.find_property("file-loop"):
            src.set_property("file-loop", 0)
        sources.append(src)

    elements = [*sources, streammux, pgie, tracker]
    tiler = None
    encoder = None

    if sink_mode == "file":
        tiler = make_element("nvmultistreamtiler", "tiler")
        rows, cols = tile_layout(num_streams)
        tiler.set_property("rows", rows)
        tiler.set_property("columns", cols)
        tiler.set_property("width", 1920)
        tiler.set_property("height", 1080)
        conv_enc = make_element("nvvideoconvert", "conv_enc")
        capsfilter = make_element("capsfilter", "enc_caps")
        encoder = make_element("nvv4l2h264enc", "encoder")
        parser = make_element("h264parse", "parser")
        mux = make_element("qtmux", "qtmux")
        sink = make_element("filesink", "sink")
        capsfilter.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
        )
        encoder.set_property("bitrate", 4000000)
        sink.set_property("location", output_path)
        sink.set_property("sync", False)
        sink.set_property("async", False)
        elements.extend(
            [tiler, conv_osd, osd, conv_enc, capsfilter, encoder, parser, mux, sink]
        )
    else:
        sink = make_element("fakesink", "sink")
        sink.set_property("sync", False)
        sink.set_property("async", False)
        sink.set_property("enable-last-sample", False)
        elements.extend([conv_osd, osd, sink])

    for element in elements:
        pipeline.add(element)

    for i, src in enumerate(sources):
        src.connect("pad-added", on_src_pad_added, (streammux, metrics, i))

    if not streammux.link(pgie):
        raise RuntimeError("link streammux -> pgie failed")
    if not pgie.link(tracker):
        raise RuntimeError("link pgie -> tracker failed")
    if sink_mode == "file":
        if not tracker.link(tiler):
            raise RuntimeError("link tracker -> tiler failed")
        if not tiler.link(conv_osd):
            raise RuntimeError("link tiler -> conv_osd failed")
        if not conv_osd.link(osd):
            raise RuntimeError("link conv_osd -> osd failed")
        if not osd.link(conv_enc):
            raise RuntimeError("link osd -> conv_enc failed")
        if not conv_enc.link(capsfilter):
            raise RuntimeError("link conv_enc -> caps failed")
        if not capsfilter.link(encoder):
            raise RuntimeError("link caps -> encoder failed")
        if not encoder.link(parser):
            raise RuntimeError("link encoder -> parser failed")
        if not parser.link(mux):
            raise RuntimeError("link parser -> qtmux failed")
        if not mux.link(sink):
            raise RuntimeError("link qtmux -> filesink failed")
    else:
        if not tracker.link(conv_osd):
            raise RuntimeError("link tracker -> conv_osd failed")
        if not conv_osd.link(osd):
            raise RuntimeError("link conv_osd -> osd failed")
        if not osd.link(sink):
            raise RuntimeError("link osd -> fakesink failed")

    mux_src = streammux.get_static_pad("src")
    pgie_src = pgie.get_static_pad("src")
    trk_src = tracker.get_static_pad("src")
    osd_sink = osd.get_static_pad("sink")
    if not all((mux_src, pgie_src, trk_src, osd_sink)):
        raise RuntimeError("missing static pad for stage probes")

    mux_src.add_probe(Gst.PadProbeType.BUFFER, stage_frame_probe("mux", metrics), None)
    pgie_src.add_probe(Gst.PadProbeType.BUFFER, stage_frame_probe("pgie", metrics), None)
    trk_src.add_probe(Gst.PadProbeType.BUFFER, stage_frame_probe("trk", metrics), None)
    if encoder is not None:
        enc_sink = encoder.get_static_pad("sink")
        if enc_sink:
            enc_sink.add_probe(
                Gst.PadProbeType.BUFFER, stage_frame_probe("enc", metrics), None
            )

    return pipeline, osd_sink


def required_files(config_path: str, tracker_config: str, num_streams: int) -> list[str]:
    missing = []
    if not Path(config_path).is_file():
        missing.append(config_path)
    if not Path(tracker_config).is_file():
        missing.append(tracker_config)
    if num_streams > 1 and not ONNX_DYNAMIC.is_file():
        missing.append(
            f"{ONNX_DYNAMIC} (run ./setup_tracker.sh — exports dynamic-batch YOLO)"
        )
    elif not Path("/workspace/models/yolo26m.onnx").is_file():
        missing.append("/workspace/models/yolo26m.onnx (run ./setup_tracker.sh)")
    if not Path("/workspace/models/Tracker/resnet50_market1501.etlt").is_file():
        missing.append(
            "/workspace/models/Tracker/resnet50_market1501.etlt (run ./setup_tracker.sh)"
        )
    parser_so = "/workspace/parsers/yolo26/libnvdsinfer_custom_impl_Yolo26.so"
    if not Path(parser_so).is_file():
        missing.append(f"{parser_so} (run ./setup_tracker.sh)")
    return missing


def main() -> int:
    maybe_reexec_nsys()

    parser = argparse.ArgumentParser(
        description="DS8 YOLO26 + NvDCF, N copies of the same URI"
    )
    parser.add_argument("input", nargs="?", default=SAMPLE_URI)
    parser.add_argument("-n", "--num-streams", type=int, default=2)
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--config", default=DEFAULT_PGIE)
    parser.add_argument("--tracker-config", default=DEFAULT_TRACKER)
    parser.add_argument("--sink", choices=("fake", "file"), default="fake")
    parser.add_argument("--log-mode", choices=("async", "probe"), default="async")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-video", action="store_true")
    parser.add_argument(
        "--no-log-frames",
        action="store_true",
        help="Skip per-frame JSONL (GPU samples still recorded in async mode)",
    )
    args = parser.parse_args()
    if args.num_streams < 1:
        print("--num-streams must be >= 1")
        return 1

    Gst.init(None)

    uri = to_uri(args.input)
    n = args.num_streams
    tracker_config = args.tracker_config
    Path("/workspace/models").mkdir(parents=True, exist_ok=True)
    Path("/workspace/output/logs").mkdir(parents=True, exist_ok=True)
    if args.sink == "file":
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    pgie_config = write_pgie_config(
        Path(args.config),
        n,
        Path(f"/workspace/output/logs/pgie_yolo26_b{n}.txt"),
    )

    missing = required_files(str(pgie_config), tracker_config, n)
    if missing:
        print("Missing required files:")
        for item in missing:
            print(f"  {item}")
        return 1

    run_name = args.run_name or f"tracker_n{n}_{args.sink}_{int(time.time())}"
    log_path = f"/workspace/output/logs/{run_name}.jsonl"
    metrics_json = f"/workspace/output/logs/{run_name}_metrics.json"

    if args.no_log_frames:
        log_sink = NullLogger()
        gpu_emit = None
        if args.log_mode == "async":
            gpu_logger = AsyncLogger(log_path)
            gpu_emit = gpu_logger.emit
            log_sink = gpu_logger
    elif args.log_mode == "probe":
        log_sink = ProbeSyncLogger(log_path)
        gpu_emit = None
    else:
        log_sink = AsyncLogger(log_path)
        gpu_emit = log_sink.emit

    metrics = StageMetrics()
    gpu = GpuSampler(emit=gpu_emit)

    nvinfer_params = parse_nvinfer_ini(str(pgie_config))
    params = {
        "pipeline": "tracker_multisrc",
        "detector": "yolo26m",
        "tracker": "NvDCF_accuracy",
        "reid": "resnet50_market1501",
        "num_streams": n,
        "sink": args.sink,
        "log_mode": args.log_mode,
        "input": uri,
        "output": args.output if args.sink == "file" else "fakesink",
        "config": str(pgie_config),
        "tracker_config": tracker_config,
        "mux_width": 1920,
        "mux_height": 1080,
        "tracker_width": 960,
        "tracker_height": 544,
        **{f"nvinfer_{k}": v for k, v in nvinfer_params.items()},
    }

    mlflow_mod = None
    try:
        mlflow_mod, _run = start_run(run_name, params, experiment="tracker_n")
    except Exception as exc:
        print(f"MLflow disabled: {exc}")

    print(f"N     : {n} x {uri}")
    print(f"Sink  : {args.sink}")
    print(f"PGIE  : {pgie_config}")
    print(f"Trk   : {tracker_config}")
    print(f"log   : mode={args.log_mode} file={log_path}")

    pipeline, osd_sink = build_pipeline(
        uri, n, str(pgie_config), tracker_config, metrics, args.sink, args.output
    )
    osd_sink.add_probe(
        Gst.PadProbeType.BUFFER,
        osd_sink_pad_probe,
        (metrics, log_sink if not args.no_log_frames else NullLogger(), args.log_mode),
    )

    gpu.start()

    failed = []

    def _bus(bus, message, loop):
        if message.type == Gst.MessageType.ERROR:
            failed.append(True)
        return bus_call(bus, message, loop)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", _bus, loop)

    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print("Could not set pipeline to PLAYING")
        gpu.stop()
        log_sink.close()
        return 1

    try:
        loop.run()
    except KeyboardInterrupt:
        print("Interrupted")
        pipeline.send_event(Gst.Event.new_eos())
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        gpu.stop()
        log_sink.close()

    summary = metrics.summary()
    summary.update(gpu.summary())
    if args.sink == "file" and metrics.t_start and metrics.t_end:
        dt = metrics.t_end - metrics.t_start
        if dt > 0 and summary.get("frames_trk"):
            summary["fps_e2e"] = summary["frames_trk"] / dt
    summary["num_streams"] = n
    summary["fps_per_stream"] = summary["fps_e2e"] / n if n else 0.0
    summary["log_dropped"] = getattr(log_sink, "dropped", 0)
    summary["log_mode"] = args.log_mode
    summary["sink"] = args.sink
    Path(metrics_json).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    artifacts = [str(pgie_config), tracker_config, metrics_json]
    if args.log_video and args.sink == "file" and Path(args.output).is_file():
        artifacts.append(args.output)

    if mlflow_mod is not None:
        log_finish(mlflow_mod, summary, artifacts, metrics_json)

    if args.sink == "file" and Path(args.output).exists():
        size_mb = Path(args.output).stat().st_size / (1024 * 1024)
        print(f"Wrote {args.output} ({size_mb:.1f} MiB)")
    if failed or summary.get("frames_osd", 0) == 0:
        print("Pipeline produced no frames or ended with ERROR")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
