#!/usr/bin/env python3
"""
DeepStream 8 tracker (Gst-Python + pyds).

  nvurisrcbin --vsrc_0--> nvstreammux --> nvinfer (YOLO26)
       --> nvtracker (NvDCF accuracy + ResNet50 Market-1501 ReID)
       --> nvvideoconvert --> nvdsosd --> nvvideoconvert --> NV12
       --> nvv4l2h264enc --> h264parse --> qtmux --> filesink

OSD text: class label, detector confidence, tracker object id.
"""

from __future__ import annotations

import argparse
import json
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

import pyds  # noqa: E402

from common.async_logger import AsyncLogger, ProbeSyncLogger
from common.ds_latency import read_component_latencies_ms, stage_frame_probe
from common.gpu_sampler import GpuSampler
from common.mlflow_run import log_finish, parse_nvinfer_ini, start_run
from common.stage_metrics import StageMetrics

SAMPLE_URI = (
    "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
)
DEFAULT_PGIE = "/workspace/configs/pgie_yolo26.txt"
DEFAULT_TRACKER = "/workspace/configs/config_tracker_NvDCF_accuracy.yml"
DEFAULT_OUTPUT = "/workspace/output/tracker_annotated.mp4"
TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
UNTRACKED = getattr(pyds, "UNTRACKED_OBJECT_ID", 0xFFFFFFFFFFFFFFFF)


def maybe_reexec_nsys() -> None:
    if "--profile" not in sys.argv:
        return
    sys.argv = [a for a in sys.argv if a != "--profile"]
    if os.environ.get("NSYS_ACTIVE") == "1":
        return
    out = f"/workspace/output/nsys/tracker_{int(time.time())}"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    os.environ["NSYS_ACTIVE"] = "1"
    os.environ["NSYS_REP"] = out
    os.execvp(
        "nsys",
        [
            "nsys",
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--force-overwrite=true",
            f"--output={out}",
            sys.executable,
            *sys.argv,
        ],
    )


def make_element(factory: str, name: str) -> Gst.Element:
    element = Gst.ElementFactory.make(factory, name)
    if not element:
        raise RuntimeError(f"Failed to create {factory} ({name})")
    return element


def to_uri(path_or_uri: str) -> str:
    if path_or_uri.startswith(("file://", "http://", "https://", "rtsp://")):
        return path_or_uri
    return Path(path_or_uri).resolve().as_uri()


def count_probe(stage: str, metrics: StageMetrics):
    def _probe(pad, info, user_data):
        metrics.bump(stage)
        return Gst.PadProbeReturn.OK

    return _probe


def on_src_pad_added(src, pad, data):
    streammux, metrics = data
    pad_name = pad.get_name()
    if not pad_name.startswith("vsrc"):
        print(f"Ignoring non-video pad {pad_name}")
        return
    sinkpad = streammux.request_pad_simple("sink_0")
    if sinkpad is None:
        print("Could not request nvstreammux sink_0")
        return
    if pad.link(sinkpad) != Gst.PadLinkReturn.OK:
        print(f"Failed to link {pad_name} -> streammux.sink_0")
        return
    pad.add_probe(Gst.PadProbeType.BUFFER, count_probe("src", metrics), None)
    print(f"Linked {pad_name} -> streammux.sink_0")


def _annotate_object(obj_meta) -> None:
    label = obj_meta.obj_label or str(obj_meta.class_id)
    conf = obj_meta.confidence
    oid = obj_meta.object_id
    oid_s = "-" if oid == UNTRACKED else str(oid)
    conf_s = f"{conf:.2f}" if 0.0 <= conf <= 1.0 else "n/a"

    txt = obj_meta.text_params
    txt.display_text = f"{label} {conf_s} id:{oid_s}"
    txt.x_offset = int(obj_meta.rect_params.left)
    txt.y_offset = max(0, int(obj_meta.rect_params.top) - 18)
    txt.font_params.font_name = "Serif"
    txt.font_params.font_size = 12
    txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
    txt.set_bg_clr = 1
    txt.text_bg_clr.set(0.0, 0.0, 0.0, 0.65)

    rect = obj_meta.rect_params
    if rect.border_width < 2:
        rect.border_width = 2


def osd_sink_pad_probe(pad, info, user_data):
    metrics, log_sink, log_mode = user_data
    silent = bool(getattr(log_sink, "silent", False))
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    ds_ms = read_component_latencies_ms(batch_meta)
    if ds_ms:
        metrics.note_ds_latencies(ds_ms)

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        by_class: dict[str, int] = {}
        n_obj = 0
        ids: list[int] = []
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            n_obj += 1
            if not silent:
                _annotate_object(obj_meta)
            label = obj_meta.obj_label or str(obj_meta.class_id)
            by_class[label] = by_class.get(label, 0) + 1
            if obj_meta.object_id != UNTRACKED:
                ids.append(int(obj_meta.object_id))
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        stream_id = int(getattr(frame_meta, "pad_index", frame_meta.source_id))
        e2e_ms, hops = metrics.note_osd_frame(
            frame_meta.frame_num, n_obj, by_class, stream_id=stream_id
        )
        hops = metrics.hops_for_frame(frame_meta.frame_num, stream_id=stream_id) or hops

        record = {
            "type": "frame",
            "source_id": stream_id,
            "frame": frame_meta.frame_num,
            "n_obj": n_obj,
            "ids": ids,
            "e2e_ms": round(e2e_ms, 3),
            "hops_ms": {k: round(v, 3) for k, v in hops.items()},
            "ds_ms": {k: round(v, 3) for k, v in ds_ms.items()},
            "log_mode": log_mode,
        }
        t0 = time.perf_counter()
        if not silent:
            log_sink.emit(record)
        metrics.note_probe_log_us((time.perf_counter() - t0) * 1e6)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def bus_call(bus, message, loop):
    msg_type = message.type
    if msg_type == Gst.MessageType.EOS:
        print("End of stream")
        loop.quit()
    elif msg_type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"ERROR: {err} ({debug})")
        loop.quit()
    return True


def build_pipeline(
    uri: str,
    output_path: str,
    pgie_config: str,
    tracker_config: str,
    metrics: StageMetrics,
):
    pipeline = Gst.Pipeline.new("ds8-file-tracker")

    source = make_element("nvurisrcbin", "source")
    streammux = make_element("nvstreammux", "streammux")
    pgie = make_element("nvinfer", "pgie")
    tracker = make_element("nvtracker", "tracker")
    conv_osd = make_element("nvvideoconvert", "conv_osd")
    osd = make_element("nvdsosd", "osd")
    conv_enc = make_element("nvvideoconvert", "conv_enc")
    capsfilter = make_element("capsfilter", "enc_caps")
    encoder = make_element("nvv4l2h264enc", "encoder")
    parser = make_element("h264parse", "parser")
    mux = make_element("qtmux", "qtmux")
    sink = make_element("filesink", "sink")

    source.set_property("uri", uri)
    streammux.set_property("batch-size", 1)
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batched-push-timeout", 33000)
    streammux.set_property("live-source", 0)
    pgie.set_property("config-file-path", pgie_config)

    tracker.set_property("ll-lib-file", TRACKER_LIB)
    tracker.set_property("ll-config-file", tracker_config)
    tracker.set_property("tracker-width", 960)
    tracker.set_property("tracker-height", 544)
    tracker.set_property("gpu-id", 0)
    tracker.set_property("display-tracking-id", 1)

    osd.set_property("display-text", 1)
    osd.set_property("display-bbox", 1)

    capsfilter.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
    )
    encoder.set_property("bitrate", 4000000)
    sink.set_property("location", output_path)
    sink.set_property("sync", False)
    sink.set_property("async", False)

    for element in (
        source,
        streammux,
        pgie,
        tracker,
        conv_osd,
        osd,
        conv_enc,
        capsfilter,
        encoder,
        parser,
        mux,
        sink,
    ):
        pipeline.add(element)

    source.connect("pad-added", on_src_pad_added, (streammux, metrics))

    if not streammux.link(pgie):
        raise RuntimeError("link streammux -> pgie failed")
    if not pgie.link(tracker):
        raise RuntimeError("link pgie -> tracker failed")
    if not tracker.link(conv_osd):
        raise RuntimeError("link tracker -> conv_osd failed")
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

    mux_src = streammux.get_static_pad("src")
    pgie_src = pgie.get_static_pad("src")
    trk_src = tracker.get_static_pad("src")
    osd_sink = osd.get_static_pad("sink")
    enc_sink = encoder.get_static_pad("sink")
    if not all((mux_src, pgie_src, trk_src, osd_sink, enc_sink)):
        raise RuntimeError("missing static pad for stage probes")

    mux_src.add_probe(Gst.PadProbeType.BUFFER, stage_frame_probe("mux", metrics), None)
    pgie_src.add_probe(Gst.PadProbeType.BUFFER, stage_frame_probe("pgie", metrics), None)
    trk_src.add_probe(Gst.PadProbeType.BUFFER, stage_frame_probe("trk", metrics), None)
    enc_sink.add_probe(Gst.PadProbeType.BUFFER, stage_frame_probe("enc", metrics), None)

    return pipeline, osd_sink


def main() -> int:
    maybe_reexec_nsys()

    parser = argparse.ArgumentParser(
        description="DS8 YOLO26 + NvDCF accuracy tracker -> annotated MP4"
    )
    parser.add_argument("input", nargs="?", default=SAMPLE_URI)
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--config", default=DEFAULT_PGIE)
    parser.add_argument("--tracker-config", default=DEFAULT_TRACKER)
    parser.add_argument("--log-mode", choices=("async", "probe"), default="async")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-video", action="store_true")
    args = parser.parse_args()

    Gst.init(None)

    uri = to_uri(args.input)
    output_path = args.output
    config_path = args.config
    tracker_config = args.tracker_config
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path("/workspace/models").mkdir(parents=True, exist_ok=True)
    Path("/workspace/output/logs").mkdir(parents=True, exist_ok=True)

    missing = []
    if not Path(config_path).is_file():
        missing.append(config_path)
    if not Path(tracker_config).is_file():
        missing.append(tracker_config)
    if not Path("/workspace/models/yolo26m.onnx").is_file():
        missing.append("/workspace/models/yolo26m.onnx (run ./setup_tracker.sh)")
    if not Path("/workspace/models/Tracker/resnet50_market1501.etlt").is_file():
        missing.append(
            "/workspace/models/Tracker/resnet50_market1501.etlt (run ./setup_tracker.sh)"
        )
    parser_so = "/workspace/parsers/yolo26/libnvdsinfer_custom_impl_Yolo26.so"
    if not Path(parser_so).is_file():
        missing.append(f"{parser_so} (run ./setup_tracker.sh)")
    if missing:
        print("Missing required files:")
        for item in missing:
            print(f"  {item}")
        return 1

    run_name = args.run_name or f"tracker_{args.log_mode}_{int(time.time())}"
    log_path = f"/workspace/output/logs/{run_name}.jsonl"
    metrics_json = f"/workspace/output/logs/{run_name}_metrics.json"

    if args.log_mode == "probe":
        log_sink = ProbeSyncLogger(log_path)
    else:
        log_sink = AsyncLogger(log_path)

    metrics = StageMetrics()
    gpu = GpuSampler(emit=log_sink.emit if args.log_mode == "async" else None)

    nvinfer_params = parse_nvinfer_ini(config_path)
    params = {
        "pipeline": "tracker",
        "detector": "yolo26m",
        "tracker": "NvDCF_accuracy",
        "reid": "resnet50_market1501",
        "log_mode": args.log_mode,
        "input": uri,
        "output": output_path,
        "config": config_path,
        "tracker_config": tracker_config,
        "mux_width": 1920,
        "mux_height": 1080,
        "tracker_width": 960,
        "tracker_height": 544,
        "encoder_bitrate": 4000000,
        **{f"nvinfer_{k}": v for k, v in nvinfer_params.items()},
    }
    if os.environ.get("NSYS_REP"):
        params["nsys_rep"] = os.environ["NSYS_REP"]

    mlflow_mod = None
    try:
        mlflow_mod, _run = start_run(run_name, params, experiment="tracker")
    except Exception as exc:
        print(f"MLflow disabled: {exc}")

    print(f"Input : {uri}")
    print(f"Output: {output_path}")
    print(f"PGIE  : {config_path}")
    print(f"Trk   : {tracker_config}")
    print(f"log   : mode={args.log_mode} file={log_path}")

    pipeline, osd_sink = build_pipeline(
        uri, output_path, config_path, tracker_config, metrics
    )
    osd_sink.add_probe(
        Gst.PadProbeType.BUFFER,
        osd_sink_pad_probe,
        (metrics, log_sink, args.log_mode),
    )

    gpu.start()

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

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
    summary["log_dropped"] = getattr(log_sink, "dropped", 0)
    summary["log_mode"] = args.log_mode
    Path(metrics_json).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    artifacts = [config_path, tracker_config, metrics_json]
    if args.log_video and Path(output_path).is_file():
        artifacts.append(output_path)

    if mlflow_mod is not None:
        log_finish(mlflow_mod, summary, artifacts, metrics_json)

    if Path(output_path).exists():
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"Wrote {output_path} ({size_mb:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
