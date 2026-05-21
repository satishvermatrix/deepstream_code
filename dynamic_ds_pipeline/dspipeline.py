#!/usr/bin/env python3
"""
Simple DeepStream pipeline using nvmultiurisrcbin with the *new* nvstreammux
and the built-in REST server for dynamic stream add/remove.
 
Pipeline:
    nvmultiurisrcbin -> nvmultistreamtiler -> nvvideoconvert -> nvdsosd -> fakesink
 
- nvmultiurisrcbin bundles nvurisrcbin + nvstreammux + the REST server in one bin.
- The REST server listens on http://<ip-address>:<port>.
- Streams are added/removed at runtime by POSTing JSON to:
      POST http://<host>:<port>/api/v1/stream/add
      POST http://<host>:<port>/api/v1/stream/remove
 
Run in a terminal (NOT blocking a notebook cell):
    export USE_NEW_NVSTREAMMUX=yes
    python3 nvmultiurisrcbin_pipeline.py
 
Then add/remove streams from a notebook or another terminal (see rest_client.py
or the curl examples printed at startup).
"""
 
import os
# Must be set before the GStreamer plugins are loaded, so set it here as a
# safety net in addition to exporting it in the shell.
os.environ["USE_NEW_NVSTREAMMUX"] = "yes"
 
import sys
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
 
# ---- Configuration -------------------------------------------------------
REST_IP = "0.0.0.0"      # 0.0.0.0 so it is reachable from outside the container
REST_PORT = 9000
MAX_BATCH_SIZE = 4       # upper bound on simultaneous streams
MUX_WIDTH = 1280
MUX_HEIGHT = 720
 
# Start the pipeline with one stream so it prerolls cleanly; you can then
# add/remove more over REST. Set both to "" to start with no streams.
SAMPLE = "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
INITIAL_URI_LIST = ""
INITIAL_SENSOR_ID_LIST = "cam0"
# -------------------------------------------------------------------------
 
 
def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        # Do NOT quit. This is a long-running REST server: a finite (file)
        # source sending EOS must not tear down the pipeline, otherwise you
        # can never add streams afterwards. Stop with Ctrl+C instead.
        print("[bus] EOS (a source ended) — server still running. Ctrl+C to stop.")
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(f"[bus] WARNING: {err}: {debug}")
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"[bus] ERROR: {err}: {debug}")
        #loop.quit()
    return True
 
 
def make(factory, name):
    el = Gst.ElementFactory.make(factory, name)
    if not el:
        sys.exit(f"ERROR: could not create element '{factory}'. "
                 f"Run `gst-inspect-1.0 {factory}` to check it is installed.")
    return el
 
 
def main():
    Gst.init(None)
    pipeline = Gst.Pipeline.new("ds-multiuri-pipeline")
 
    # Source bin: integrates nvurisrcbin + new nvstreammux + REST server
    src = make("nvmultiurisrcbin", "multiuri-src")
    src.set_property("port", REST_PORT)
    src.set_property("ip-address", REST_IP)
    src.set_property("max-batch-size", MAX_BATCH_SIZE)
    src.set_property("width", MUX_WIDTH)
    src.set_property("file-loop", True)
    src.set_property("height", MUX_HEIGHT)
    if INITIAL_URI_LIST:
        src.set_property("uri-list", INITIAL_URI_LIST)
        src.set_property("sensor-id-list", INITIAL_SENSOR_ID_LIST)
 
    # Tile the batched output into a single surface
    tiler = make("nvmultistreamtiler", "tiler")
    tiler.set_property("rows", 2)
    tiler.set_property("columns", 2)
    tiler.set_property("width", MUX_WIDTH)
    tiler.set_property("height", MUX_HEIGHT)
 
    conv = make("nvvideoconvert", "converter")
    osd = make("nvdsosd", "onscreendisplay")
 
    # Headless-safe sink. If you have a display attached to the Spark,
    # swap fakesink for nv3dsink (and set sync=1).
    sink = make("fakesink", "sink")
    sink.set_property("sync", 0)
 
    for el in (src, tiler, conv, osd, sink):
        pipeline.add(el)
 
    # nvmultiurisrcbin exposes a single static src pad -> link directly
    if not src.link(tiler):
        sys.exit("ERROR: failed to link nvmultiurisrcbin -> tiler")
    tiler.link(conv)
    conv.link(osd)
    osd.link(sink)
 
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
 
    pipeline.set_state(Gst.State.PLAYING)
 
    print("=" * 70)
    print(f"Pipeline PLAYING. REST server on http://{REST_IP}:{REST_PORT}")
    print("Add a stream:")
    print(f"""  curl -XPOST 'http://localhost:{REST_PORT}/api/v1/stream/add' -d '{{
    "key": "sensor",
    "value": {{
      "camera_id": "cam1",
      "camera_name": "cam1",
      "camera_url": "{SAMPLE}",
      "change": "camera_add",
      "metadata": {{"resolution": "1920 x 1080", "codec": "h264", "framerate": 30}}
    }},
    "headers": {{"source": "curl", "created_at": "2024-01-01T00:00:00.000Z"}}
  }}'""")
    print("Remove a stream: same payload, endpoint .../stream/remove, "
          'change="camera_remove"')
    print("=" * 70)
 
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nInterrupted, shutting down...")
    finally:
        pipeline.set_state(Gst.State.NULL)
 
 
if __name__ == "__main__":
    main()

