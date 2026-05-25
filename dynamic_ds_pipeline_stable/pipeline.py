#!/usr/bin/env python3
import os
os.environ["USE_NEW_NVSTREAMMUX"] = "yes"
import sys, gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

REST_IP, REST_PORT = "0.0.0.0", 9000
MAX_BATCH_SIZE = 4
MUX_WIDTH, MUX_HEIGHT = 640, 360      # low res -> much more stable for remove

def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("[bus] EOS — server still running.")
    elif t == Gst.MessageType.WARNING:
        err, dbg = message.parse_warning(); print(f"[bus] WARNING: {err}: {dbg}")
    elif t == Gst.MessageType.ERROR:
        err, dbg = message.parse_error(); print(f"[bus] ERROR: {err}: {dbg}")
    return True

def make(f, n):
    el = Gst.ElementFactory.make(f, n)
    if not el: sys.exit(f"ERROR: cannot create '{f}'")
    return el

def main():
    Gst.init(None)
    pipeline = Gst.Pipeline.new("ds-multiuri-pipeline")
    src = make("nvmultiurisrcbin", "multiuri-src")
    src.set_property("port", REST_PORT)
    src.set_property("ip-address", REST_IP)
    src.set_property("max-batch-size", MAX_BATCH_SIZE)
    src.set_property("width", MUX_WIDTH)
    src.set_property("height", MUX_HEIGHT)
    src.set_property("batched-push-timeout", 33333)
    src.set_property("drop-pipeline-eos", True)
    tiler = make("nvmultistreamtiler", "tiler")
    tiler.set_property("rows", 2); tiler.set_property("columns", 2)
    tiler.set_property("width", MUX_WIDTH); tiler.set_property("height", MUX_HEIGHT)
    conv = make("nvvideoconvert", "converter")
    osd = make("nvdsosd", "onscreendisplay")
    sink = make("fakesink", "sink")
    sink.set_property("sync", 1)
    for el in (src, tiler, conv, osd, sink): pipeline.add(el)
    if not src.link(tiler): sys.exit("ERROR: link failed")
    tiler.link(conv); conv.link(osd); osd.link(sink)
    loop = GLib.MainLoop()
    bus = pipeline.get_bus(); bus.add_signal_watch(); bus.connect("message", bus_call, loop)
    pipeline.set_state(Gst.State.PLAYING)
    print(f"PLAYING. REST http://{REST_IP}:{REST_PORT}")
    try: loop.run()
    except KeyboardInterrupt: print("\nStopping...")
    finally: pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()
