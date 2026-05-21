#!/usr/bin/env python3
"""
REST client helpers for nvmultiurisrcbin stream add/remove.
 
Use from a Jupyter cell while nvmultiurisrcbin_pipeline.py is running:
 
    from rest_client import add_stream, remove_stream, SAMPLE
    add_stream("cam1", SAMPLE)
    add_stream("cam2", SAMPLE)
    remove_stream("cam1", SAMPLE)
"""
 
import datetime
import requests
 
HOST = "localhost"
PORT = 9000
BASE = f"http://{HOST}:{PORT}/api/v1"
 
SAMPLE = "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
 
 
def _payload(camera_id, url, change):
    return {
        "key": "sensor",
        "value": {
            "camera_id": camera_id,
            "camera_name": camera_id,
            "camera_url": url,
            "change": change,  # "camera_add" or "camera_remove"
            "metadata": {
                "resolution": "1920 x 1080",
                "codec": "h264",
                "framerate": 30,
            },
        },
        "headers": {
            "source": "notebook",
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        },
    }
 
 
def add_stream(camera_id, url=SAMPLE):
    r = requests.post(f"{BASE}/stream/add", json=_payload(camera_id, url, "camera_add"))
    print(f"add {camera_id}: {r.status_code} {r.text}")
    return r
 
 
def remove_stream(camera_id, url=SAMPLE):
    r = requests.post(f"{BASE}/stream/remove", json=_payload(camera_id, url, "camera_remove"))
    print(f"remove {camera_id}: {r.status_code} {r.text}")
    return r
 

