import time, requests
BASE = "http://localhost:9000/api/v1"
URI = "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
TIMEOUT = 30

def _post(ep, cam, change):
    p = {"key":"sensor","value":{"camera_id":cam,"camera_url":URI,"change":change}}
    try:
        r = requests.post(f"{BASE}/{ep}", json=p, timeout=TIMEOUT)
        return r.status_code, r.json().get("reason", r.text)
    except Exception as e:
        return "ERR", str(e)

def add(cam):
    c,r=_post("stream/add",cam,"camera_add"); print(f">>> ADD    {cam}: [{c}] {r}")
def remove(cam):
    c,r=_post("stream/remove",cam,"camera_remove"); print(f"<<< REMOVE {cam}: [{c}] {r}")

print("== single-stream churn (lowest load, most stable) ==")
for i in range(1,6):
    cam=f"cam{i}"; add(cam); time.sleep(4); remove(cam); time.sleep(2)

print("\n== up to 3 concurrent, removed while active ==")
for cam in ("camA","camB","camC"): add(cam); time.sleep(2)
time.sleep(2)
for cam in ("camA","camB","camC"): remove(cam); time.sleep(2)
print("\n== DONE ==")
