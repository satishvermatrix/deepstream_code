#!/usr/bin/env python3
"""Download YOLO26m + Market-1501 ReID, export ONNX, compile the YOLO26 parser.

Run inside ds8-dev (./setup_tracker.sh on the host wraps this).
"""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path("/workspace")
MODELS = ROOT / "models"
TRACKER_MODELS = MODELS / "Tracker"
PARSER = ROOT / "parsers" / "yolo26"

YOLO_PT = MODELS / "yolo26m.pt"
YOLO_ONNX = MODELS / "yolo26m.onnx"
YOLO_PT_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt"
REID_ETLT = TRACKER_MODELS / "resnet50_market1501.etlt"
REID_URL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet"
    "/versions/deployable_v1.0/files/resnet50_market1501.etlt"
)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1024:
        print(f"exists {dest}")
        return
    print(f"download {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def compile_parser() -> None:
    print("compile YOLO26 bbox parser")
    subprocess.check_call(["make", "-C", str(PARSER)])
    so = PARSER / "libnvdsinfer_custom_impl_Yolo26.so"
    if not so.is_file():
        raise RuntimeError(f"parser build failed: {so}")
    print(f"ok {so}")


def export_yolo26() -> None:
    if YOLO_ONNX.is_file() and YOLO_ONNX.stat().st_size > 1024:
        print(f"exists {YOLO_ONNX}")
        return
    print("pip install ultralytics (ONNX export)")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--break-system-packages",
            "-q",
            "ultralytics",
            "onnx",
            "onnxsim",
        ]
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--break-system-packages",
            "-q",
            "numpy<2",
        ]
    )
    from ultralytics import YOLO

    print(f"export {YOLO_PT} -> onnx (end2end / nms)")
    os.chdir(MODELS)
    model = YOLO(str(YOLO_PT))
    out = model.export(
        format="onnx",
        imgsz=640,
        dynamic=False,
        simplify=True,
        opset=17,
        nms=True,
        batch=1,
    )
    exported = Path(str(out))
    if exported.resolve() != YOLO_ONNX.resolve():
        exported.replace(YOLO_ONNX)
    print(f"ok {YOLO_ONNX}")


def export_yolo26_dynamic() -> None:
    """ONNX with a dynamic batch axis so nvinfer can build _bN_ engines for N>1."""
    dest = MODELS / "yolo26m_dyn.onnx"
    if dest.is_file() and dest.stat().st_size > 1024:
        print(f"exists {dest}")
        return
    print("export yolo26m dynamic-batch ONNX (max batch 16)")
    from ultralytics import YOLO

    os.chdir(MODELS)
    static_bak = MODELS / "yolo26m.onnx.staticbak"
    if YOLO_ONNX.is_file():
        YOLO_ONNX.replace(static_bak)
    try:
        model = YOLO(str(YOLO_PT))
        out = model.export(
            format="onnx",
            imgsz=640,
            dynamic=True,
            simplify=True,
            opset=17,
            nms=True,
            batch=16,
        )
        exported = Path(str(out))
        if exported.resolve() != dest.resolve():
            exported.replace(dest)
        print(f"ok {dest}")
    finally:
        if static_bak.is_file():
            static_bak.replace(YOLO_ONNX)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--break-system-packages",
                "-q",
                "numpy<2",
            ]
        )


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    TRACKER_MODELS.mkdir(parents=True, exist_ok=True)
    compile_parser()
    download(REID_URL, REID_ETLT)
    download(YOLO_PT_URL, YOLO_PT)
    export_yolo26()
    export_yolo26_dynamic()
    print("setup_tracker ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
