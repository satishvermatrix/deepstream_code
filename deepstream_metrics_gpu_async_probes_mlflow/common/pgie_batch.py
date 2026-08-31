"""Write a per-batch nvinfer config so TensorRT engines are named _bN_."""

from __future__ import annotations

import re
from pathlib import Path

ONNX_STATIC = Path("/workspace/models/yolo26m.onnx")
ONNX_DYNAMIC = Path("/workspace/models/yolo26m_dyn.onnx")
ENGINE_DIR = Path("/workspace/models")


def engine_path(onnx: Path, batch: int, gpu_id: int = 0, precision: str = "fp16") -> Path:
    return ENGINE_DIR / f"{onnx.name}_b{batch}_gpu{gpu_id}_{precision}.engine"


def write_pgie_config(src: Path, batch: int, dest: Path, onnx: Path | None = None) -> Path:
    text = src.read_text()
    onnx_path = onnx if onnx is not None else ONNX_STATIC
    if batch > 1:
        onnx_path = onnx if onnx is not None else (
            ONNX_DYNAMIC if ONNX_DYNAMIC.is_file() else ONNX_STATIC
        )
    eng = engine_path(onnx_path, batch)

    def _sub(key: str, value: str, body: str) -> str:
        pat = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
        if pat.search(body):
            return pat.sub(f"{key}={value}", body, count=1)
        return body.replace("[property]", f"[property]\n{key}={value}", 1)

    text = _sub("batch-size", str(batch), text)
    text = _sub("onnx-file", str(onnx_path), text)
    text = _sub("model-engine-file", str(eng), text)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    return dest
