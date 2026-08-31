# File 5 — `pgie_batch.py`

**Write:** `learn/src/config/pgie_batch.py`  
**Reference:** `common/pgie_batch.py`  
**Dependencies:** pathlib, re

---

## Substep 5.1 — Path constants

`ONNX_STATIC`, `ONNX_DYNAMIC`, `ENGINE_DIR` under `/workspace/models`.

---

## Substep 5.2 — `engine_path(onnx, batch, gpu_id=0, precision="fp16")`

Return `ENGINE_DIR / f"{onnx.name}_b{batch}_gpu{gpu_id}_{precision}.engine"`.

**Test:**

```python
from pathlib import Path
p = engine_path(Path("yolo26m_dyn.onnx"), 4)
assert "_b4_" in str(p)
```

---

## Substep 5.3 — Inner `_sub(key, value, body)` inside `write_pgie_config`

Regex replace `key=...` line or insert after `[property]`.

---

## Substep 5.4 — `write_pgie_config(src, batch, dest, onnx=None)`

1. Read src config text
2. Pick ONNX: static for batch 1; dynamic if batch>1 and file exists
3. Patch `batch-size`, `onnx-file`, `model-engine-file`
4. Write dest; return dest Path

**Test:**

```sh
python3 -c "
from pathlib import Path
from config.pgie_batch import write_pgie_config
p = write_pgie_config(
    Path('/workspace/configs/pgie_yolo26.txt'), 4,
    Path('/tmp/pgie_b4.txt'))
print(p.read_text()[:500])
"
```

Expect `batch-size=4` and `_b4_` engine path.

---

## Checkpoint

Required before multisrc `n > 1` pipeline.

Next: [06-mlflow_run.md](06-mlflow_run.md)
