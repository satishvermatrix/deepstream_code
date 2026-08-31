"""Background nvidia-smi sampler. Must not run inside a pad probe."""

from __future__ import annotations

import subprocess
import threading
import time
from typing import Any, Callable

QUERY = (
    "utilization.gpu,memory.used,memory.total,"
    "utilization.decoder,utilization.encoder,utilization.ofa,"
    "power.draw,temperature.gpu,clocks.sm,clocks.mem"
)
FIELDS = [
    "gpu_util",
    "mem_used_mb",
    "mem_total_mb",
    "nvdec_util",
    "nvenc_util",
    "ofa_util",
    "power_w",
    "temp_c",
    "clock_sm_mhz",
    "clock_mem_mhz",
]


def _to_float(raw: str) -> float | None:
    raw = raw.strip()
    if raw in ("", "[N/A]", "N/A", "[Not Supported]"):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def query_gpu() -> dict[str, float | None]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={QUERY}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=3,
    )
    parts = [p.strip() for p in out.strip().split(",")]
    row: dict[str, float | None] = {"t": time.time()}
    for name, raw in zip(FIELDS, parts):
        row[name] = _to_float(raw)
    return row


class GpuSampler:
    def __init__(self, emit: Callable[[dict[str, Any]], None] | None, interval: float = 1.0):
        self._emit = emit
        self._interval = interval
        self.samples: list[dict[str, float | None]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)

    def start(self) -> None:
        try:
            self.samples.append(query_gpu())
        except Exception:
            pass
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                row = query_gpu()
                self.samples.append(row)
                if self._emit:
                    payload = {"type": "gpu", **row}
                    self._emit(payload)
            except Exception:
                pass
            self._stop.wait(self._interval)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def summary(self) -> dict[str, float]:
        numeric: dict[str, list[float]] = {k: [] for k in FIELDS}
        for row in self.samples:
            for k in FIELDS:
                val = row.get(k)
                if isinstance(val, (int, float)):
                    numeric[k].append(float(val))

        def avg(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        def mx(xs: list[float]) -> float:
            return max(xs) if xs else 0.0

        return {
            "gpu_util_avg": avg(numeric["gpu_util"]),
            "mem_used_mb_max": mx(numeric["mem_used_mb"]),
            "mem_total_mb": mx(numeric["mem_total_mb"]),
            "nvdec_util_avg": avg(numeric["nvdec_util"]),
            "nvenc_util_avg": avg(numeric["nvenc_util"]),
            "ofa_util_avg": avg(numeric["ofa_util"]),
            "power_w_avg": avg(numeric["power_w"]),
            "temp_c_max": mx(numeric["temp_c"]),
            "gpu_samples": float(len(self.samples)),
        }
