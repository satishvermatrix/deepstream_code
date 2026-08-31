"""Per-stage frame counts, FPS, pad-to-pad hops, and DS plugin latency (no I/O)."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any

_HOP_PREV = {
    "pgie": "mux",
    "trk": "pgie",
    "osd": "trk",
    "enc": "osd",
}
_TS_CAP = 4096
_SAMPLE_CAP = 4096


def frame_key(stream_id: int, frame_num: int) -> int:
    return (int(stream_id) << 32) | (int(frame_num) & 0xFFFFFFFF)


def _percentile(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(p / 100.0 * len(ordered)) - 1)))
    return ordered[idx]


def _stats(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0, "n": 0.0}
    return {
        "avg": sum(samples) / len(samples),
        "min": min(samples),
        "max": max(samples),
        "p95": _percentile(samples, 95.0),
        "n": float(len(samples)),
    }


def sanitize_metric_name(name: str) -> str:
    out = []
    for ch in name:
        out.append(ch if ch.isalnum() else "_")
    collapsed = "".join(out).strip("_")
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed or "unknown"


class StageMetrics:
    def __init__(self) -> None:
        self.counts = defaultdict(int)
        self.objects = 0
        self.by_class: dict[str, int] = defaultdict(int)
        self.dropped_frames = 0
        self._last_frame_num: dict[int, int] = {}
        self._stage_ts: dict[str, dict[int, float]] = defaultdict(dict)
        self._hop_samples: dict[str, list[float]] = defaultdict(list)
        self._ds_samples: dict[str, list[float]] = defaultdict(list)
        self._frame_hops: dict[int, dict[str, float]] = {}
        self._probe_log_sum_us = 0.0
        self._probe_log_n = 0
        self.t_start: float | None = None
        self.t_end: float | None = None

    def bump(self, stage: str) -> None:
        if self.t_start is None:
            self.t_start = time.perf_counter()
        self.counts[stage] += 1

    def mark_mux_frame(self, frame_num: int, stream_id: int = 0) -> dict[str, float]:
        return self.mark_stage("mux", frame_num, stream_id=stream_id)

    def mark_stage(
        self, stage: str, frame_num: int, stream_id: int = 0
    ) -> dict[str, float]:
        now = time.perf_counter()
        if self.t_start is None:
            self.t_start = now
        self.counts[stage] += 1
        key = frame_key(stream_id, frame_num)
        bucket = self._stage_ts[stage]
        bucket[key] = now
        if len(bucket) > _TS_CAP:
            bucket.pop(min(bucket), None)

        hops: dict[str, float] = {}
        prev = _HOP_PREV.get(stage)
        if prev == "trk" and key not in self._stage_ts.get("trk", {}):
            prev = "pgie"
        if prev is not None:
            t_prev = self._stage_ts.get(prev, {}).get(key)
            if t_prev is not None:
                hops[f"{prev}_{stage}"] = (now - t_prev) * 1000.0
        if stage == "osd":
            t_mux = self._stage_ts.get("mux", {}).get(key)
            if t_mux is not None:
                hops["mux_osd"] = (now - t_mux) * 1000.0
        for hop, ms in hops.items():
            samples = self._hop_samples[hop]
            samples.append(ms)
            if len(samples) > _SAMPLE_CAP:
                del samples[: len(samples) - _SAMPLE_CAP]
            self._frame_hops.setdefault(key, {})[hop] = ms
            if len(self._frame_hops) > _TS_CAP:
                self._frame_hops.pop(min(self._frame_hops), None)
        return hops

    def hops_for_frame(self, frame_num: int, stream_id: int = 0) -> dict[str, float]:
        return dict(self._frame_hops.get(frame_key(stream_id, frame_num)) or {})

    def note_osd_frame(
        self,
        frame_num: int,
        n_obj: int,
        by_class: dict[str, int],
        stream_id: int = 0,
    ) -> tuple[float, dict[str, float]]:
        hops = self.mark_stage("osd", frame_num, stream_id=stream_id)
        self.t_end = time.perf_counter()
        self.objects += n_obj
        for k, v in by_class.items():
            self.by_class[k] += v
        last = self._last_frame_num.get(stream_id, -1)
        if last >= 0 and frame_num > last + 1:
            self.dropped_frames += frame_num - last - 1
        self._last_frame_num[stream_id] = frame_num
        return hops.get("mux_osd", 0.0), hops

    def note_ds_latencies(self, by_component: dict[str, float]) -> None:
        for name, ms in by_component.items():
            key = sanitize_metric_name(name)
            samples = self._ds_samples[key]
            samples.append(ms)
            if len(samples) > _SAMPLE_CAP:
                del samples[: len(samples) - _SAMPLE_CAP]

    def note_probe_log_us(self, elapsed_us: float) -> None:
        self._probe_log_sum_us += elapsed_us
        self._probe_log_n += 1

    def fps_e2e(self) -> float:
        if self.t_start is None or self.t_end is None:
            return 0.0
        dt = self.t_end - self.t_start
        if dt <= 0:
            return 0.0
        return self.counts["osd"] / dt

    def latency_e2e_ms_avg(self) -> float:
        samples = self._hop_samples.get("mux_osd") or []
        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    def probe_log_us_avg(self) -> float:
        if self._probe_log_n == 0:
            return 0.0
        return self._probe_log_sum_us / self._probe_log_n

    def summary(self) -> dict[str, Any]:
        hops = {k: _stats(v) for k, v in sorted(self._hop_samples.items())}
        ds = {k: _stats(v) for k, v in sorted(self._ds_samples.items())}
        out: dict[str, Any] = {
            "fps_e2e": self.fps_e2e(),
            "frames_src": int(self.counts["src"]),
            "frames_mux": int(self.counts["mux"]),
            "frames_pgie": int(self.counts["pgie"]),
            "frames_trk": int(self.counts["trk"]),
            "frames_osd": int(self.counts["osd"]),
            "frames_enc": int(self.counts["enc"]),
            "dropped_frames": self.dropped_frames,
            "objects_total": self.objects,
            "by_class": dict(self.by_class),
            "latency_e2e_ms_avg": self.latency_e2e_ms_avg(),
            "probe_log_us_avg": self.probe_log_us_avg(),
            "latency_hops_ms": hops,
            "latency_ds_ms": ds,
        }
        for hop, st in hops.items():
            for stat, val in st.items():
                out[f"latency_hop_{hop}_ms_{stat}"] = val
        for name, st in ds.items():
            for stat, val in st.items():
                out[f"latency_ds_{name}_ms_{stat}"] = val
        return out
