"""Read NvDsMetaCompLatency from batch/frame user meta (pyds has no wrapper)."""

from __future__ import annotations

import ctypes
from typing import Any

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

import pyds  # noqa: E402

from common.stage_metrics import StageMetrics

MAX_COMPONENT_LEN = 64


class NvDsMetaSubCompLatency(ctypes.Structure):
    _fields_ = [
        ("sub_comp_name", ctypes.c_char * MAX_COMPONENT_LEN),
        ("in_system_timestamp", ctypes.c_double),
        ("out_system_timestamp", ctypes.c_double),
    ]


class NvDsMetaCompLatency(ctypes.Structure):
    _fields_ = [
        ("component_name", ctypes.c_char * MAX_COMPONENT_LEN),
        ("in_system_timestamp", ctypes.c_double),
        ("out_system_timestamp", ctypes.c_double),
        ("source_id", ctypes.c_uint),
        ("frame_num", ctypes.c_uint),
        ("pad_index", ctypes.c_uint),
        ("sub_comp_latencies", NvDsMetaSubCompLatency * 16),
        ("num_sub_comps", ctypes.c_uint),
    ]


def _glist_foreach(lst, caster):
    while lst is not None:
        try:
            yield caster(lst.data)
        except StopIteration:
            break
        try:
            lst = lst.next
        except StopIteration:
            break


def _as_ptr(obj: Any) -> int:
    getter = getattr(pyds, "get_ptr", None)
    if getter is not None:
        return int(getter(obj))
    if isinstance(obj, int):
        return obj
    return int(hash(obj))


def _decode_name(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", "replace").strip()


def _comp_from_user_meta(user_meta) -> tuple[str, float] | None:
    if user_meta.base_meta.meta_type != pyds.NVDS_LATENCY_MEASUREMENT_META:
        return None
    try:
        lat = NvDsMetaCompLatency.from_address(_as_ptr(user_meta.user_meta_data))
    except (ValueError, TypeError, OSError):
        return None
    name = _decode_name(lat.component_name)
    if not name or not all(ch.isalnum() or ch in "-_." for ch in name):
        return None
    ms = float(lat.out_system_timestamp - lat.in_system_timestamp)
    if ms < 0.0 or ms > 60_000.0:
        return None
    return name, ms


def read_component_latencies_ms(batch_meta) -> dict[str, float]:
    """Official DeepStream plugin latencies (ms) attached when env flags are on."""
    out: dict[str, float] = {}
    if batch_meta is None:
        return out

    def _take(user_meta) -> None:
        parsed = _comp_from_user_meta(user_meta)
        if parsed is None:
            return
        name, ms = parsed
        prev = out.get(name)
        if prev is None or ms > prev:
            out[name] = ms

    for user_meta in _glist_foreach(batch_meta.batch_user_meta_list, pyds.NvDsUserMeta.cast):
        _take(user_meta)
    for frame_meta in _glist_foreach(batch_meta.frame_meta_list, pyds.NvDsFrameMeta.cast):
        for user_meta in _glist_foreach(
            frame_meta.frame_user_meta_list, pyds.NvDsUserMeta.cast
        ):
            _take(user_meta)
    return out


def stage_frame_probe(stage: str, metrics: StageMetrics):
    """Pad probe: count + timestamp by NvDs frame_num. No I/O."""

    def _probe(pad, info, user_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            metrics.bump(stage)
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            metrics.bump(stage)
            return Gst.PadProbeReturn.OK

        n = 0
        for frame_meta in _glist_foreach(batch_meta.frame_meta_list, pyds.NvDsFrameMeta.cast):
            stream_id = int(getattr(frame_meta, "pad_index", frame_meta.source_id))
            metrics.mark_stage(stage, frame_meta.frame_num, stream_id=stream_id)
            n += 1
        if n == 0:
            metrics.bump(stage)
        return Gst.PadProbeReturn.OK

    return _probe
