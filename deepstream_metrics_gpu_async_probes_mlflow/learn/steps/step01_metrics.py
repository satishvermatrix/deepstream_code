#!/usr/bin/env python3
"""Checkpoint for tutorial 01 — run after StageMetrics.summary() works."""
import json
import time

from metrics.stage_metrics import StageMetrics

m = StageMetrics()
for fn in range(50):
    m.mark_stage("mux", fn, stream_id=0)
    time.sleep(0.001)
    m.mark_stage("pgie", fn, stream_id=0)
    time.sleep(0.002)
    m.mark_stage("trk", fn, stream_id=0)
    time.sleep(0.001)
    m.note_osd_frame(fn, 2, {"person": 2}, stream_id=0)

print(json.dumps(m.summary(), indent=2))
