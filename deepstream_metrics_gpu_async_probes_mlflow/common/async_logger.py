"""Bounded-queue JSONL logger. Never call from a pad probe except via put_nowait."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any


_SENTINEL = object()


class AsyncLogger:
    def __init__(self, path: str, maxsize: int = 1024):
        self.path = path
        self.dropped = 0
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._run, name="async-logger", daemon=True)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def emit(self, record: dict[str, Any]) -> None:
        try:
            self._q.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            while True:
                rec = self._q.get()
                if rec is _SENTINEL:
                    fh.flush()
                    break
                fh.write(json.dumps(rec, default=str) + "\n")
                if self._q.empty():
                    fh.flush()

    def close(self, timeout: float = 15.0) -> None:
        try:
            self._q.put(_SENTINEL, timeout=timeout)
        except queue.Full:
            self.dropped += 1
            return
        self._thread.join(timeout=timeout)


class ProbeSyncLogger:
    """Blocking JSONL writer for the A/B experiment (I/O on the streaming thread)."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.dropped = 0
        self._fh = open(path, "a", encoding="utf-8")

    def emit(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def close(self, timeout: float = 15.0) -> None:
        self._fh.flush()
        self._fh.close()
