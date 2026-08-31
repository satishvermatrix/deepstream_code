# File 2 — `async_logger.py`

**Write:** `learn/src/logging/async_logger.py`  
**Reference:** `common/async_logger.py`  
**Dependencies:** none

---

## Substep 2.1 — Module header and `_SENTINEL`

```python
from __future__ import annotations
import json, queue, threading
from pathlib import Path
from typing import Any

_SENTINEL = object()
```

---

## Substep 2.2 — `AsyncLogger.__init__(path, maxsize=1024)`

1. `self.path`, `self.dropped = 0`
2. `self._q = queue.Queue(maxsize=maxsize)`
3. Create daemon thread `async-logger` targeting `self._run`
4. `mkdir` parent of path; `self._thread.start()`

**Test:** construct logger; thread is alive.

---

## Substep 2.3 — `AsyncLogger.emit(record)`

`put_nowait(record)`; on `queue.Full`, increment `dropped`.

**Test:** queue maxsize=2, emit 5 records quickly — some dropped or queue works.

---

## Substep 2.4 — `AsyncLogger._run`

Loop: `get()` from queue; if `_SENTINEL`, flush and break; else `json.dumps` + newline; flush when queue empty.

**Test:** emit 3 records, close, file has 3 lines.

---

## Substep 2.5 — `AsyncLogger.close(timeout=15.0)`

Put sentinel (with timeout on failure), `join` thread.

**Test:** after close, thread stopped; file complete.

---

## Substep 2.6 — `ProbeSyncLogger.__init__(path)`

Open append file; `dropped = 0`.

---

## Substep 2.7 — `ProbeSyncLogger.emit(record)`

Sync write + flush (blocking).

---

## Substep 2.8 — `ProbeSyncLogger.close(timeout=15.0)`

Flush and close file handle.

---

## Checkpoint — `learn/steps/step02_logger.py`

```python
from logging.async_logger import AsyncLogger

log = AsyncLogger("/tmp/learn_test.jsonl", maxsize=100)
for i in range(20):
    log.emit({"type": "frame", "i": i})
log.close()
with open("/tmp/learn_test.jsonl") as f:
    lines = f.readlines()
print(f"lines={len(lines)} dropped={log.dropped}")
```

Expected: `lines=20`, `dropped=0`.

Next: [03-gpu_sampler.md](03-gpu_sampler.md)
