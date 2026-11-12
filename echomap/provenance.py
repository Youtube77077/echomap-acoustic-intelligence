"""Processing history: every step that touches evidence is logged with parameters and timing."""
from __future__ import annotations

import platform
import threading
import time
from datetime import datetime, timezone


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class History:
    def __init__(self):
        self._lock = threading.Lock()
        self.steps: list[dict] = []
        self.environment = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        }

    def log(self, step: str, detail: str = "", **params) -> dict:
        rec = {"time_utc": utcnow(), "step": step, "detail": detail, "params": params}
        with self._lock:
            self.steps.append(rec)
        return rec

    class _Timer:
        def __init__(self, hist: "History", step: str, detail: str, params: dict):
            self.h, self.step, self.detail, self.params = hist, step, detail, params

        def __enter__(self):
            self.t0 = time.perf_counter()
            self.rec = self.h.log(self.step, self.detail + " (started)", **self.params)
            return self.rec

        def __exit__(self, et, ev, tb):
            dt = time.perf_counter() - self.t0
            status = "failed: %s" % ev if et else "ok"
            self.rec["detail"] = self.detail
            self.rec["duration_s"] = round(dt, 3)
            self.rec["status"] = status
            return False

    def timed(self, step: str, detail: str = "", **params):
        return History._Timer(self, step, detail, params)

    def to_json(self) -> dict:
        return {"environment": self.environment, "steps": list(self.steps)}
