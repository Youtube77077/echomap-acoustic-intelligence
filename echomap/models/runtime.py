"""Lazy ONNX Runtime sessions: loaded on demand and released after each case."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import onnxruntime as ort

_lock = threading.Lock()
_sessions: dict[str, ort.InferenceSession] = {}


def session(path: str | Path) -> ort.InferenceSession:
    key = str(path)
    with _lock:
        if key not in _sessions:
            if not Path(path).exists():
                raise FileNotFoundError(f"Model file not found: {path}")
            so = ort.SessionOptions()
            so.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
            so.log_severity_level = 3
            _sessions[key] = ort.InferenceSession(key, sess_options=so, providers=["CPUExecutionProvider"])
        return _sessions[key]


def release_all() -> None:
    with _lock:
        _sessions.clear()


def version() -> str:
    return ort.__version__
