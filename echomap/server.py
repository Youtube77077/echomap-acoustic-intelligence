"""Local API server for the EchoMap window.

Security: binds to 127.0.0.1 on a random port. Every /api request must carry the per-launch random
token (header X-EchoMap-Token), and the Host header must be the loopback address. That blocks other
local web pages and DNS-rebinding attacks from using the API.
"""
from __future__ import annotations

import io
import json
import mimetypes
import secrets
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import config
from .geo import gazetteer
from .models import registry

WEB = config.resource_root() / "web"
# Optional online map providers (contacted only when the user enables the online 3D map in settings)
MAP_HOSTS = "https://tiles.openfreemap.org https://s3.amazonaws.com https://tiles.maps.eox.at"


class AppState:
    def __init__(self, settings: config.Settings):
        self.settings = settings
        self.case = None
        self.token = secrets.token_urlsafe(24)
        self.window = None          # pywebview window (set by app.py)
        self.file_dialog = None     # fallback native open dialog when no pywebview window
        self._wav_cache = None

    def wav_bytes(self) -> bytes:
        c = self.case
        if self._wav_cache and self._wav_cache[0] is c:
            return self._wav_cache[1]
        pcm = (np.clip(c.x, -1, 1) * 32767).astype("<i2").tobytes()
        b = io.BytesIO()
        with wave.open(b, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(c.sr)
            w.writeframes(pcm)
        self._wav_cache = (c, b.getvalue())
        return self._wav_cache[1]


def make_handler(state: AppState):
    class H(BaseHTTPRequestHandler):
        server_version = "EchoMap"

        def log_message(self, *a):  # quiet
            pass

        # ------------------------------------------------------------ helpers
        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").split(":")[0]
            return host in ("127.0.0.1", "localhost")

        def _auth(self) -> bool:
            return self._host_ok() and secrets.compare_digest(self.headers.get("X-EchoMap-Token", ""), state.token)

        def _send(self, code=200, body=b"", ctype="application/json", headers=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False, default=_default).encode("utf-8"))

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 2_000_000:
                return {}
            return json.loads(self.rfile.read(n) or b"{}")

        # ------------------------------------------------------------ routing
        def do_GET(self):
            if not self._host_ok():
                return self._send(403, b"forbidden", "text/plain")
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path in ("/", "/index.html"):
                html = (WEB / "index.html").read_text(encoding="utf-8").replace("__ECHOMAP_TOKEN__", state.token)
                return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8",
                                  {"Content-Security-Policy": "default-src 'self'; img-src 'self' data: blob: " + MAP_HOSTS + "; "
                                   "media-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; "
                                   "worker-src 'self' blob:; child-src blob:; frame-src 'self' data: blob: about:; "
                                   "connect-src 'self' " + MAP_HOSTS})
            if u.path.startswith("/static/"):
                p = (WEB / u.path[len("/static/"):]).resolve()
                if WEB.resolve() not in p.parents or not p.is_file():
                    return self._send(404, b"not found", "text/plain")
                ct = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
                return self._send(200, p.read_bytes(), ct)
            if u.path == "/audio":
                # <audio> cannot send custom headers, so it authenticates with ?t=
                if not secrets.compare_digest(q.get("t", ""), state.token) or not state.case or state.case.x is None:
                    return self._send(403, b"", "text/plain")
                return self._range(state.wav_bytes(), "audio/wav")
            if not u.path.startswith("/api/") or not self._auth():
                return self._send(403, b"forbidden", "text/plain")
            try:
                return self._api_get(u.path[5:], q)
            except Exception as e:  # noqa: BLE001
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_POST(self):
            if not self._auth():
                return self._send(403, b"forbidden", "text/plain")
            u = urlparse(self.path)
            try:
                return self._api_post(u.path[5:], self._body())
            except Exception as e:  # noqa: BLE001
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def _range(self, data: bytes, ctype: str):
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                a, _, b = rng[6:].partition("-")
                start = int(a or 0)
                end = int(b) if b else len(data) - 1
                end = min(end, len(data) - 1)
                chunk = data[start:end + 1]
                return self._send(206, chunk, ctype, {"Content-Range": f"bytes {start}-{end}/{len(data)}",
                                                      "Accept-Ranges": "bytes"})
            return self._send(200, data, ctype, {"Accept-Ranges": "bytes"})

        # ------------------------------------------------------------ GET API
        def _api_get(self, path, q):
            c = state.case
            if path == "status":
                return self._json({"state": c.state if c else "idle", "progress": c.progress if c else None,
                                   "error": c.error if c else None})
            if path == "result":
                if not c:
                    return self._json({"error": "no case"}, 404)
                r = {k: v for k, v in c.result.items() if k not in ("birdnet_raw",)}
                return self._json({"sha256": c.sha256, "meta": c.meta, "result": r, "state": c.state,
                                   "case_dir": str(c.dir) if c.dir else None,
                                   "history": c.history.to_json()})
            if path == "spectrogram":
                if not c or c.x is None:
                    return self._json({"error": "no audio"}, 404)
                v = c.spectrogram(t0=float(q.get("t0", 0)), t1=float(q.get("t1", 10)), fmin=float(q.get("fmin", 0)),
                                  fmax=float(q.get("fmax", 16000)), scale=q.get("scale", "linear"),
                                  width=int(q.get("w", 1200)), height=int(q.get("h", 360)),
                                  db_range=(float(q.get("dbmin", -110)), float(q.get("dbmax", -10))))
                data = v.pop("data")
                return self._send(200, data.tobytes(), "application/octet-stream", {"X-Meta": json.dumps(v)})
            if path == "search_text":
                if not c or getattr(c, "_sed", None) is None:
                    return self._json({"error": "no analysed case"}, 404)
                from .detect import textsearch
                return self._json(textsearch.search(q.get("q", ""), c._sed, c._labels))
            if path == "models":
                return self._json({"dir": state.settings.models_dir, "models": registry.status(state.settings.models_dir)})
            if path == "settings":
                return self._json(vars(state.settings))
            if path == "countries":
                return self._json(gazetteer.country_list())
            if path == "cities":
                return self._json(gazetteer.search_city(q.get("q", ""), q.get("cc") or None))
            if path == "report":
                if not c or not c.result.get("report_html"):
                    return self._send(404, b"Report not generated yet", "text/plain")
                return self._send(200, Path(c.result["report_html"]).read_bytes(), "text/html; charset=utf-8")
            return self._json({"error": "unknown endpoint"}, 404)

        # ------------------------------------------------------------ POST API
        def _api_post(self, path, body):
            from .pipeline import Case
            s = state.settings
            if path == "open":
                p = body.get("path")
                if not p and state.window is not None:
                    import webview
                    exts = ";".join(f"*{e}" for e in sorted(config.SUPPORTED_EXT))
                    res = state.window.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=False,
                                                          file_types=(f"Audio/video ({exts})", "All files (*.*)"))
                    p = res[0] if res else None
                elif not p and state.file_dialog is not None:
                    p = state.file_dialog()
                if not p:
                    return self._json({"cancelled": True})
                c = Case(p, s)
                info = c.prepare()
                state.case = c
                state._wav_cache = None
                return self._json({"ok": True, **info, "models": registry.status(s.models_dir), "state": c.state})
            if path == "analyse":
                c = state.case
                if not c or c.state != "prepared":
                    return self._json({"error": "no prepared case"}, 400)
                threading.Thread(target=c.analyse, daemon=True).start()
                return self._json({"ok": True})
            if path == "context":
                c = state.case
                if not c or c.state not in ("analysed", "complete", "error") or not c.result.get("inventory") is not None:
                    return self._json({"error": "analysis not finished"}, 400)
                threading.Thread(target=c.contextualise, args=(body.get("context") or {},), daemon=True).start()
                return self._json({"ok": True})
            if path == "settings":
                for k, v in body.items():
                    if hasattr(s, k) and k != "extra":
                        setattr(s, k, v)
                s.save()
                return self._json({"ok": True, "settings": vars(s)})
            if path == "choose_models_dir":
                if state.window is None:
                    return self._json({"cancelled": True})
                import webview
                res = state.window.create_file_dialog(webview.FileDialog.FOLDER)
                if res:
                    s.models_dir = res[0]
                    s.save()
                return self._json({"dir": s.models_dir, "models": registry.status(s.models_dir)})
            if path == "export":
                c = state.case
                fmt = body.get("format", "html")
                if not c or not c.result.get("report_html"):
                    return self._json({"error": "no report"}, 400)
                if state.window is None:
                    return self._json({"path": c.result["report_html"]})
                import webview
                if fmt == "pdf":
                    from .report.pdf import print_to_pdf
                    res = state.window.create_file_dialog(webview.FileDialog.SAVE, save_filename="echomap_report.pdf")
                    if not res:
                        return self._json({"cancelled": True})
                    out = res if isinstance(res, str) else res[0]
                    return self._json(print_to_pdf(state, c.result["report_html"], out))
                name = "echomap_report.html" if fmt == "html" else "echomap_case.json"
                res = state.window.create_file_dialog(webview.FileDialog.SAVE, save_filename=name)
                if not res:
                    return self._json({"cancelled": True})
                out = Path(res if isinstance(res, str) else res[0])
                src = Path(c.result["report_html"]) if fmt == "html" else c.dir / "case.json"
                out.write_bytes(src.read_bytes())
                c.history.log("export", f"Exported {fmt}", path=str(out))
                return self._json({"ok": True, "path": str(out)})
            return self._json({"error": "unknown endpoint"}, 404)

    return H


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def start(state: AppState) -> int:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]
