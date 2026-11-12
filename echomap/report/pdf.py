"""PDF export without extra dependencies.

Uses the Edge WebView2 engine already on Windows (CoreWebView2.PrintToPdfAsync) through pywebview's
native control. This keeps the executable small (no PDF library bundled). If that API is unavailable,
it falls back to the system print dialog, where the user can choose "Microsoft Print to PDF".
"""
from __future__ import annotations

import threading
import time
from pathlib import Path


def print_to_pdf(state, report_html_path: str, out_path: str) -> dict:
    import webview

    done = threading.Event()
    result = {}
    html = Path(report_html_path).read_text(encoding="utf-8")
    win = webview.create_window("EchoMap report (PDF export)", html=html, hidden=True, width=1100, height=1400)

    def on_loaded():
        try:
            ctrl = win.native.webview  # Microsoft.Web.WebView2.WinForms.WebView2

            def run():
                task = ctrl.CoreWebView2.PrintToPdfAsync(str(out_path), None)
                return task

            task = win.native.Invoke(run) if hasattr(win.native, "Invoke") else run()
            for _ in range(600):
                if Path(out_path).exists() and Path(out_path).stat().st_size > 0 and (task is None or task.IsCompleted):
                    break
                time.sleep(0.1)
            result.update({"ok": Path(out_path).exists(), "path": str(out_path), "method": "WebView2 PrintToPdf"})
        except Exception as e:  # noqa: BLE001
            result.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
        finally:
            done.set()

    win.events.loaded += on_loaded
    done.wait(90)
    try:
        win.destroy()
    except Exception:  # noqa: BLE001
        pass
    if not result.get("ok"):
        # fallback: system print dialog on the report
        try:
            state.window.evaluate_js("window.EchoMap && window.EchoMap.printReport()")
            result["fallback"] = "Opened the system print dialog. Choose 'Microsoft Print to PDF'."
        except Exception:  # noqa: BLE001
            pass
    if state.case:
        state.case.history.log("export", "PDF export", path=str(out_path), result={k: v for k, v in result.items() if k != "path"})
    return result
