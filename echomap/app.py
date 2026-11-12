"""EchoMap desktop entry point."""
from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from . import config
from .server import AppState, start


def _log(msg: str) -> None:
    try:
        p = config.app_home() / "logs"
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "startup.log", "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    except OSError:
        pass


def _default_models_dir(settings: config.Settings) -> None:
    """Prefer a 'models' folder next to EchoMap.exe (portable install), else the configured folder."""
    exe_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
    portable = exe_dir / "models"
    if portable.is_dir() and not Path(settings.models_dir).is_dir():
        settings.models_dir = str(portable)


def _unblock_bundle() -> int:
    """Remove the 'downloaded from the internet' mark (Zone.Identifier stream) from EchoMap's OWN bundled
    .dll/.pyd/.exe files. Windows adds it when a downloaded ZIP is extracted, and .NET Framework then refuses
    to load pythonnet's Python.Runtime.dll ("Failed to resolve Python.Runtime.Loader.Initialize").
    This is what Explorer's 'Unblock' does; it touches nothing outside the EchoMap folder."""
    if os.name != "nt" or not getattr(sys, "frozen", False):
        return 0
    root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    n = 0
    for p in root.rglob("*"):
        if p.suffix.lower() in (".dll", ".pyd", ".exe"):
            try:
                os.remove(str(p) + ":Zone.Identifier")
                n += 1
            except OSError:
                pass
    return n


def _open_file_dialog() -> str | None:
    """Native Windows open-file dialog via comdlg32 (used when the pywebview window is unavailable)."""
    import ctypes
    from ctypes import wintypes

    class OFN(ctypes.Structure):
        _fields_ = [("lStructSize", wintypes.DWORD), ("hwndOwner", wintypes.HWND), ("hInstance", wintypes.HINSTANCE),
                    ("lpstrFilter", wintypes.LPCWSTR), ("lpstrCustomFilter", wintypes.LPWSTR), ("nMaxCustFilter", wintypes.DWORD),
                    ("nFilterIndex", wintypes.DWORD), ("lpstrFile", wintypes.LPWSTR), ("nMaxFile", wintypes.DWORD),
                    ("lpstrFileTitle", wintypes.LPWSTR), ("nMaxFileTitle", wintypes.DWORD), ("lpstrInitialDir", wintypes.LPCWSTR),
                    ("lpstrTitle", wintypes.LPCWSTR), ("Flags", wintypes.DWORD), ("nFileOffset", wintypes.WORD),
                    ("nFileExtension", wintypes.WORD), ("lpstrDefExt", wintypes.LPCWSTR), ("lCustData", wintypes.LPARAM),
                    ("lpfnHook", ctypes.c_void_p), ("lpTemplateName", wintypes.LPCWSTR), ("pvReserved", ctypes.c_void_p),
                    ("dwReserved", wintypes.DWORD), ("FlagsEx", wintypes.DWORD)]
    buf = ctypes.create_unicode_buffer(1024)
    exts = ";".join(f"*{e}" for e in sorted(config.SUPPORTED_EXT))
    ofn = OFN()
    ofn.lStructSize = ctypes.sizeof(OFN)
    ofn.lpstrFilter = f"Audio and video files ({exts})\0{exts}\0All files\0*.*\0\0"
    ofn.lpstrFile = ctypes.cast(buf, wintypes.LPWSTR)
    ofn.nMaxFile = 1024
    ofn.lpstrTitle = "EchoMap: import recording"
    ofn.Flags = 0x00001000 | 0x00000800 | 0x00080000  # FILEMUSTEXIST | PATHMUSTEXIST | EXPLORER
    return buf.value if ctypes.windll.comdlg32.GetOpenFileNameW(ctypes.byref(ofn)) else None


def _edge_app_window(url: str) -> bool:
    """Fallback window: Microsoft Edge in app mode with a private EchoMap profile. Blocks until it closes."""
    for base in (os.environ.get("ProgramFiles(x86)", ""), os.environ.get("ProgramFiles", "")):
        edge = Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
        if edge.exists():
            prof = config.app_home() / "edge-profile"
            prof.mkdir(parents=True, exist_ok=True)
            p = subprocess.Popen([str(edge), f"--app={url}", f"--user-data-dir={prof}", "--no-first-run",
                                  "--window-size=1480,940"])
            p.wait()
            return True
    return False


def main() -> None:
    settings = config.Settings.load()
    _default_models_dir(settings)
    n = _unblock_bundle()
    if n:
        _log(f"removed download mark from {n} bundled files")
    state = AppState(settings)
    state.file_dialog = _open_file_dialog if os.name == "nt" else None
    port = start(state)
    url = f"http://127.0.0.1:{port}/"
    hf = os.environ.get("ECHOMAP_HEADLESS_FILE")
    if hf:  # used by the CI smoke test of the built executable
        Path(hf).write_text(f"{url} {state.token}", encoding="utf-8")
    if os.environ.get("ECHOMAP_HEADLESS"):
        print(url, state.token, flush=True)
        while True:
            time.sleep(3600)
    try:
        import webview
        win = webview.create_window(f"{config.APP_NAME} {config.APP_VERSION}: acoustic intelligence", url,
                                    width=1480, height=940, min_size=(1100, 700), background_color="#0d1320",
                                    text_select=True)
        state.window = win
        storage = config.app_home() / "webview"
        storage.mkdir(parents=True, exist_ok=True)
        webview.start(gui="edgechromium", private_mode=True, storage_path=str(storage))
        return
    except Exception:  # noqa: BLE001
        _log("native window failed, falling back to Edge app window:\n" + traceback.format_exc())
        state.window = None
    if hf:
        while True:
            time.sleep(3600)
    if not _edge_app_window(url):
        import webbrowser
        webbrowser.open(url)
        while True:  # keep the local server alive for the browser tab
            time.sleep(3600)


if __name__ == "__main__":
    main()
