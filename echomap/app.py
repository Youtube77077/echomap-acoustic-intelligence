"""EchoMap desktop entry point."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from . import config
from .server import AppState, start


def _default_models_dir(settings: config.Settings) -> None:
    """Prefer a 'models' folder next to EchoMap.exe (portable install), else the configured folder."""
    exe_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
    portable = exe_dir / "models"
    if portable.is_dir() and not Path(settings.models_dir).is_dir():
        settings.models_dir = str(portable)


def main() -> None:
    settings = config.Settings.load()
    _default_models_dir(settings)
    state = AppState(settings)
    port = start(state)
    url = f"http://127.0.0.1:{port}/"
    if os.environ.get("ECHOMAP_HEADLESS"):
        print(url, state.token, flush=True)
        import time
        while True:
            time.sleep(3600)
    try:
        import webview
    except ImportError:
        import webbrowser
        print("pywebview not available; opening the default browser instead:", url)
        webbrowser.open(url)
        input("Press Enter to quit EchoMap...")
        return
    win = webview.create_window(f"{config.APP_NAME} {config.APP_VERSION}: acoustic intelligence", url,
                                width=1480, height=940, min_size=(1100, 700), background_color="#0d1320",
                                text_select=True)
    state.window = win
    storage = config.app_home() / "webview"
    storage.mkdir(parents=True, exist_ok=True)
    webview.start(gui="edgechromium", private_mode=True, storage_path=str(storage))


if __name__ == "__main__":
    main()
