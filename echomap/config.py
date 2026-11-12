"""User settings. Stored as JSON in %LOCALAPPDATA%\\EchoMap\\config.json."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

APP_NAME = "EchoMap"
APP_VERSION = "1.0.3"
MAX_DURATION_S = 300.0  # five minutes
SUPPORTED_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".mp4", ".mov", ".mkv", ".webm"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm"}


def app_home() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_NAME


def resource_root() -> Path:
    """Folder holding echomap/web and echomap/data (works both from source and from a PyInstaller bundle)."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "echomap"
    return Path(__file__).resolve().parent


@dataclass
class Settings:
    models_dir: str = str(app_home() / "models")
    cases_dir: str = str(app_home() / "cases")
    keep_analysis_wav: bool = False
    list_threshold: float = 0.15       # minimum model score listed as an event (all scores kept in case.json)
    allow_online_gbif: bool = False
    allow_online_osm: bool = False
    allow_online_weather: bool = False
    allow_online_maps: bool = False   # 3D map tiles from OpenFreeMap / AWS Terrain Tiles / EOX (off by default)
    transcribe_speech: bool = True   # local Whisper transcription of all speech, including faint background speech
    analysis_mode: str = "deep"  # quick | standard | deep | experimental
    sed_hop_s: float = 0.5
    birdnet_hop_s: float = 1.5
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Settings":
        p = app_home() / "config.json"
        s = cls()
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for k, v in data.items():
                    if hasattr(s, k):
                        setattr(s, k, v)
            except (OSError, ValueError):
                pass
        return s

    def save(self) -> None:
        p = app_home() / "config.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
