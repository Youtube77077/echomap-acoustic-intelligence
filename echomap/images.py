"""Offline picture pack (v1.0.9): a small credited photo for everything EchoMap identifies, except language, speech and
mains hum. The pack (EchoMap-images-v1.zip, built by tools/build_images.py from Wikimedia Commons) is a separate
download; it is found next to EchoMap.exe, in the models folder, in %LOCALAPPDATA%\\EchoMap or in Downloads/Desktop.
Without it the app works as before, just without pictures. Nothing is fetched online.
"""
from __future__ import annotations

import json
import re
import sys
import threading
import zipfile
from functools import lru_cache
from pathlib import Path

from . import config

PACK_GLOB = "EchoMap-images*.zip"
_lock = threading.Lock()
NO_PICTURE_CATS = {"language", "buildings_acoustics", "unknown"}
# derived / DSP records -> the AudioSet class whose picture fits
LABEL_ALIASES = [(re.compile(r"siren", re.I), "Siren"), (re.compile(r"church bell|bell ring", re.I), "Church bell"),
                 (re.compile(r"wind gust|wind noise", re.I), "Wind"), (re.compile(r"engine", re.I), "Engine"),
                 (re.compile(r"level crossing|crossing signal", re.I), "Railroad car, train wagon"),
                 (re.compile(r"avas|electric vehicle", re.I), "Car")]


def find_pack(models_dir: str | None = None) -> Path | None:
    roots = []
    if models_dir:
        roots += [Path(models_dir), Path(models_dir).parent]
    exe = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
    roots += [exe, exe / "models", config.app_home(), config.app_home() / "models", Path.home() / "Downloads",
              Path.home() / "Desktop"]
    for r in roots:
        try:
            hit = sorted(r.glob(PACK_GLOB))
        except OSError:
            hit = []
        if hit:
            return hit[-1]
    return None


@lru_cache(maxsize=2)
def _open(path: str):
    z = zipfile.ZipFile(path)
    return z, json.loads(z.read("index.json"))["entries"]


def index(models_dir: str | None = None) -> tuple[zipfile.ZipFile | None, dict]:
    p = find_pack(models_dir)
    if not p:
        return None, {}
    try:
        return _open(str(p))
    except (OSError, zipfile.BadZipFile, KeyError, ValueError):
        return None, {}


def key_for(r: dict, idx: dict) -> str | None:
    if r.get("category") in NO_PICTURE_CATS or r.get("language_code") or str(r.get("label", "")).startswith("Mains hum"):
        return None
    sci = (r.get("scientific_name") or "").strip().lower()
    if sci and "sp:" + sci in idx:
        return "sp:" + sci
    vt = (r.get("evidence") or {}).get("vehicle_type") or {}
    if vt.get("top") and vt.get("confident") and "vt:" + vt["top"] in idx:
        return "vt:" + vt["top"]
    if "as:" + str(r.get("label")) in idx:
        return "as:" + r["label"]
    for rx, name in LABEL_ALIASES:
        if rx.search(str(r.get("label", ""))) and "as:" + name in idx:
            return "as:" + name
    for c in r.get("candidates") or []:     # e.g. "Bird vocalization" record whose AudioSet label is a candidate
        if "as:" + str(c.get("label")) in idx:
            return "as:" + c["label"]
    return None


def attach(records: list[dict], models_dir: str | None) -> int:
    """Add r['image'] = {key, title, licence, artist, source} to every record that has a picture."""
    _, idx = index(models_dir)
    if not idx:
        return 0
    n = 0
    for r in records:
        k = key_for(r, idx)
        if k:
            e = idx[k]
            r["image"] = {"key": k, "title": e["title"], "licence": e["licence"], "artist": e["artist"],
                          "source": e["source"]}
            n += 1
    return n


def read(key: str, models_dir: str | None) -> bytes | None:
    z, idx = index(models_dir)
    e = idx.get(key)
    if not z or not e:
        return None
    with _lock:
        return z.read(e["img"])
