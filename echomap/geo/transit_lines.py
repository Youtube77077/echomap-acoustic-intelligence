"""Announced stops -> public-transport line, operator and a pin at the stop (v1.0.9).

Data: EchoMap-transit-v1.sqlite.gz (tools/build_transit_lines.py, from Mobility Database GTFS feeds, worldwide):
every line with the stop sequence of its longest trip. On first use it is unpacked once to
%LOCALAPPDATA%\\EchoMap\\ and queried on disk (a few MB of memory, not the whole dataset).

Transcribed speech is searched for stop names. A stop counts when it is distinctive (multi-word, or a rare single
word) and either follows an announcement cue ("next stop", "volgende halte", "nächster Halt", ...) or two or more heard
stops lie on the same line. Lines serving every heard stop are ranked (bonus when the stops are consecutive; the
transport mode heard in the sound breaks ties). The pin is the stop itself; a stop name shared by several cities stays
ambiguous until other clues (language, mains hum) decide.
"""
from __future__ import annotations

import gzip
import shutil
import sqlite3
import sys
import threading
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from .. import config
from .places import CUES, _norm

PACK_GLOBS = ("EchoMap-transit*.sqlite", "EchoMap-transit*.sqlite.gz")
DEFAULT_MODELS_DIR: str | None = None   # set by the pipeline so callers without a models dir can find the pack
GENERIC_MAX_PLACES = 25        # a stop name used in more places than this is generic ("Station", "Kerk")
MODE_TITLE = {"tram": "Tram", "metro": "Metro", "rail": "Train", "bus": "Bus", "ferry": "Ferry", "trolleybus": "Trolleybus",
              "funicular": "Funicular", "cable tram": "Cable car", "aerial lift": "Cable car", "monorail": "Monorail"}
MODE_PRIORITY = ["tram", "metro", "rail", "bus", "trolleybus", "ferry", "funicular", "cable tram", "aerial lift", "monorail"]
_lock = threading.Lock()


# ----------------------------------------------------------------------------- building (CI and tests)
def _keys(name: str) -> set[str]:
    ks = {_norm(name)}
    if "," in name:                      # "Amsterdam, Centraal Station" -> also "centraal station"
        ks.add(_norm(name.split(",", 1)[1]))
    return {k for k in ks if k}


def _pack(ids) -> bytes:
    import array
    return array.array("i", ids).tobytes()


def _unpack(b) -> list[int]:
    import array
    a = array.array("i")
    a.frombytes(b or b"")
    return a.tolist()


def build_sqlite(d: dict, path: Path) -> None:
    """d = {"stops": [[name, lat, lon]], "operators": [[name, cc, city]], "lines": [[op, mode, short, long, [stop ids]]]}"""
    path.unlink(missing_ok=True)
    con = sqlite3.connect(str(path))
    con.executescript("""
        CREATE TABLE stops(id INTEGER PRIMARY KEY, name TEXT, lat REAL, lon REAL, lines BLOB);
        CREATE TABLE operators(id INTEGER PRIMARY KEY, name TEXT, cc TEXT, city TEXT);
        CREATE TABLE lines(id INTEGER PRIMARY KEY, op INTEGER, mode TEXT, short TEXT, long TEXT, stops BLOB);
        CREATE TABLE keys(key TEXT PRIMARY KEY, places INTEGER, stops BLOB) WITHOUT ROWID;
        CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT);
        CREATE TABLE tram_networks(name TEXT, operator TEXT, cc TEXT, lat REAL, lon REAL, extent_km REAL, stops INTEGER);
    """)
    lines_of: dict[int, list[int]] = defaultdict(list)
    for li, l in enumerate(d["lines"]):
        for sid in dict.fromkeys(l[4]):
            lines_of[sid].append(li)
    con.executemany("INSERT INTO stops VALUES (?,?,?,?,?)", ((i, s[0], s[1], s[2], _pack(lines_of.get(i, [])))
                                                             for i, s in enumerate(d["stops"])))
    places: dict[str, set] = defaultdict(set)
    members: dict[str, list] = defaultdict(list)
    for i, s in enumerate(d["stops"]):
        for k in _keys(s[0]):
            members[k].append(i)
            places[k].add((round(s[1], 1), round(s[2], 1)))
    con.executemany("INSERT INTO keys VALUES (?,?,?)", ((k, len(places[k]), _pack(v)) for k, v in members.items()))
    con.executemany("INSERT INTO operators VALUES (?,?,?,?)", ((i, *o) for i, o in enumerate(d["operators"])))
    con.executemany("INSERT INTO lines VALUES (?,?,?,?,?,?)",
                    ((i, l[0], l[1], l[2], l[3] if l[3] != l[2] else "", _pack(l[4])) for i, l in enumerate(d["lines"])))
    con.execute("INSERT INTO meta VALUES ('max_len', ?)", (str(max((len(k.split()) for k in places), default=1)),))
    con.executemany("INSERT INTO tram_networks VALUES (?,?,?,?,?,?,?)",
                    ((n["name"], n["operator"], n["cc"], n["lat"], n["lon"], n["extent_km"], n["stops"])
                     for n in _derive_tram_networks(d)))
    con.commit()
    con.execute("VACUUM")
    con.close()


def _derive_tram_networks(d: dict) -> list[dict]:
    """One network per tram operator (split per city when an operator spans several): centre, extent, stop count."""
    import numpy as np
    from .gazetteer import nearest_city
    groups: dict[int, dict[tuple, set]] = defaultdict(lambda: defaultdict(set))
    for op, mode, _, _, ids in d["lines"]:
        if mode in ("tram", "cable tram"):
            for sid in ids:
                _, la, lo = d["stops"][sid]
                groups[op][(round(la * 2) / 2, round(lo * 2) / 2)].add(sid)
    nets = []
    for op, clusters in groups.items():
        allsids = set().union(*clusters.values())
        la = np.array([d["stops"][s][1] for s in allsids])
        lo = np.array([d["stops"][s][2] for s in allsids])
        clat, clon = float(np.median(la)), float(np.median(lo))
        spread = np.hypot((la - clat) * 111.0, (lo - clon) * 111.0 * np.cos(np.radians(clat)))
        parts = list(clusters.values()) if np.percentile(spread, 90) > 120 else [allsids]
        for ids in parts:
            la = np.array([d["stops"][s][1] for s in ids])
            lo = np.array([d["stops"][s][2] for s in ids])
            clat, clon = float(np.median(la)), float(np.median(lo))
            dist = np.hypot((la - clat) * 111.0, (lo - clon) * 111.0 * np.cos(np.radians(clat)))
            operator, cc, city = d["operators"][op]
            city = city or (nearest_city(clat, clon) or {}).get("name", "")
            nets.append({"name": city or operator, "operator": operator, "cc": cc, "lat": round(clat, 4),
                         "lon": round(clon, 4), "extent_km": round(float(min(40.0, max(3.0, np.percentile(dist, 90)))), 1),
                         "stops": int(len(ids))})
    return nets


# ----------------------------------------------------------------------------- runtime
def find_pack(models_dir: str | None = None) -> Path | None:
    models_dir = models_dir or DEFAULT_MODELS_DIR
    roots = [Path(models_dir), Path(models_dir).parent] if models_dir else []
    exe = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2]
    roots += [exe, exe / "models", config.app_home(), config.app_home() / "models", Path.home() / "Downloads",
              Path.home() / "Desktop"]
    for r in roots:
        for g in PACK_GLOBS:
            try:
                hit = sorted(r.glob(g))
            except OSError:
                hit = []
            if hit:
                return sorted(hit, key=lambda h: ("full" not in h.name, h.name))[0]   # the full pack (with buses) wins
    return None


@lru_cache(maxsize=2)
def _connect(path: str) -> sqlite3.Connection:
    p = Path(path)
    if p.suffix == ".gz":                     # unpack once next to the settings (queried on disk afterwards)
        dst = config.app_home() / p.name[:-3]
        if not dst.exists() or dst.stat().st_mtime < p.stat().st_mtime:
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".tmp")
            with gzip.open(p, "rb") as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
            tmp.replace(dst)
        p = dst
    return sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True, check_same_thread=False)


def db(models_dir: str | None = None) -> sqlite3.Connection | None:
    p = find_pack(models_dir)
    return _connect(str(p)) if p else None


def modes_heard(records: list[dict]) -> list[str]:
    """Transport modes suggested by the sound itself (used to rank lines that share a stop)."""
    names = " ".join(str(r.get("label", "")) + " " + str((r.get("evidence") or {}).get("vehicle_type", {}).get("top", ""))
                     for r in records).lower()
    out = []
    if "tram" in names:
        out.append("tram")
    if "subway" in names or "metro" in names:
        out.append("metro")
    if "train" in names or "rail" in names:
        out += ["rail", "metro", "tram"]
    if "bus" in names:
        out.append("bus")
    return out


def match(segments: list[dict], models_dir: str | None = None, max_lines: int = 5, prefer_modes=None) -> dict | None:
    con = db(models_dir)
    if con is None or not segments:
        return None
    with _lock:
        return _match(con, segments, max_lines, prefer_modes)


def _match(con, segments, max_lines, prefer_modes):
    q = con.execute
    max_len = int(q("SELECT v FROM meta WHERE k='max_len'").fetchone()[0])
    heard: dict[str, dict] = {}
    for seg in segments:
        if seg.get("quality") not in ("clear", "uncertain"):
            continue
        words = _norm(seg.get("text", "")).split()
        for size in range(min(max_len, 6), 0, -1):
            for i in range(len(words) - size + 1):
                name = " ".join(words[i:i + size])
                if size == 1 and len(name) < 6:
                    continue
                row = q("SELECT places FROM keys WHERE key=?", (name,)).fetchone()
                if not row or row[0] > GENERIC_MAX_PLACES or (size == 1 and row[0] > 3):
                    continue
                before = " ".join(words[max(0, i - 5):i])
                cue = next((c for c in CUES if c in before), None)
                e = heard.setdefault(name, {"name": " ".join(words[i:i + size]), "cue": None, "t": seg.get("start"),
                                            "text": seg.get("text", "")[:160]})
                e["cue"] = e["cue"] or cue
    if not heard:
        return None
    score: dict[int, set] = defaultdict(set)
    stops_of: dict[str, set] = {}
    for name in heard:
        stops_of[name] = set()
        for sid in _unpack(q("SELECT stops FROM keys WHERE key=?", (name,)).fetchone()[0]):
            stops_of[name].add(sid)
            for li in _unpack(q("SELECT lines FROM stops WHERE id=?", (sid,)).fetchone()[0]):
                score[li].add(name)
    if not score:
        return None
    best_n = max(len(v) for v in score.values())
    cand_ids = [li for li, v in score.items() if len(v) == best_n]
    if best_n < 2 and not any(heard[n]["cue"] for li in cand_ids for n in score[li]):
        return None
    pref = list(prefer_modes or []) + MODE_PRIORITY
    cands = []
    for li in cand_ids[:5000]:
        names = score[li]
        op, mode, short, long, blob = q("SELECT op, mode, short, long, stops FROM lines WHERE id=?", (li,)).fetchone()
        seq = _unpack(blob)
        want = set().union(*(stops_of[n] for n in names))
        hit = []
        for pos, sid in enumerate(seq):
            if sid in want:
                n, la, lo = q("SELECT name, lat, lon FROM stops WHERE id=?", (sid,)).fetchone()
                hit.append((pos, n, la, lo))
        pos = sorted({h[0] for h in hit})
        adjacent = sum(1 for a, b in zip(pos, pos[1:]) if b - a == 1)
        long = long or short
        rank = pref.index(mode) if mode in pref else len(pref)
        cands.append((-(len(names) + 0.5 * adjacent), rank, str(short).startswith("N"), li, op, mode, short, long, hit))
    cands.sort(key=lambda c: c[:4])
    out_lines, pins = [], {}
    for sc, _, _, li, op, mode, short, long, hit in cands:
        operator, cc, city = q("SELECT name, cc, city FROM operators WHERE id=?", (op,)).fetchone()
        stops = sorted({(h[1], h[2], h[3]) for h in hit})
        if not stops:
            continue
        if not city:
            from .gazetteer import nearest_city
            city = (nearest_city(stops[0][1], stops[0][2]) or {}).get("name", "")
        name = f"{MODE_TITLE.get(mode, mode.title())} {short}".strip() if short else (long or MODE_TITLE.get(mode, mode))
        if any(o["line"] == name and o["operator"] == operator for o in out_lines):
            continue
        out_lines.append({"line": name, "long_name": long, "mode": mode, "operator": operator, "country": cc, "city": city,
                          "score": round(-sc, 2), "stops_heard": sorted({s[0] for s in stops}),
                          "lat": round(sum(s[1] for s in stops) / len(stops), 5),
                          "lon": round(sum(s[2] for s in stops) / len(stops), 5)})
        for s in stops:
            pins[s] = {"name": s[0], "lat": s[1], "lon": s[2], "city": city, "cc": cc}
        if len(out_lines) >= max_lines:
            break
    return {"heard": [{"stop": e["name"], "cue": e["cue"], "time_s": e["t"], "context": e["text"]} for e in heard.values()],
            "lines": out_lines, "pins": list(pins.values())[:40], "n_matching_lines": len(cands),
            "confidence": 0.7 if best_n >= 2 else 0.5,
            "basis": (f"{best_n} announced stops on the same line" if best_n >= 2 else "an announced stop after a cue phrase")}


def tram_networks(models_dir: str | None = None) -> list[dict]:
    con = db(models_dir)
    if con is None:
        return []
    with _lock:
        rows = con.execute("SELECT name, operator, cc, lat, lon, extent_km, stops FROM tram_networks").fetchall()
    return [{"name": r[0], "operator": r[1], "cc": r[2], "lat": r[3], "lon": r[4], "extent_km": r[5], "stops": r[6],
             "source": "GTFS tram lines (Mobility Database)"} for r in rows]
