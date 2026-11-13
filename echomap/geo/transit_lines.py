"""Announced stops -> public-transport line, operator and a pin at the stop (v1.0.9).

Data: EchoMap-transit-v1.json.gz (tools/build_transit_lines.py, from Mobility Database GTFS feeds): every line with the
stop sequence of its longest trip. Transcribed speech is searched for stop names; a stop counts when it is
distinctive (multi-word or a rare single word) and either follows an announcement cue ("next stop", "volgende halte",
"nächster Halt", ...) or two or more heard stops lie on the same line. Lines that serve every heard stop are ranked,
with a bonus when the heard stops are consecutive on that line. The result is precise (the stop's coordinates), but
a stop name shared by several cities stays ambiguous until other clues (language, mains hum) decide between them.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from .. import config
from .places import CUES, _norm

PACK_GLOB = "EchoMap-transit*.json.gz"
GENERIC_MAX_PLACES = 25        # a stop name used in more places than this is generic ("Station", "Kerk")
MODE_TITLE = {"tram": "Tram", "metro": "Metro", "rail": "Train", "bus": "Bus", "ferry": "Ferry", "trolleybus": "Trolleybus",
              "funicular": "Funicular", "cable tram": "Cable car", "aerial lift": "Cable car", "monorail": "Monorail"}


def find_pack(models_dir: str | None = None) -> Path | None:
    roots = [Path(models_dir), Path(models_dir).parent] if models_dir else []
    exe = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2]
    roots += [exe, exe / "models", config.app_home(), config.app_home() / "models", Path.home() / "Downloads",
              Path.home() / "Desktop", Path(__file__).resolve().parents[1] / "data"]
    for r in roots:
        try:
            hit = sorted(r.glob(PACK_GLOB))
        except OSError:
            hit = []
        if hit:
            return hit[-1]
    return None


@lru_cache(maxsize=2)
def _load(path: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as g:
        d = json.load(g)
    by_name: dict[str, list[int]] = defaultdict(list)
    keys: list[set] = []
    for i, (name, _, _) in enumerate(d["stops"]):
        ks = {_norm(name)}
        if "," in name:                      # "Amsterdam, Centraal Station" -> also "centraal station"
            ks.add(_norm(name.split(",", 1)[1]))
        keys.append(ks)
        for k in ks:
            by_name[k].append(i)
    d["keys"] = keys
    lines_of: dict[int, list[int]] = defaultdict(list)
    for li, ln in enumerate(d["lines"]):
        for s in ln[4]:
            lines_of[s].append(li)
    d["by_name"], d["lines_of"] = dict(by_name), dict(lines_of)
    d["max_len"] = max((len(k.split()) for k in by_name), default=1)
    return d


def data(models_dir: str | None = None) -> dict | None:
    p = find_pack(models_dir)
    return _load(str(p)) if p else None


def line_label(d: dict, li: int) -> dict:
    op, mode, short, long, _ = d["lines"][li]
    operator, cc, city = d["operators"][op]
    name = f"{MODE_TITLE.get(mode, mode.title())} {short}".strip() if short else (long or MODE_TITLE.get(mode, mode))
    return {"line": name, "long_name": long, "mode": mode, "operator": operator, "country": cc, "city": city}


MODE_PRIORITY = ["tram", "metro", "rail", "bus", "trolleybus", "ferry", "funicular", "cable tram", "aerial lift", "monorail"]


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
    d = data(models_dir)
    if not d or not segments:
        return None
    heard: dict[str, dict] = {}
    for seg in segments:
        if seg.get("quality") not in ("clear", "uncertain"):
            continue
        text = _norm(seg.get("text", ""))
        words = text.split()
        for size in range(min(d["max_len"], 6), 0, -1):
            for i in range(len(words) - size + 1):
                name = " ".join(words[i:i + size])
                ids = d["by_name"].get(name)
                if not ids:
                    continue
                places = {(round(d["stops"][s][1], 1), round(d["stops"][s][2], 1)) for s in ids}
                if size == 1 and (len(name) < 6 or len(places) > 3):
                    continue
                if len(places) > GENERIC_MAX_PLACES:
                    continue
                before = " ".join(words[max(0, i - 5):i])
                cue = next((c for c in CUES if c in before), None)
                e = heard.setdefault(name, {"name": " ".join(words[i:i + size]), "ids": ids, "cue": None,
                                            "t": seg.get("start"), "text": seg.get("text", "")[:160]})
                e["cue"] = e["cue"] or cue
    if not heard:
        return None
    # lines serving the heard stops
    score: dict[int, set] = defaultdict(set)
    for name, e in heard.items():
        for s in e["ids"]:
            for li in d["lines_of"].get(s, []):
                score[li].add(name)
    if not score:
        return None
    best_n = max(len(v) for v in score.values())
    cands = []
    for li, names in score.items():
        if len(names) < best_n:
            continue
        seq = d["lines"][li][4]
        pos = sorted({j for j, sid in enumerate(seq) if d["keys"][sid] & names})
        adjacent = sum(1 for a, b in zip(pos, pos[1:]) if b - a == 1)
        cands.append((len(names) + 0.5 * adjacent, li, names))
    pref = list(prefer_modes or []) + MODE_PRIORITY

    def mrank(li):
        m = d["lines"][li][1]
        return pref.index(m) if m in pref else len(pref)
    cands.sort(key=lambda c: (-c[0], mrank(c[1]), str(d["lines"][c[1]][2]).startswith("N")))
    strong = best_n >= 2 or any(heard[n]["cue"] for n in cands[0][2])
    if not strong:
        return None
    out_lines, pins = [], {}
    for sc, li, names in cands[:max_lines * 4]:
        lab = line_label(d, li)
        if any(o["line"] == lab["line"] and o["operator"] == lab["operator"] for o in out_lines):
            continue
        stops = [d["stops"][s] for s in d["lines"][li][4] if d["keys"][s] & names]
        if not stops:
            continue
        if not lab["city"]:
            from .gazetteer import nearest_city
            nc = nearest_city(stops[0][1], stops[0][2])
            lab["city"] = (nc or {}).get("name", "")
        lab.update({"score": round(sc, 2), "stops_heard": [s[0] for s in stops],
                    "lat": round(sum(s[1] for s in stops) / len(stops), 5), "lon": round(sum(s[2] for s in stops) / len(stops), 5)})
        out_lines.append(lab)
        for s in stops:
            pins[(s[0], s[1], s[2])] = {"name": s[0], "lat": s[1], "lon": s[2], "city": lab["city"], "cc": lab["country"]}
        if len(out_lines) >= max_lines:
            break
    return {"heard": [{"stop": e["name"], "cue": e["cue"], "time_s": e["t"], "context": e["text"]} for e in heard.values()],
            "lines": out_lines, "pins": list(pins.values())[:40], "n_matching_lines": len(cands),
            "confidence": 0.7 if best_n >= 2 else 0.5,
            "basis": (f"{best_n} announced stops on the same line" if best_n >= 2 else "an announced stop after a cue phrase")}
