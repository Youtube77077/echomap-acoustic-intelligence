"""Point-location clues (v1.0.3): tram networks and place names spoken in the recording.

Tram networks
    echomap/data/tram_networks.json.gz is built from OpenStreetMap railway=tram track (tools/build_trams.py, ODbL).
    The AudioSet sound model has no "tram" class, so a tram is only hypothesised when
      (a) an event closely resembles a tram recording in the reference library (embedding similarity tier), or
      (b) rail-vehicle sound co-occurs with a bell/ding or wheel squeal (a weak rule, low confidence).
    The likelihood is a kernel around every tram network, scaled by sqrt(network size) relative to the largest
    network (bigger networks offer more places to hear a tram), with a floor because rail sounds are ambiguous.

Spoken place names
    Transcribed speech is searched for names of GeoNames cities (≥ 15,000 inhabitants). Mentioning a place does not
    mean being there (people talk about other cities), so the evidence is moderate and floored; announcement cues
    ("next stop", "příští zastávka", ...) raise confidence. Only Latin-script names are matched in this version.
"""
from __future__ import annotations

import gzip
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

from .grid import countries, haversine_km

DATA = Path(__file__).resolve().parents[1] / "data"

RAIL_LABELS = {"Rail transport", "Train", "Railroad car, train wagon", "Train wheels squealing", "Subway, metro, underground"}
BELL_LABELS = {"Bell", "Ding", "Bicycle bell", "Ding-dong", "Jingle bell"}
SQUEAL = "Train wheels squealing"

# announcement / arrival phrases (lower-case, accents removed) that make a following place name more meaningful
CUES = ["next stop", "next station", "the next stop is", "now arriving", "arriving at", "welcome to", "this is", "change here for",
        "pristi zastavka", "pristi stanice", "konecna stanice", "nachster halt", "nachste haltestelle", "nachste station",
        "endstation", "prochain arret", "prochaine station", "terminus", "proxima parada", "proxima estacion", "proxima estacao",
        "prossima fermata", "volgende halte", "volgende station", "nastepny przystanek", "nastepna stacja", "nasta station",
        "nasta hallplats", "neste stopp", "neste stasjon", "naeste station", "seuraava pysakki", "kovetkezo megallo",
        "urmatoarea statie", "sledeca stanica", "sledeci stanica", "pristi postaja", "sljedeca stanica", "dalsia zastavka",
        "pakopa", "sonraki durak", "sonraki istasyon", "kitas stotelė", "nakamaru"]
# frequent words that are also city names; matched only with an announcement cue
COMMON = set("""
nice bath reading split mobile best mean bend hope union normal liberty independence victoria florence marion
york salem eden essen ede hall halle born borne bergen dover march orange temple deal sale wells wels rome
alexandria aurora concord commerce enterprise opportunity paradise pleasant plain plains spring springs story
stockton troy vista warren worth brest most tours nancy angers grasse bastia lens cannes vienne roanne bra
sur cali leon mar mara santa santo san sao porto rio paz lima baker gary george lincoln jackson madison
washington franklin clinton columbus hamilton lafayette marshall monroe phoenix polo royal sandy sterling
""".split())


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower().replace("-", " ")).split())


# ---------------------------------------------------------------- tram networks
@lru_cache(maxsize=1)
def tram_networks() -> dict:
    p = DATA / "tram_networks.json.gz"
    if not p.exists():
        return {"meta": {}, "networks": []}
    with gzip.open(p, "rt", encoding="utf-8") as g:
        return json.load(g)


def tram_hypothesis(records: list[dict], references: dict | None) -> dict | None:
    """Return {"c", "record", "basis"} when the recording plausibly contains a tram, else None."""
    best = None
    for e in (references or {}).get("events", []):
        for m in e.get("matches", []):
            if m.get("category") != "tram":
                continue
            c = {"strong acoustic similarity": 0.55, "same-category similarity": 0.35}.get(m.get("tier"), 0.0)
            if c and (best is None or c > best["c"]):
                best = {"c": c, "record": e["record"],
                        "basis": f"{m['tier']} ({m['similarity']}) to the tram recording '{m['title'][:60]}'"}
    # (a2) the trained vehicle-type head says "tram" (MELAUDIS; confidence scaled by its measured tram precision)
    from ..models.vehicle_head import tram_precision
    prec = tram_precision() or 0.5
    for r in records:
        vt = (r.get("evidence") or {}).get("vehicle_type")
        if vt and vt["top"] == "tram" and vt["p"] >= 0.6:
            c = round(min(0.6, 0.6 * prec * vt["p"]), 3)
            if best is None or c > best["c"]:
                best = {"c": c, "record": r["id"],
                        "basis": f"vehicle-type model says tram (p = {vt['p']}; tram precision on held-out sites {prec:.2f}; trained in Melbourne)"}
    rail = [r for r in records if r["label"] in RAIL_LABELS and r["confidence"] >= 0.3]
    if rail:
        bells = [r for r in records if r["label"] in BELL_LABELS and r["confidence"] >= 0.2]
        for r in rail:
            near = [b for b in bells if b["start"] - 15 <= r["start"] <= b["end"] + 15]
            squeal = r["label"] == SQUEAL
            if near or squeal:
                c = 0.3
                if best is None or c > best["c"]:
                    why = "rail-vehicle sound with wheel squeal" if squeal else f"rail-vehicle sound with a {near[0]['label'].lower()} nearby"
                    best = {"c": c, "record": r["id"], "basis": why + " (weak rule: trains and metros can sound alike)"}
    return best


def kernel_sigma(extent_km: float, pts) -> np.ndarray:
    """Kernel width: the clue's own size, widened by half a grid cell so coarse grids do not miss a small place."""
    half_cell = 0.5 * np.sqrt(np.asarray(pts.area_km2, np.float64))
    return np.hypot(max(extent_km, 2.0), half_cell)


def tram_likelihood(pts, nets: list[dict]) -> np.ndarray:
    if not nets:
        return np.full(len(pts.lat), 1.0, np.float32)
    wmax = max(n["ways"] for n in nets)
    L = np.zeros(len(pts.lat), np.float64)
    for n in nets:
        d = haversine_km(n["lat"], n["lon"], pts.lat, pts.lon)
        near = d < 400
        if not near.any():
            continue
        s = kernel_sigma(n["extent_km"], pts)[near] if np.ndim(pts.area_km2) else kernel_sigma(n["extent_km"], pts)
        size = max(0.3, float(np.sqrt(n["ways"] / wmax)))
        L[near] = np.maximum(L[near], size * np.exp(-0.5 * (d[near] / s) ** 2))
    return np.maximum(L, 0.03).astype(np.float32)


# ---------------------------------------------------------------- spoken place names
@lru_cache(maxsize=1)
def _name_index() -> dict:
    from .gazetteer import _cities
    rows, _, _ = _cities()
    idx: dict[str, list] = {}
    for r in rows:
        for n in {_norm(r[0]), _norm(r[1])}:
            if len(n) >= 4:
                idx.setdefault(n, []).append(r)
    return idx


CAP_SEQ = re.compile(r"(?<![\w])([A-ZÀ-ÖØ-ÞĀ-Ž][\w'’\-]+(?:\s+(?:[A-ZÀ-ÖØ-ÞĀ-Ž][\w'’\-]+|nad|pod|am|an|der|de|del|la|le|sur|upon|on)){0,3})")


def place_mentions(segments: list[dict]) -> list[dict]:
    """Find city names in transcript segments. Returns one entry per distinct mentioned city name."""
    idx = _name_index()
    found: dict[str, dict] = {}
    for seg in segments or []:
        if seg.get("quality") == "likely spurious" or not seg.get("text"):
            continue
        text = seg["text"]
        ntext = _norm(text)
        cue = next((c for c in CUES if c in ntext), None)
        for m in CAP_SEQ.finditer(text):
            words = m.group(1).split()
            sentence_start = m.start() == 0 or text[:m.start()].rstrip()[-1:] in ".!?"
            # longest sub-span first
            done = False
            for size in range(len(words), 0, -1):
                for i in range(0, len(words) - size + 1):
                    name = _norm(" ".join(words[i:i + size]))
                    rows = idx.get(name)
                    if not rows:
                        continue
                    top_pop = max(r[6] for r in rows)
                    if size == 1 and not cue:
                        if name in COMMON or top_pop < 50_000 or (sentence_start and i == 0 and top_pop < 300_000):
                            continue
                    if name in COMMON and not cue:
                        continue
                    e = found.setdefault(name, {"name": " ".join(words[i:i + size]), "norm": name, "cue": None,
                                                "segments": [], "quality": "uncertain", "rows": rows})
                    e["segments"].append({"start": seg.get("start"), "end": seg.get("end"), "text": text[:200]})
                    e["cue"] = e["cue"] or cue
                    if seg.get("quality") == "clear":
                        e["quality"] = "clear"
                    e["language"] = seg.get("language")
                    done = True
                    break
                if done:
                    break
    return list(found.values())


def mention_confidence(m: dict) -> float:
    c = 0.35 if m["quality"] == "clear" else 0.2
    if m["cue"]:
        c += 0.25
    if len(m["segments"]) >= 2:
        c += 0.05
    return min(c, 0.7)


def mention_likelihood(pts, m: dict) -> tuple[np.ndarray, list[dict]]:
    """Mixture over all cities with this name, weighted by population and by the spoken language's share there."""
    lang = m.get("language")
    cs = countries()
    rows = m["rows"]
    w = np.array([r[6] * (0.05 + cs.get(r[2], {}).get("languages", {}).get(lang, 0.0) if lang else r[6]) for r in rows], np.float64)
    w = w / w.max()
    keep = [i for i in np.argsort(-w)[:12] if w[i] >= 0.02]
    L = np.zeros(len(pts.lat), np.float64)
    out = []
    for i in keep:
        r = rows[i]
        rad = 6.0 + 4.0 * np.log10(max(r[6], 15000) / 15000)  # rough city radius from population
        d = haversine_km(r[4], r[5], pts.lat, pts.lon)
        L = np.maximum(L, w[i] * np.exp(-0.5 * (d / kernel_sigma(rad, pts)) ** 2))
        out.append({"name": r[0], "cc": r[2], "admin1": r[3], "lat": r[4], "lon": r[5], "population": r[6],
                    "weight": round(float(w[i]), 3), "radius_km": round(float(rad), 1)})
    return np.maximum(L, 0.05).astype(np.float32), out
