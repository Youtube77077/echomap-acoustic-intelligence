"""Offline gazetteer (GeoNames cities ≥ 15,000 inhabitants, CC BY 4.0). Nothing is sent online."""
from __future__ import annotations

import gzip
import json
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

from .grid import countries, haversine_km

DATA = Path(__file__).resolve().parents[1] / "data"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace("-", " ").split())


@lru_cache(maxsize=1)
def _cities():
    with gzip.open(DATA / "cities.json.gz", "rt", encoding="utf-8") as g:
        d = json.load(g)
    rows = d["rows"]
    lat = np.array([r[4] for r in rows], np.float32)
    lon = np.array([r[5] for r in rows], np.float32)
    return rows, lat, lon


def search_city(name: str, country: str | None = None, limit: int = 8) -> list[dict]:
    rows, _, _ = _cities()
    q = _norm(name)
    if not q:
        return []
    out = []
    for r in rows:
        if country and r[2] != country:
            continue
        n1, n2 = _norm(r[0]), _norm(r[1])
        if n1 == q or n2 == q:
            rank = 0
        elif n1.startswith(q) or n2.startswith(q):
            rank = 1
        else:
            continue
        out.append((rank, -r[6], r))
        if len(out) > 400:
            break
    out.sort(key=lambda t: (t[0], t[1]))
    return [{"name": r[0], "country": r[2], "country_name": countries().get(r[2], {}).get("name", r[2]),
             "admin1": r[3], "lat": r[4], "lon": r[5], "population": r[6]} for _, _, r in out[:limit]]


def cities_in_region(country: str, region: str) -> list[dict]:
    rows, _, _ = _cities()
    q = _norm(region)
    return [{"name": r[0], "lat": r[4], "lon": r[5], "population": r[6]} for r in rows
            if r[2] == country and q and (_norm(r[3]) == q or _norm(r[3]).startswith(q))]


def nearest_city(lat: float, lon: float) -> dict | None:
    rows, la, lo = _cities()
    d = haversine_km(lat, lon, la, lo)
    i = int(np.argmin(d))
    r = rows[i]
    return {"name": r[0], "country": r[2], "country_name": countries().get(r[2], {}).get("name", r[2]),
            "admin1": r[3], "distance_km": round(float(d[i]), 1), "population": r[6]}


def major_cities(min_pop: int = 150000) -> list[list]:
    """[name, lat, lon, population] for cities above min_pop (map labels)."""
    rows, _, _ = _cities()
    return [[r[0], r[4], r[5], r[6]] for r in rows if r[6] >= min_pop]


def country_list() -> list[dict]:
    return sorted(({"iso2": k, "name": v["name"]} for k, v in countries().items()), key=lambda d: d["name"])
