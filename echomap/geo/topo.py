"""Minimal TopoJSON decoder (spec: https://github.com/topojson/topojson-specification)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


def _decode_arcs(topo: dict) -> list[list[tuple[float, float]]]:
    tr = topo.get("transform")
    out = []
    for arc in topo["arcs"]:
        pts = []
        x = y = 0
        for p in arc:
            if tr:
                x += p[0]
                y += p[1]
                pts.append((x * tr["scale"][0] + tr["translate"][0], y * tr["scale"][1] + tr["translate"][1]))
            else:
                pts.append((p[0], p[1]))
        out.append(pts)
    return out


def _ring(arcs: list, idx: list[int]) -> list[tuple[float, float]]:
    ring: list[tuple[float, float]] = []
    for i in idx:
        a = arcs[i] if i >= 0 else arcs[~i][::-1]
        ring.extend(a if not ring else a[1:])
    return ring


def features(topo: dict, obj: str) -> list[dict]:
    """Return GeoJSON-like features: {"id", "name", "polygons": [[outer, hole, ...], ...]}."""
    arcs = _decode_arcs(topo)
    out = []
    for g in topo["objects"][obj]["geometries"]:
        polys = []
        if g["type"] == "Polygon":
            polys = [[_ring(arcs, r) for r in g["arcs"]]]
        elif g["type"] == "MultiPolygon":
            polys = [[_ring(arcs, r) for r in poly] for poly in g["arcs"]]
        out.append({"id": g.get("id"), "name": (g.get("properties") or {}).get("name"), "polygons": polys})
    return out


@lru_cache(maxsize=1)
def world_countries() -> list[dict]:
    p = Path(__file__).resolve().parents[1] / "web" / "vendor" / "countries-50m.json"
    return features(json.loads(p.read_text(encoding="utf-8")), "countries")


_NAME_TO_ISO2 = {"Kosovo": "XK", "N. Cyprus": "CY", "Somaliland": "SO", "Siachen Glacier": "IN", "Indian Ocean Ter.": "AU",
                 "Ashmore and Cartier Is.": "AU", "Cyprus U.N. Buffer Zone": "CY", "Baikonur": "KZ",
                 "Br. Indian Ocean Ter.": "IO", "Fr. S. Antarctic Lands": "TF"}


@lru_cache(maxsize=1)
def _country_polys():
    from .grid import countries
    num = {v.get("numeric"): k for k, v in countries().items()}
    out = []
    for f in world_countries():
        iso = num.get(str(f["id"]).zfill(3)) if f["id"] is not None else None
        iso = iso or _NAME_TO_ISO2.get(f["name"] or "")
        if not iso:
            continue
        for poly in f["polygons"]:
            ring = poly[0]
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            out.append((iso, (min(xs), min(ys), max(xs), max(ys)), poly))
    return out


def _in_ring(x: float, y: float, ring) -> bool:
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi:
            inside = not inside
        j = i
    return inside


def country_at(lat: float, lon: float) -> str | None:
    """ISO2 code of the Natural Earth 1:50m country containing the point (None at sea or unmapped)."""
    for iso, (x0, y0, x1, y1), poly in _country_polys():
        if x0 <= lon <= x1 and y0 <= lat <= y1 and _in_ring(lon, lat, poly[0]) and not any(_in_ring(lon, lat, h) for h in poly[1:]):
            return iso
    return None
