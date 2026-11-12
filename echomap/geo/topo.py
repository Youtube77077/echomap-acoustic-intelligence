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
