"""Build EchoMap's tram-network dataset from OpenStreetMap (CI only; runs on GitHub's runner).

Usage: python tools/build_trams.py
Output (committed): echomap/data/tram_networks.json.gz  (ODbL 1.0, © OpenStreetMap contributors)

Method: every OSM way tagged railway=tram (track that exists today; disused/abandoned track uses other tags) is fetched
from the Overpass API as its centre point, in map tiles. Centres are snapped to a ~2 km raster and joined into
networks by 8-neighbour connectivity with one cell of dilation (lines within ~4 km of each other belong to one network).
Each network keeps its centroid, the number of track ways, an extent radius (90th percentile distance from the centroid
plus 1 km), its country from Natural Earth borders, and the largest GeoNames city (≥ 15,000 inhabitants)
in that country within the network's extent + 15 km for its name (otherwise 'near <closest city>').
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echomap.geo.gazetteer import _cities  # noqa: E402
from echomap.geo.grid import haversine_km  # noqa: E402
from echomap.geo.topo import country_at  # noqa: E402

OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
# GitHub runners have no IPv6 route: force IPv4 so hosts with AAAA records stay reachable
import urllib3.util.connection as _uc  # noqa: E402
_uc.HAS_IPV6 = False
UA = {"User-Agent": "EchoMap-dataset-builder/1.0 (one-off OSM extract for an offline acoustic analysis tool)"}
# tiles (south, west, north, east): small enough for the public Overpass server
LON = [-180, -100, -60, -10, 0, 5, 10, 15, 20, 30, 40, 60, 100, 125, 180]
TILES = [(s, w, n, e) for w, e in zip(LON[:-1], LON[1:]) for s, n in ((-60, 35), (35, 48), (48, 80))]
CELL = 0.02  # degrees, ~2 km


def fetch_tile(s: float, w: float, n: float, e: float) -> list[tuple[float, float]]:
    q = f'[out:json][timeout:240][maxsize:1073741824];way["railway"="tram"]({s},{w},{n},{e});out center;'
    for attempt in range(4):
        url = OVERPASS[attempt % len(OVERPASS)]
        try:
            r = requests.post(url, data={"data": q}, headers=UA, timeout=300)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
                els = r.json().get("elements", [])
                return [(e["center"]["lat"], e["center"]["lon"]) for e in els if "center" in e]
            print("overpass status", r.status_code, url, r.text[:200])
        except Exception as e:  # noqa: BLE001
            print("overpass error", url, e)
        time.sleep(10 * (attempt + 1))
    print(f"SKIPPED tile {s},{w},{n},{e} (all Overpass servers failed)")
    return None


def components(cells: np.ndarray) -> np.ndarray:
    """Label connected components of occupied raster cells (8-neighbour, one cell dilation)."""
    key = {tuple(c): i for i, c in enumerate(map(tuple, cells))}
    parent = list(range(len(cells)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, (r, c) in enumerate(cells):
        for dr in (-2, -1, 0, 1, 2):
            for dc in (-2, -1, 0, 1, 2):
                j = key.get((r + dr, c + dc))
                if j is not None:
                    a, b = find(i), find(j)
                    if a != b:
                        parent[a] = b
    return np.array([find(i) for i in range(len(cells))])


def main():
    pts, skipped = [], []
    for t in TILES:
        b = fetch_tile(*t)
        if b is None:
            skipped.append(list(t))
            continue
        print(f"tile {t}: {len(b)} tram ways")
        pts += b
        time.sleep(5)
    pts = sorted(set(pts))  # a way on a tile border is returned twice
    P = np.array(pts, np.float64)
    cells = np.stack([np.floor(P[:, 0] / CELL), np.floor(P[:, 1] / CELL)], 1).astype(np.int64)
    ucells, inv = np.unique(cells, axis=0, return_inverse=True)
    lab = components(ucells)[inv.ravel()]
    rows, clat, clon = _cities()
    nets = []
    for L in np.unique(lab):
        m = lab == L
        if m.sum() < 5:
            continue
        la, lo = P[m, 0], P[m, 1]
        x = np.radians(la)
        y = np.radians(lo)
        v = np.array([np.cos(x) * np.cos(y), np.cos(x) * np.sin(y), np.sin(x)]).mean(1)
        cla, clo = float(np.degrees(np.arcsin(v[2] / np.linalg.norm(v)))), float(np.degrees(np.arctan2(v[1], v[0])))
        d = haversine_km(cla, clo, la, lo)
        extent = float(np.percentile(d, 90)) + 1.0
        dc = haversine_km(cla, clo, clat, clon)
        cc = country_at(cla, clo) or country_at(float(la[0]), float(lo[0]))
        near = [k for k in np.where(dc <= max(extent, 10.0) + 15)[0] if cc is None or rows[k][2] == cc]
        if near:
            r = rows[max(near, key=lambda k: rows[k][6])]
            name = r[0]
        else:
            r = rows[int(np.argmin(dc))]
            name = f"near {r[0]}"
        nets.append({"name": name, "cc": cc or r[2], "admin1": r[3] if (cc is None or r[2] == cc) else "",
                     "lat": round(cla, 4), "lon": round(clo, 4), "ways": int(m.sum()), "extent_km": round(extent, 1)})
    nets.sort(key=lambda n: -n["ways"])
    meta = {"source": "OpenStreetMap railway=tram ways via Overpass API", "licence": "ODbL 1.0 (© OpenStreetMap contributors)",
            "built": time.strftime("%Y-%m-%d"), "tram_ways": len(P), "networks": len(nets),
            "countries": len({n["cc"] for n in nets}), "skipped_tiles": skipped, "method": __doc__.split("Method: ")[1].strip()}
    out = ROOT / "echomap" / "data" / "tram_networks.json.gz"
    with gzip.open(out, "wt", encoding="utf-8") as g:
        json.dump({"meta": meta, "networks": nets}, g, ensure_ascii=False, separators=(",", ":"))
    print(json.dumps(meta, ensure_ascii=False))
    print("CZ:", [(n["name"], n["ways"], n["extent_km"]) for n in nets if n["cc"] == "CZ"])


if __name__ == "__main__":
    main()
