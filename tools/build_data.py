"""Build EchoMap's compact datasets from the approved raw sources in data-sources/.

Run once:  python tools/build_data.py
Outputs (committed): echomap/data/{countries.json, land_grid.npy, coastal_grid.npy, grid_meta.json,
                                   cities.json.gz, audioset.json}
Sources and licences: see THIRD_PARTY_NOTICES.md
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data-sources"
OUT = ROOT / "echomap" / "data"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from echomap.geo.topo import world_countries  # noqa: E402

# ------------------------------------------------------------------ mains frequency
# Source: Wikipedia "Mains electricity by country" (read 2026-11-12), transcribed.
# 50 is the default; listed here are 60 Hz and mixed territories.
MAINS_60 = set("AS AI AG AW BS BZ BM BR VG CA KY CO CR CU DO EC SV GU GT GY HT HN LR MX FM MS NI PW PA PE PH PR "
               "MF KN SA SX KR SR TW TT US VI VE MH".split())
MAINS_MIXED = {"JP": "east 50 Hz (Hokkaido, Tohoku, Kanto), west 60 Hz", "PF": "listed as 50/60"}

# CLDR language code -> Whisper language code (only where they differ)
CLDR_TO_WHISPER = {"gsw": "de", "nds": "de", "bar": "de", "ksh": "de", "yue": "zh", "wuu": "zh", "hak": "zh",
                   "nan": "zh", "nb": "no", "fil": "tl", "jv": "jw", "iw": "he", "zgh": "ber", "fy": "nl"}

# Natural Earth geometries without an ISO numeric id
NAME_TO_ISO2 = {"Kosovo": "XK", "N. Cyprus": "CY", "Somaliland": "SO", "Siachen Glacier": "IN",
                "Indian Ocean Ter.": "AU", "Ashmore and Cartier Is.": "AU", "Cyprus U.N. Buffer Zone": "CY",
                "Baikonur": "KZ", "Br. Indian Ocean Ter.": "IO", "Fr. S. Antarctic Lands": "TF"}


def build_countries():
    info = {}
    for line in (RAW / "geonames_countryInfo.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        info[f[0]] = {"iso2": f[0], "iso3": f[1], "numeric": f[2].zfill(3), "name": f[4],
                      "population": int(f[7] or 0), "continent": f[8]}
    terr = json.loads((RAW / "cldr_territoryInfo.json").read_text(encoding="utf-8"))["supplemental"]["territoryInfo"]
    out = {}
    for iso2, c in info.items():
        langs: dict[str, float] = {}
        for code, lp in (terr.get(iso2, {}).get("languagePopulation") or {}).items():
            base = code.split("_")[0]
            w = CLDR_TO_WHISPER.get(base, base)
            share = min(1.0, float(lp.get("_populationPercent", 0)) / 100.0)
            langs[w] = round(max(langs.get(w, 0.0), share), 4)
        mains = "50/60" if iso2 in MAINS_MIXED else (60 if iso2 in MAINS_60 else 50)
        out[iso2] = {**c, "mains_hz": mains, "mains_note": MAINS_MIXED.get(iso2, ""), "languages": langs}
    (OUT / "countries.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return out


def build_grid(countries):
    num_to_iso = {c["numeric"]: k for k, c in countries.items()}
    feats = world_countries()
    iso_list = [""]  # index 0 = no land / unknown
    res = 0.5
    lats = 90 - res / 2 - np.arange(int(180 / res)) * res
    lons = -180 + res / 2 + np.arange(int(360 / res)) * res
    LON, LAT = np.meshgrid(lons, lats)
    grid = np.zeros(LAT.shape, np.uint16)
    unmapped = []
    for f in feats:
        iso = num_to_iso.get(str(f["id"]).zfill(3)) if f["id"] else None
        iso = iso or NAME_TO_ISO2.get(f["name"])
        if not iso:
            unmapped.append(f["name"])
            continue
        geom = MultiPolygon([Polygon(p[0], p[1:]) for p in f["polygons"] if len(p[0]) >= 4]).buffer(0)
        minx, miny, maxx, maxy = geom.bounds
        m = (LON >= minx - res) & (LON <= maxx + res) & (LAT >= miny - res) & (LAT <= maxy + res)
        if not m.any():
            continue
        shapely.prepare(geom)
        inside = shapely.contains_xy(geom, LON[m], LAT[m])
        # also accept cells whose square overlaps the polygon (small islands)
        if not inside.any():
            c = geom.representative_point()
            r, k = int((90 - c.y) / res), int((c.x + 180) / res)
            idx = iso_list.index(iso) if iso in iso_list else (iso_list.append(iso) or len(iso_list) - 1)
            if grid[r, k] == 0:
                grid[r, k] = idx
            continue
        idx = iso_list.index(iso) if iso in iso_list else (iso_list.append(iso) or len(iso_list) - 1)
        sub = grid[m]
        sub[inside & (sub == 0)] = idx
        grid[m] = sub
    land = grid > 0
    # coastal = land cell with a non-land 8-neighbour (longitude wraps)
    nb = np.zeros_like(land)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                sh = np.roll(np.roll(~land, dr, axis=0), dc, axis=1)
                nb |= sh
    coastal = (land & nb).astype(np.uint8)
    np.save(OUT / "land_grid.npy", grid)
    np.save(OUT / "coastal_grid.npy", coastal)
    meta = {"resolution_deg": res, "lat0": float(lats[0]), "lon0": float(lons[0]), "iso_list": iso_list,
            "land_cells": int(land.sum()), "coastal_cells": int(coastal.sum()), "unmapped": unmapped,
            "source": "Natural Earth 1:50m (world-atlas 2.0.2), cell centre in polygon"}
    (OUT / "grid_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    print("grid:", meta["land_cells"], "land cells,", meta["coastal_cells"], "coastal; unmapped:", unmapped)


def build_cities():
    admin1 = {}
    for line in (RAW / "geonames_admin1CodesASCII.txt").read_text(encoding="utf-8").splitlines():
        f = line.split("\t")
        if len(f) >= 2:
            admin1[f[0]] = f[1]
    z = zipfile.ZipFile(RAW / "geonames_cities15000.zip")
    rows = []
    with z.open("cities15000.txt") as fh:
        for line in io.TextIOWrapper(fh, encoding="utf-8"):
            f = line.rstrip("\n").split("\t")
            rows.append([f[1], f[2], f[8], admin1.get(f"{f[8]}.{f[10]}", ""), round(float(f[4]), 4),
                         round(float(f[5]), 4), int(f[14] or 0)])
    rows.sort(key=lambda r: -r[6])
    with gzip.open(OUT / "cities.json.gz", "wt", encoding="utf-8") as g:
        json.dump({"fields": ["name", "ascii", "cc", "admin1", "lat", "lon", "pop"], "rows": rows}, g,
                  ensure_ascii=False, separators=(",", ":"))
    print("cities:", len(rows))


def build_audioset():
    onto = {n["id"]: n for n in json.loads((RAW / "audioset_ontology.json").read_text(encoding="utf-8"))}
    parent = {}
    for nid, n in onto.items():
        for c in n["child_ids"]:
            parent.setdefault(c, []).append(nid)

    def ancestors(mid, seen=None):
        seen = seen or []
        for p in parent.get(mid, []):
            if p not in seen:
                seen.append(p)
                ancestors(p, seen)
        return seen

    classes = []
    with open(RAW / "audioset_class_labels_indices.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mid = row["mid"]
            anc = ancestors(mid)
            classes.append({"idx": int(row["index"]), "mid": mid, "name": row["display_name"],
                            "ancestors": [onto[a]["name"] for a in anc if a in onto],
                            "description": onto.get(mid, {}).get("description", "")[:240]})
    (OUT / "audioset.json").write_text(json.dumps(classes, ensure_ascii=False, indent=0), encoding="utf-8")
    print("audioset classes:", len(classes))


if __name__ == "__main__":
    c = build_countries()
    print("countries:", len(c))
    build_grid(c)
    build_cities()
    build_audioset()
