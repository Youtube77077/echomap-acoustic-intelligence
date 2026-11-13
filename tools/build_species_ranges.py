"""Where and when every identifiable species occurs, from GBIF occurrence statistics (CI only).

Usage: python tools/build_species_ranges.py <shard> <n_shards> <out.json.gz> <names.txt>
       python tools/build_species_ranges.py --merge <echomap/data/species_ranges.json.gz> <parts...>

For each scientific name (BirdNET+ labels and iNatSounds species): GBIF name match -> occurrence counts per country
(facet) and per month (facet), using only occurrences with coordinates and no geospatial issues. Stored compactly:
{name: {"k": gbif key, "n": total, "c": {ISO2: count}, "m": [12 monthly counts]}}. EchoMap uses this for species that
the BirdNET+ geomodel does not cover, and for the "is this species expected here / in this month?" check.
GBIF data: CC0 / CC BY per dataset (aggregated counts only). Nothing runs on the user's PC.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import requests

API = "https://api.gbif.org/v1"
S = requests.Session()
S.headers["User-Agent"] = "EchoMap-range-builder/1.0 (offline species range statistics; low request rate)"


def get(url, params):
    for attempt in range(6):
        try:
            r = S.get(url, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code in (429, 503):
            time.sleep(int(r.headers.get("Retry-After", 10 * (attempt + 1))))
            continue
        if r.status_code != 200:
            return None
        return r.json()
    return None


def one(name: str) -> dict | None:
    m = get(f"{API}/species/match", {"name": name, "strict": "true"})
    if not m or m.get("matchType") in (None, "NONE", "HIGHERRANK") or not m.get("usageKey"):
        return None
    key = m.get("acceptedUsageKey") or m["usageKey"]
    f = get(f"{API}/occurrence/search", {"taxonKey": key, "limit": 0, "hasCoordinate": "true",
                                          "hasGeospatialIssue": "false", "facet": ["country", "month"],
                                          "facetLimit": 300, "country.facetLimit": 300})
    if not f:
        return None
    out = {"k": key, "n": f.get("count", 0), "c": {}, "m": [0] * 12}
    for fa in f.get("facets", []):
        if fa["field"] == "COUNTRY":
            out["c"] = {c["name"]: c["count"] for c in fa["counts"]}
        elif fa["field"] == "MONTH":
            for c in fa["counts"]:
                if str(c["name"]).isdigit() and 1 <= int(c["name"]) <= 12:
                    out["m"][int(c["name"]) - 1] = c["count"]
    return out


def main():
    shard, n, out, names_f = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    names = sorted({l.strip() for l in names_f.read_text(encoding="utf-8").splitlines() if l.strip()})
    names = [x for i, x in enumerate(names) if i % n == shard]
    res, miss, t0 = {}, 0, time.time()
    for i, nm in enumerate(names):
        r = one(nm)
        if r:
            res[nm.lower()] = r
        else:
            miss += 1
        time.sleep(0.15)
        if i % 200 == 0:
            print(f"{i}/{len(names)} done, {len(res)} found, {miss} missing, {time.time() - t0:.0f} s", flush=True)
    with gzip.open(out, "wt", encoding="utf-8") as g:
        json.dump(res, g, separators=(",", ":"))
    print(f"DONE shard {shard}: {len(res)} species, {miss} missing")


def merge(dst: Path, parts: list[Path]):
    res = {}
    for p in parts:
        with gzip.open(p, "rt", encoding="utf-8") as g:
            res.update(json.load(g))
    with gzip.open(dst, "wt", encoding="utf-8") as g:
        json.dump({"source": "GBIF occurrence facets (country, month), coordinates without geospatial issues",
                   "built": time.strftime("%Y-%m-%d"), "species": res}, g, separators=(",", ":"))
    print(f"MERGED {len(res)} species -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    if sys.argv[1] == "--merge":
        merge(Path(sys.argv[2]), [Path(x) for x in sys.argv[3:]])
    else:
        main()
