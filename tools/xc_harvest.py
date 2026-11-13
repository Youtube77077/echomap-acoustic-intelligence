"""Xeno-canto metadata harvest + species where/when grids (CI only). Key from env XC_KEY (GitHub environment "Xeno").

Usage: python tools/xc_harvest.py --harvest <shard> <n_shards> <part.jsonl.gz>     (parallel page shards)
       python tools/xc_harvest.py <out_meta.jsonl.gz> <out_ranges.json.gz> <parts...>  (merge + ranges)

Pages through every recording of every group (birds, frogs, grasshoppers, bats, land mammals) via the Xeno-canto API v3
(500 per page, ~1 request/s). Keeps per recording: id, scientific name, English name, group, lat, lon, date, quality,
licence, length, download URL. From the coordinates and dates builds, per species, the occupied 1-degree cells (with
counts) and a 12-month histogram: EchoMap's fine-grained "where and when" layer for animals. The key is never printed.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
from collections import defaultdict

import requests

API = "https://xeno-canto.org/api/3/recordings"
GROUPS = ["birds", "frogs", "grasshoppers", "bats", '"land mammals"']
S = requests.Session()
S.headers["User-Agent"] = "EchoMap-xc-harvest/1.0 (offline species range statistics; 1 request/s)"


def page(query: str, n: int, key: str) -> dict:
    for attempt in range(8):
        try:
            r = S.get(API, params={"query": query, "page": n, "per_page": 500, "key": key}, timeout=120)
        except requests.RequestException as e:
            print(f"  retry ({type(e).__name__})", flush=True)
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(int(r.headers.get("Retry-After", 15 * (attempt + 1))))
            continue
        if r.status_code != 200:
            print(f"Xeno-canto API HTTP {r.status_code} for {query!r}: {r.text[:200]}", flush=True)
            return {}
        return r.json()
    raise SystemExit("Xeno-canto API kept failing")


def harvest(shard: int, nsh: int, out_path: str):
    """Pages (over all groups) with global index % nsh == shard -> partial metadata file."""
    key = os.environ.get("XC_KEY", "").strip()
    if not key:
        raise SystemExit("XC_KEY not set (GitHub environment 'Xeno', variable or secret XENO)")
    gidx, n = 0, 0
    with gzip.open(out_path, "wt", encoding="utf-8") as out:
        for grp in GROUPS:
            first = page(f"grp:{grp}", 1, key)
            if not first:
                print(f"group {grp}: skipped (query rejected)", flush=True)
                continue
            pages = int(first.get("numPages", 1))
            print(f"group {grp}: {first.get('numRecordings')} recordings, {pages} pages", flush=True)
            for p in range(1, pages + 1):
                gidx += 1
                if gidx % nsh != shard:
                    continue
                d = first if p == 1 else page(f"grp:{grp}", p, key)
                for r in d.get("recordings", []):
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n += 1
                if p % 20 == 0:
                    print(f"  {grp} page {p}/{pages}: {n} recordings so far", flush=True)
                time.sleep(1.0)
    print(f"DONE shard {shard}: {n} recordings")


def main():
    meta_out, rng_out = sys.argv[1], sys.argv[2]
    parts = sys.argv[3:]
    cells = defaultdict(lambda: defaultdict(int))
    months = defaultdict(lambda: [0] * 12)
    names, groups_of, n_all = {}, {}, 0
    def raw():
        for pp in parts:
            with gzip.open(pp, "rt", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)
    seen = set()
    with gzip.open(meta_out, "wt", encoding="utf-8") as out:
        for grp in ["merged"]:
            for p in [1]:
                for r in raw():
                    if r.get("id") in seen:
                        continue
                    seen.add(r.get("id"))
                    sci = f"{r.get('gen', '')} {r.get('sp', '')}".strip()
                    if not r.get("sp") or r.get("sp") == "mystery":
                        continue
                    lat, lon = r.get("lat"), r.get("lon", r.get("lng"))
                    rec = {"id": r.get("id"), "sci": sci, "en": r.get("en"), "grp": r.get("grp") or "",
                           "lat": lat, "lon": lon, "date": r.get("date"), "q": r.get("q"), "lic": r.get("lic"),
                           "len": r.get("length"), "file": r.get("file"), "cnt": r.get("cnt")}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_all += 1
                    k = sci.lower()
                    names[k], groups_of[k] = r.get("en"), rec["grp"]
                    try:
                        la, lo = float(lat), float(lon)
                        cells[k][(int(la // 1), int(lo // 1))] += 1
                    except (TypeError, ValueError):
                        pass
                    try:
                        m = int(str(r.get("date", ""))[5:7])
                        if 1 <= m <= 12:
                            months[k][m - 1] += 1
                    except ValueError:
                        pass
    ranges = {k: {"en": names[k], "grp": groups_of[k], "m": months[k],
                  "cells": [[a, b, n] for (a, b), n in sorted(cells[k].items())]} for k in names}
    with gzip.open(rng_out, "wt", encoding="utf-8") as g:
        json.dump({"source": "Xeno-canto API v3 (recording coordinates and dates)", "built": time.strftime("%Y-%m-%d"),
                   "recordings": n_all, "species": ranges}, g, separators=(",", ":"))
    print(f"DONE {n_all} recordings, {len(ranges)} species")


if __name__ == "__main__":
    if sys.argv[1] == "--harvest":
        harvest(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        main()
