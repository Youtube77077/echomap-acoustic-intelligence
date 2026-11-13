"""Worldwide public-transport lines with their stops, from GTFS feeds in the Mobility Database (CI only).

Usage: python tools/build_transit_lines.py <shard> <n_shards> <out.json.gz> [max_feeds]
       python tools/build_transit_lines.py --merge <EchoMap-transit-v1.json.gz> <shard files...>

For every usable GTFS feed (not deprecated, not redirected; mirrored by the Mobility Database), keep each route
(tram, metro, rail, ferry, cable car, funicular, trolleybus and bus) with the stop sequence of its longest trip in each
direction. Stop names and coordinates come from stops.txt (child platforms folded into their parent station).
EchoMap uses this to turn announced stop names in transcripts into a line, an operator and a pin at the stop.
Feed licences vary (recorded per feed); the data is used offline inside EchoMap for personal, non-commercial use.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import requests

SOURCES = "https://storage.googleapis.com/storage/v1/b/mdb-csv/o/sources.csv?alt=media"
MODES = {0: "tram", 1: "metro", 2: "rail", 3: "bus", 4: "ferry", 5: "cable tram", 6: "aerial lift", 7: "funicular",
         11: "trolleybus", 12: "monorail"}
MAX_BYTES = 900e6


def mode_of(rt: str) -> str | None:
    try:
        t = int(rt)
    except ValueError:
        return None
    if t in MODES:
        return MODES[t]
    if 100 <= t < 200:
        return "rail"
    if 200 <= t < 300 or 700 <= t < 800:
        return "bus"
    if 400 <= t < 500:
        return "metro"
    if 800 <= t < 900:
        return "trolleybus"
    if 900 <= t < 1000:
        return "tram"
    if 1000 <= t < 1300:
        return "ferry"
    if 1300 <= t < 1500:
        return "aerial lift"
    return None


def rows(z: zipfile.ZipFile, name: str):
    n = next((m for m in z.namelist() if m.split("/")[-1] == name), None)
    if not n:
        return
    with z.open(n) as fh:
        yield from csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8-sig", errors="replace"))


def feed_lines(path: Path, meta: dict) -> list[dict]:
    z = zipfile.ZipFile(path)
    routes = {}
    for r in rows(z, "routes.txt"):
        m = mode_of(r.get("route_type", ""))
        if m:
            routes[r["route_id"]] = {"short": (r.get("route_short_name") or "").strip(),
                                     "long": (r.get("route_long_name") or "").strip(), "mode": m,
                                     "agency": r.get("agency_id", "")}
    agencies = {a.get("agency_id", ""): a.get("agency_name", "") for a in rows(z, "agency.txt")}
    trips = defaultdict(list)          # (route, direction) -> trip ids
    trip_key = {}
    for t in rows(z, "trips.txt"):
        if t["route_id"] in routes:
            k = (t["route_id"], t.get("direction_id", ""))
            if len(trips[k]) < 40:     # a sample of trips is enough to find the longest pattern
                trips[k].append(t["trip_id"])
                trip_key[t["trip_id"]] = k
    seqs = defaultdict(list)
    for st in rows(z, "stop_times.txt"):
        tid = st.get("trip_id")
        if tid in trip_key:
            try:
                seqs[tid].append((int(st.get("stop_sequence") or 0), st["stop_id"]))
            except ValueError:
                pass
    stops, parent = {}, {}
    for s in rows(z, "stops.txt"):
        try:
            stops[s["stop_id"]] = (s.get("stop_name", "").strip(), round(float(s["stop_lat"]), 5), round(float(s["stop_lon"]), 5))
        except (ValueError, KeyError):
            continue
        if s.get("parent_station"):
            parent[s["stop_id"]] = s["parent_station"]
    out = []
    for (rid, d), tids in trips.items():
        best = max((sorted(seqs[t]) for t in tids if seqs.get(t)), key=len, default=None)
        if not best or len(best) < 2:
            continue
        seq, last = [], None
        for _, sid in best:
            sid = parent.get(sid, sid) if parent.get(sid) in stops else sid
            if sid in stops and stops[sid][0] and stops[sid][0] != last:
                seq.append(stops[sid])
                last = stops[sid][0]
        r = routes[rid]
        out.append({"feed": meta["mdb_source_id"], "operator": agencies.get(r["agency"]) or next(iter(agencies.values()), "")
                    or meta["provider"], "country": meta["location.country_code"], "city": meta["location.municipality"],
                    "mode": r["mode"], "short": r["short"], "long": r["long"], "dir": d, "stops": seq})
    # merge the two directions when they serve the same stops (keeps the file small)
    merged, seen = [], {}
    for ln in out:
        key = (ln["operator"], ln["mode"], ln["short"], ln["long"], frozenset(s[0] for s in ln["stops"]))
        if key not in seen:
            seen[key] = True
            merged.append(ln)
    return merged


def main():
    shard, n, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
    S = requests.Session()
    S.headers["User-Agent"] = "EchoMap-transit-builder/1.0 (one-off offline dataset build)"
    src = list(csv.DictReader(io.StringIO(S.get(SOURCES, timeout=120).content.decode("utf-8"))))
    feeds = [f for f in src if f["data_type"] == "gtfs" and f["status"] != "deprecated" and not f["redirect.id"]
             and f["urls.latest"]]
    feeds = [f for i, f in enumerate(feeds) if i % n == shard][:cap]
    print(f"shard {shard}/{n}: {len(feeds)} feeds", flush=True)
    lines, log = [], []
    tmp = out.with_suffix(".tmp.zip")
    t0 = time.time()
    for k, f in enumerate(feeds):
        try:
            with S.get(f["urls.latest"], stream=True, timeout=300) as r:
                r.raise_for_status()
                size = int(r.headers.get("x-goog-stored-content-length") or r.headers.get("content-length") or 0)
                if size > MAX_BYTES:
                    log.append({"feed": f["mdb_source_id"], "skipped": f"{size / 1e6:.0f} MB"})
                    continue
                with open(tmp, "wb") as fh:
                    for ch in r.iter_content(1 << 20):
                        fh.write(ch)
            got = feed_lines(tmp, f)
            lines += got
            log.append({"feed": f["mdb_source_id"], "provider": f["provider"], "country": f["location.country_code"],
                        "lines": len(got), "licence": f["urls.license"]})
        except Exception as e:  # noqa: BLE001
            log.append({"feed": f["mdb_source_id"], "error": f"{type(e).__name__}: {e}"[:200]})
        if k % 10 == 0:
            print(f"  {k + 1}/{len(feeds)} feeds, {len(lines)} lines, {time.time() - t0:.0f} s", flush=True)
    tmp.unlink(missing_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as g:
        json.dump({"lines": lines, "feeds": log}, g, ensure_ascii=False, separators=(",", ":"))
    print(f"DONE shard {shard}: {len(lines)} lines from {sum(1 for l in log if 'lines' in l)} feeds, "
          f"{out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f} s", flush=True)




def merge(parts: list[Path], out: Path):
    """Combine shard files into the compact pack the app reads: shared stop table + lines referencing it."""
    stop_ix, stops, ops, op_ix, lines, feeds = {}, [], [], {}, [], []
    for p in parts:
        with gzip.open(p, "rt", encoding="utf-8") as g:
            d = json.load(g)
        feeds += d["feeds"]
        for ln in d["lines"]:
            ids = []
            for s in ln["stops"]:
                key = (s[0], s[1], s[2])
                if key not in stop_ix:
                    stop_ix[key] = len(stops)
                    stops.append([s[0], s[1], s[2]])
                ids.append(stop_ix[key])
            op = (ln["operator"], ln["country"], ln["city"])
            if op not in op_ix:
                op_ix[op] = len(ops)
                ops.append(list(op))
            lines.append([op_ix[op], ln["mode"], ln["short"], ln["long"], ids])
    with gzip.open(out, "wt", encoding="utf-8") as g:
        json.dump({"version": 1, "source": "Mobility Database GTFS feeds", "stops": stops, "operators": ops,
                   "lines": lines, "feeds": feeds}, g, ensure_ascii=False, separators=(",", ":"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from echomap.geo.transit_lines import build_sqlite
    import shutil
    for tag, keep in (("full", None), ("rail", {"tram", "metro", "rail", "ferry", "funicular", "cable tram", "aerial lift",
                                                 "monorail"})):
        sel = [l for l in lines if keep is None or l[1] in keep]
        used = sorted({sid for l in sel for sid in l[4]})
        remap = {old: new for new, old in enumerate(used)}
        db = out.with_name(f"EchoMap-transit-{tag}-v1.sqlite")
        build_sqlite({"stops": [stops[i] for i in used], "operators": ops,
                      "lines": [[l[0], l[1], l[2], l[3], [remap[x] for x in l[4]]] for l in sel]}, db)
        with open(db, "rb") as src, gzip.open(str(db) + ".gz", "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        print(f"SQLITE {tag}: {len(sel)} lines, {db.stat().st_size / 1e6:.0f} MB, gz {Path(str(db) + '.gz').stat().st_size / 1e6:.0f} MB")
    print(f"MERGED {len(lines)} lines, {len(stops)} stops, {len(ops)} operators, "
          f"{sum(1 for f in feeds if 'lines' in f)} feeds, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    if sys.argv[1] == "--merge":
        merge([Path(x) for x in sys.argv[3:]], Path(sys.argv[2]))
    else:
        main()
