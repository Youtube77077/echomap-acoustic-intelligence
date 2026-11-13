"""Soundscape benchmark: 10 real recordings with known coordinates, full Deep analysis, distance errors, screenshots.

Usage: python tools/soundscape_bench.py <models_dir> <work_dir>
Outputs: docs/eval/soundscape_set.json (the fixed set, reused on later runs so results are comparable),
         docs/eval/soundscapes.json + docs/eval/SOUNDSCAPES.md (errors per recording and summary),
         docs/screenshots/bench/*.png (Findings, analysis, inventory, map, "where found" for each recording).
Recordings: Wikimedia Commons sound files with coordinates (free licences), mixed scenes (rain, rainforest, wind,
streets with talking, markets, stations, bells, coast, night insects...), topped up with iNaturalist research-grade
non-bird recordings when Commons has too few. Audio is never committed. Test recordings are excluded from EchoMap's
reference library during the run (ECHOMAP_EXCLUDE_REFS), so a recording cannot match itself.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TAG = os.environ.get("BENCH_NAME", "")          # e.g. "ireland": separate set, results and screenshots
SUF = f"_{TAG}" if TAG else ""
SETF = ROOT / "docs" / "eval" / f"soundscape_set{SUF}.json"
SHOTS = ROOT / "docs" / "screenshots" / f"bench{SUF}"
S = requests.Session()
S.headers["User-Agent"] = (f"EchoMap-evaluation/1.0 (private research tool; low volume; "
                           f"contact: {os.environ.get('WIKIMEDIA_CONTACT', 'not set')})")
API = "https://commons.wikimedia.org/w/api.php"
QUERIES = ["rainforest ambience", "tropical forest night insects", "rain ambience", "thunderstorm rain", "street ambience",
           "market ambience voices", "railway station announcement", "city soundscape traffic", "beach waves ambience",
           "dawn chorus forest", "snow footsteps", "wind ambience", "crowd festival", "tram passing", "church bells town",
           "frogs night", "harbour boats", "mosque call to prayer", "temple bells", "village ambience"]
FREE = re.compile(r"^(cc0|public domain|pd|cc[ -]by)", re.I)


def get(params):
    for _ in range(5):
        r = S.get(API, params={**params, "format": "json"}, timeout=60)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 30)))
            continue
        r.raise_for_status()
        return r.json()
    return {}


def commons_candidates() -> list[dict]:
    from echomap.geo.topo import country_at
    out = []
    for q in QUERIES:
        hits = get({"action": "query", "list": "search", "srsearch": f"{q} filetype:audio", "srnamespace": 6,
                    "srlimit": 40}).get("query", {}).get("search", [])
        titles = [h["title"] for h in hits]
        for i in range(0, len(titles), 40):
            pages = get({"action": "query", "titles": "|".join(titles[i:i + 40]), "prop": "coordinates|imageinfo",
                         "iiprop": "url|size|extmetadata"}).get("query", {}).get("pages", {})
            for pg in pages.values():
                co = (pg.get("coordinates") or [None])[0]
                ii = (pg.get("imageinfo") or [{}])[0]
                em = ii.get("extmetadata") or {}
                lic = (em.get("LicenseShortName") or {}).get("value", "")
                dur = float(ii.get("duration") or 0)
                if not co or not FREE.match(lic) or not (20 <= dur <= 1800) or ii.get("size", 0) > 80e6:
                    continue
                out.append({"source": "Wikimedia Commons", "query": q, "title": pg["title"], "url": ii.get("url"),
                            "page": ii.get("descriptionurl"), "licence": lic, "lat": co["lat"], "lon": co["lon"],
                            "duration_s": dur, "country": country_at(co["lat"], co["lon"])})
        print(f"query '{q}': {len(out)} candidates so far", flush=True)
        time.sleep(1)
    return out


def inat_candidates(n: int) -> list[dict]:
    from echomap.geo.topo import country_at
    out = []
    for taxa, region in [("Amphibia", (-25, -60, 10, -35)), ("Insecta", (35, -10, 55, 30)), ("Mammalia", (-40, 110, -10, 155)),
                         ("Amphibia", (0, 95, 25, 125)), ("Insecta", (25, -110, 45, -75)), ("Mammalia", (-30, 15, 5, 40))]:
        r = S.get("https://api.inaturalist.org/v1/observations", params={
            "sounds": "true", "quality_grade": "research", "iconic_taxa": taxa, "sound_license": "cc0,cc-by,cc-by-sa",
            "swlat": region[0], "swlng": region[1], "nelat": region[2], "nelng": region[3], "per_page": 5,
            "order_by": "random", "geoprivacy": "open", "acc_below": 1000}, timeout=60).json()
        for o in r.get("results", []):
            snd = (o.get("sounds") or [{}])[0]
            if snd.get("file_url") and o.get("geojson"):
                lon, lat = o["geojson"]["coordinates"]
                out.append({"source": "iNaturalist", "query": taxa, "title": (o.get("taxon") or {}).get("name"),
                            "url": snd["file_url"], "page": o.get("uri"), "licence": snd.get("license_code"),
                            "lat": lat, "lon": lon, "date": o.get("observed_on"), "country": country_at(lat, lon)})
                break
        time.sleep(1.5)
    return out[:n]


def near_place(name: str, lat: float, lon: float) -> dict | None:
    """One free-licence recording made within ~10 km of a named place: Commons geosearch first, then iNaturalist."""
    from echomap.geo.topo import country_at
    pages = get({"action": "query", "generator": "geosearch", "ggscoord": f"{lat}|{lon}", "ggsradius": 10000,
                 "ggsnamespace": 6, "ggslimit": 500, "prop": "coordinates|imageinfo",
                 "iiprop": "url|size|mediatype|extmetadata"}).get("query", {}).get("pages", {})
    best = None
    for pg in pages.values():
        ii = (pg.get("imageinfo") or [{}])[0]
        co = (pg.get("coordinates") or [None])[0]
        lic = ((ii.get("extmetadata") or {}).get("LicenseShortName") or {}).get("value", "")
        dur = float(ii.get("duration") or 0)
        if ii.get("mediatype") not in ("AUDIO", "VIDEO") or not co or not FREE.match(lic) or ii.get("size", 0) > 80e6:
            continue
        if dur and not 15 <= dur <= 1800:
            continue
        c = {"source": "Wikimedia Commons", "query": name, "title": pg["title"], "url": ii.get("url"),
             "page": ii.get("descriptionurl"), "licence": lic, "lat": co["lat"], "lon": co["lon"], "duration_s": dur,
             "country": country_at(co["lat"], co["lon"])}
        if best is None or (ii.get("mediatype") == "AUDIO") > (best.get("_audio", False)):
            best = {**c, "_audio": ii.get("mediatype") == "AUDIO"}
    print(f"{name}: {len(pages)} geotagged Commons files nearby, chosen {best and best['title']}", flush=True)
    if best:
        best.pop("_audio", None)
        return best
    r = S.get("https://api.inaturalist.org/v1/observations", params={
        "sounds": "true", "lat": lat, "lng": lon, "radius": 15, "sound_license": "cc0,cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa",
        "per_page": 30, "geoprivacy": "open", "acc_below": 2000, "order_by": "observed_on"}, timeout=60).json()
    for o in r.get("results", []):
        snd = (o.get("sounds") or [{}])[0]
        if snd.get("file_url") and o.get("geojson"):
            lo, la = o["geojson"]["coordinates"]
            print(f"{name}: iNaturalist recording {o.get('uri')}", flush=True)
            return {"source": "iNaturalist", "query": name, "title": f"{name}: {(o.get('taxon') or {}).get('name')}",
                    "url": snd["file_url"], "page": o.get("uri"), "licence": snd.get("license_code"), "lat": la,
                    "lon": lo, "date": o.get("observed_on"), "country": country_at(la, lo)}
    return None


def choose(cands: list[dict], n: int = 10) -> list[dict]:
    chosen, seen_c, seen_q = [], set(), set()
    for c in cands:                       # one per query and per country first: as varied as possible
        if c["query"] not in seen_q and c["country"] not in seen_c:
            chosen.append(c)
            seen_q.add(c["query"])
            seen_c.add(c["country"])
        if len(chosen) >= n:
            break
    for c in cands:
        if len(chosen) >= n:
            break
        if c not in chosen and c["country"] not in seen_c:
            chosen.append(c)
            seen_c.add(c["country"])
    return chosen


def fetch_wav(item: dict, work: Path) -> Path | None:
    from echomap.ingest.media import decode_mono
    raw = work / ("raw_" + re.sub(r"[^A-Za-z0-9._-]", "_", item["url"].split("/")[-1].split("?")[0]))
    if not raw.exists():
        r = S.get(item["url"], timeout=300)
        if r.status_code != 200:
            return None
        raw.write_bytes(r.content)
    out = raw.with_suffix(".bench.wav")
    try:
        x, sr, _ = decode_mono(raw, 32000, max_seconds=290)
    except Exception as e:  # noqa: BLE001
        print("decode failed", item["title"], e)
        return None
    x = x[: 290 * 32000]
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(32000)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return out


def main():
    models, work = sys.argv[1], Path(sys.argv[2])
    work.mkdir(parents=True, exist_ok=True)
    if SETF.exists():
        items = json.loads(SETF.read_text(encoding="utf-8"))["items"]
    elif os.environ.get("BENCH_PLACES"):    # "Derry:54.997,-7.309;Dublin:53.349,-6.260"
        items = []
        for spec in os.environ["BENCH_PLACES"].split(";"):
            nm, ll = spec.split(":")
            la, lo = map(float, ll.split(","))
            it = near_place(nm.strip(), la, lo)
            if it:
                items.append(it)
            time.sleep(1)
        SETF.write_text(json.dumps({"created": time.strftime("%Y-%m-%d"), "items": items}, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    else:
        cands = commons_candidates()
        items = choose(cands)
        if len(items) < 10:
            items += inat_candidates(10 - len(items))
        SETF.parent.mkdir(parents=True, exist_ok=True)
        SETF.write_text(json.dumps({"created": time.strftime("%Y-%m-%d"), "items": items}, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    os.environ["ECHOMAP_EXCLUDE_REFS"] = "\n".join(filter(None, [i.get("page") for i in items] + [i.get("url") for i in items]
                                                          + [i.get("title", "").replace("File:", "") for i in items]))
    from echomap import config
    from echomap.geo.grid import haversine_km
    from echomap.pipeline import Case
    from tools.showcase import shoot
    rows = []
    for k, it in enumerate(items, 1):
        name = f"{k:02d}"
        f = fetch_wav(it, work)
        if not f:
            rows.append({**it, "error": "download/decode failed"})
            continue
        s = config.Settings.load()
        s.analysis_mode = "deep"
        s.models_dir = models
        s.cases_dir = str(work / "cases")
        c = Case(str(f), s)
        t0 = time.time()
        c.prepare()
        c.analyse()
        dt = time.time() - t0
        r = c.result
        g = r.get("geo") or {}
        cands = g.get("candidates") or g.get("weak_candidates") or []
        places = (g.get("likely_places") or {}).get("places") or []
        row = {**it, "state": c.state, "error": c.error, "analysis_s": round(dt, 1), "geo_status": g.get("status"),
               "headline": (r.get("findings") or {}).get("where", {}).get("headline")}
        if cands:
            d = [float(haversine_km(it["lat"], it["lon"], q["centre"]["lat"], q["centre"]["lon"])) for q in cands[:5]]
            row.update({"best": cands[0]["name"], "best_error_km": round(d[0]), "best_radius_km": cands[0]["uncertainty_radius_km"],
                        "inside_best": d[0] <= cands[0]["uncertainty_radius_km"], "min_error_top5_km": round(min(d))})
        if places:
            dp = float(haversine_km(it["lat"], it["lon"], places[0]["lat"], places[0]["lon"]))
            row.update({"top_place": places[0]["name"], "top_place_error_km": round(dp)})
        f_ = r.get("findings") or {}
        row["languages"] = [f"{l['name']} {round(100 * l['confidence'])}%" for l in f_.get("languages", [])]
        row["sounds"] = [x["text"] for x in f_.get("sounds", [])[:8]]
        row["clues"] = (f_.get("where") or {}).get("clues", [])
        try:
            row["screenshots"] = [str(Path(p).relative_to(ROOT)) for p in shoot(c, SHOTS, name)]
        except Exception as e:  # noqa: BLE001
            row["screenshot_error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("title", "country", "geo_status", "best", "best_error_km", "top_place",
                                                  "top_place_error_km", "headline", "languages", "clues")}, ensure_ascii=False), flush=True)
    est = [r for r in rows if "best_error_km" in r and r.get("geo_status") == "estimate"]
    errs = sorted(r["best_error_km"] for r in est)
    summary = {"recordings": len(rows), "estimates": len(est),
               "median_error_km": errs[len(errs) // 2] if errs else None,
               "within_100km": sum(e <= 100 for e in errs), "within_500km": sum(e <= 500 for e in errs),
               "inside_circle": sum(bool(r.get("inside_best")) for r in est)}
    (ROOT / "docs" / "eval" / f"soundscapes{SUF}.json").write_text(
        json.dumps({"generated": time.strftime("%Y-%m-%d %H:%M"), "summary": summary, "rows": rows}, indent=1,
                   ensure_ascii=False), encoding="utf-8")
    md = [f"# Soundscape benchmark{(' (' + TAG + ')') if TAG else ''}: real recordings with known locations, Deep mode", "",
          f"Summary: {json.dumps(summary)}", "",
          "| # | Recording | True place | Estimate | Error | Clues used |", "|---|---|---|---|---|---|"]
    for k, r in enumerate(rows, 1):
        md.append(f"| {k} | {str(r.get('title'))[:50]} | {r.get('country')} ({r['lat']:.2f}, {r['lon']:.2f}) | "
                  f"{r.get('best') or r.get('geo_status')} | {r.get('best_error_km', '-')} km | {'; '.join(r.get('clues') or [])[:90]} |")
    (ROOT / "docs" / "eval" / f"SOUNDSCAPES{SUF}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
