"""Build EchoMap's city transit sound library from openly licensed Wikimedia Commons recordings (CI only).

Usage: python tools/build_transit_refs.py <models_dir> <work_dir>
Output: echomap/data/transit_refs.json.gz (embeddings + fingerprints + city + kind + licence) and
        <work_dir>/clips/* (raw files, published as release asset 'transit-refs-v1' for provenance), coverage.json.

Method
  1. Walk Commons audio categories about trams, metros, trains, buses and public-transport announcements
     (sub-categories to depth 5).
  2. Keep files whose licence is CC0 / public domain / CC BY / CC BY-SA.
  3. Work out the city from the file's own coordinates, else from city names in its categories or title
     (GeoNames ≥ 15,000 inhabitants). Each entry records how its location was found.
  4. Classify the kind (tram, metro, train, bus, announcement) from category and title keywords.
  5. Download politely (rate-limited, honouring HTTP 429), then compute the same embedding and landmark
     fingerprint as the main reference library. At most a few clips per city and kind, so coverage is spread
     over as many cities as possible within the time budget.
"""
from __future__ import annotations

import base64
import gzip
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echomap.detect.references import fingerprint  # noqa: E402
from echomap.geo.gazetteer import _cities  # noqa: E402
from echomap.geo.places import _name_index, _norm  # noqa: E402
from echomap.geo.topo import country_at  # noqa: E402
from echomap.ingest.media import decode_mono, resample  # noqa: E402
from echomap.models import sed  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"
S = requests.Session()
S.headers["User-Agent"] = "EchoMap-transit-library-builder/1.0 (private research tool; low volume; rate-limited)"
ROOTS = ["Category:Sounds of trams", "Category:Audio files about public transport", "Category:Audio files about rail transport",
         "Category:Voice messages in public transport", "Category:Sounds of trains", "Category:Sounds of rail transport",
         "Category:Sounds of buses", "Category:Sounds of metro", "Category:Sounds of rapid transit",
         "Category:Audio files of train announcements", "Category:Sounds of public transport", "Category:Station announcements"]
OK_LIC = re.compile(r"^(CC0|Public domain|PD|CC BY(-SA)? [0-9.]+.*)$", re.I)
AUDIO = (".ogg", ".oga", ".opus", ".wav", ".mp3", ".flac", ".webm")
KINDS = [("announcement", r"announc|ansage|hlášen|hlasen|anuncio|annonce|annuncio|mededeling|komunikat|voice message|gong|jingle|chime|melod"),
         ("tram", r"\btram|streetcar|straßenbahn|strassenbahn|tramvaj|tranvía|tranvia|tramway|villamos|tramwaj|electrico|eléctrico|trolley"),
         ("metro", r"metro|subway|underground|u-bahn|\bmrt\b|rapid transit|tube|métro|s-bahn"),
         ("bus", r"\bbus|trolleybus|autobus|omnibus"),
         ("train", r"train|railway|rail transport|locomotive|eisenbahn|\bzug\b|vlak|tren|treno|trein|pociąg|level crossing")]
CAP_PER_CITY_KIND = int(os.environ.get("TRANSIT_CAP_PER_CITY_KIND", "3"))


def api(params: dict) -> dict:
    for a in range(6):
        try:
            r = S.get(API, params={**params, "format": "json", "maxlag": 5}, timeout=60)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
                j = r.json()
                if "error" not in j or j["error"].get("code") != "maxlag":
                    time.sleep(0.3)
                    return j
            time.sleep(float(r.headers.get("Retry-After", 0) or 5 * (a + 1)))
        except Exception as e:  # noqa: BLE001
            print("api retry", e)
            time.sleep(5 * (a + 1))
    return {}


def members(cat: str, typ: str) -> list[str]:
    out, cont = [], {}
    while True:
        j = api({"action": "query", "list": "categorymembers", "cmtitle": cat, "cmtype": typ, "cmlimit": 500, **cont})
        out += [m["title"] for m in (j.get("query") or {}).get("categorymembers", [])]
        if "continue" not in j:
            return out
        cont = j["continue"]


def crawl() -> dict[str, set]:
    """file title -> set of category names it was found under (plus its own categories later)."""
    files: dict[str, set] = {}
    depth = {r: 0 for r in ROOTS}
    queue, seen = list(ROOTS), set()
    while queue:
        c = queue.pop(0)
        if c in seen or depth[c] > 5:
            continue
        seen.add(c)
        for f in members(c, "file"):
            if f.lower().endswith(AUDIO):
                files.setdefault(f, set()).add(c)
        for sc in members(c, "subcat"):
            if sc not in depth:
                depth[sc] = depth[c] + 1
                queue.append(sc)
    print(f"crawled {len(seen)} categories, {len(files)} audio files")
    return files


def details(titles: list[str]) -> dict[str, dict]:
    out = {}
    for i in range(0, len(titles), 40):
        chunk = titles[i:i + 40]
        j = api({"action": "query", "titles": "|".join(chunk), "prop": "imageinfo|coordinates|categories", "cllimit": 500,
                 "iiprop": "url|mime|size|extmetadata", "iiextmetadatafilter": "LicenseShortName|Artist|DateTimeOriginal"})
        for p in (j.get("query") or {}).get("pages", {}).values():
            out[p["title"]] = p
    return out


def locate(page: dict, cats: set) -> dict | None:
    rows, _, _ = _cities()
    co = (page.get("coordinates") or [None])[0]
    text = " | ".join(sorted(cats | {c["title"] for c in page.get("categories", [])})) + " | " + page["title"]
    idx = _name_index()
    hits = []
    for m in re.finditer(r"[A-ZÀ-ÖØ-ÞĀ-Ž][\w'’\-]+(?:[ \-][A-ZÀ-ÖØ-ÞĀ-Ž][\w'’\-]+){0,2}", text):
        words = m.group(0).replace("-", " ").split()
        for size in range(len(words), 0, -1):
            for k in range(len(words) - size + 1):
                n = _norm(" ".join(words[k:k + size]))
                if n in idx and len(n) >= 4:
                    hits += idx[n]
    if co:
        lat, lon = co["lat"], co["lon"]
        near = [r for r in hits if abs(r[4] - lat) < 0.6 and abs(r[5] - lon) < 0.9]
        r = max(near, key=lambda r: r[6]) if near else None
        return {"lat": lat, "lon": lon, "city": r[0] if r else None, "cc": country_at(lat, lon) or (r[2] if r else None),
                "location_basis": "Commons file coordinates"}
    if hits:
        cnt = Counter(r[0] for r in hits)
        best = max(hits, key=lambda r: (cnt[r[0]], r[6]))
        if best[6] >= 50_000 or cnt[best[0]] >= 2:
            return {"lat": best[4], "lon": best[5], "city": best[0], "cc": best[2],
                    "location_basis": "city named in the file's Commons categories/title (city centre)"}
    return None


def kind_of(page: dict, cats: set) -> str | None:
    text = (" ".join(cats | {c["title"] for c in page.get("categories", [])}) + " " + page["title"]).lower()
    for k, rx in KINDS:
        if re.search(rx, text):
            return k
    return None


def fetch(url: str) -> bytes | None:
    for a in range(5):
        r = S.get(url, timeout=120)
        if r.status_code == 200 and not r.headers.get("content-type", "").startswith("text/"):
            time.sleep(2.0)
            return r.content
        wait = float(r.headers.get("Retry-After", 0) or 0) or 10 * (a + 1)
        print("download status", r.status_code, "waiting", wait)
        time.sleep(min(wait, 120))
    return None


def main():
    models, work = sys.argv[1], Path(sys.argv[2])
    budget = float(os.environ.get("TRANSIT_BUDGET_MIN", "80")) * 60
    t0 = time.time()
    clips = work / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    files = crawl()
    info = details(sorted(files))
    cands = []
    for title, page in info.items():
        ii = (page.get("imageinfo") or [{}])[0]
        lic = ii.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "").strip()
        if not OK_LIC.match(lic) or not (20_000 <= ii.get("size", 0) <= 25_000_000):
            continue
        cats = files.get(title, set())
        loc, k = locate(page, cats), kind_of(page, cats)
        if loc and k:
            cands.append((title, page, loc, k, lic))
    print(f"{len(cands)} licensed, located, classified candidates "
          f"({len({(c[2]['city'] or round(c[2]['lat'], 1)) for c in cands})} places)")
    # round-robin over (city, kind) so as many cities as possible are covered before the budget ends
    groups: dict[tuple, list] = {}
    for c in cands:
        groups.setdefault((c[2]["city"] or f"{c[2]['lat']:.2f},{c[2]['lon']:.2f}", c[3]), []).append(c)
    order = []
    for rnd in range(CAP_PER_CITY_KIND):
        order += [g[rnd] for g in groups.values() if len(g) > rnd]
    labels = sed.load_labels(models)
    refs = []
    for title, page, loc, k, lic in order:
        if time.time() - t0 > budget:
            print("time budget reached; writing what has been built")
            break
        ii = page["imageinfo"][0]
        dst = clips / re.sub(r"[^A-Za-z0-9._-]+", "_", title[5:])[:120]
        try:
            body = fetch(ii["url"])
            if body is None:
                continue
            dst.write_bytes(body)
            x, sr, _ = decode_mono(dst, 32000)
        except Exception as e:  # noqa: BLE001
            print("skip", title, e)
            dst.unlink(missing_ok=True)
            continue
        dur = len(x) / sr
        if dur < 1.0:
            dst.unlink(missing_ok=True)
            continue
        x = x[: int(sr * 180)]  # first 3 minutes are plenty for a signature
        r = sed.run(x, models)
        emb = r["embeddings"].mean(axis=0)
        emb /= np.linalg.norm(emb) + 1e-9
        top = np.argsort(-r["scores"].max(axis=0))[:3]
        em = ii.get("extmetadata", {})
        refs.append({"id": f"T{len(refs) + 1:04d}", "title": title[5:], "category": k, "kind": k, "query": "commons-category",
                     "city": loc["city"], "country": loc["cc"], "lat": loc["lat"], "lon": loc["lon"],
                     "location_basis": loc["location_basis"], "coordinate_precision": loc["location_basis"],
                     "source_url": f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}", "license": lic,
                     "artist": re.sub(r"<[^>]+>", "", em.get("Artist", {}).get("value", ""))[:120],
                     "date": em.get("DateTimeOriginal", {}).get("value"), "duration_s": round(dur, 2),
                     "audioset_top": [labels[i]["name"] for i in top], "clip_file": dst.name,
                     "embedding": base64.b64encode(emb.astype(np.float16).tobytes()).decode(),
                     "fingerprint": fingerprint(resample(x, 32000, 8000))[:3000]})
        print(f"{k:12s} {str(loc['city'])[:22]:22s} {loc['cc'] or '':3s} {title[5:70]:65s} t={int(time.time() - t0)}s")
    cities = {(r["city"] or "?", r["country"]) for r in refs}
    meta = {"built": time.strftime("%Y-%m-%d"), "n": len(refs), "source": "Wikimedia Commons (per-file licences recorded)",
            "candidates_found": len(cands), "places": len(cities), "countries": len({r["country"] for r in refs}),
            "kinds": dict(Counter(r["kind"] for r in refs)), "method": __doc__.split("Method")[1].strip()}
    with gzip.open(ROOT / "echomap" / "data" / "transit_refs.json.gz", "wt", encoding="utf-8") as g:
        json.dump({"meta": meta, "refs": refs}, g, ensure_ascii=False, separators=(",", ":"))
    cov = {}
    for r in refs:
        cov.setdefault(f"{r['city']} ({r['country']})", Counter())[r["kind"]] += 1
    (work / "coverage.json").write_text(json.dumps({"meta": meta, "coverage": {k: dict(v) for k, v in sorted(cov.items())}},
                                                   ensure_ascii=False, indent=1), encoding="utf-8")
    (work / "manifest.json").write_text(json.dumps([{k: v for k, v in r.items() if k not in ("embedding", "fingerprint")}
                                                    for r in refs], ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
