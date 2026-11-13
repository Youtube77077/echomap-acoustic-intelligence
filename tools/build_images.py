"""Build EchoMap's offline picture pack (CI only): one small photo per identifiable thing.

Usage: python tools/build_images.py <models_dir> <out.zip> [inat_categories.json]

Keys:  as:<AudioSet class name>   sounds (vehicles, machines, instruments, animals, objects), via the class's
                                   Freebase id (Wikidata P646) -> Wikidata image (P18)
       sp:<scientific name>       every BirdNET+ and iNatSounds species, via Wikidata taxon name (P225) -> P18
       vt:<vehicle type>          car / bus / truck / tram / motorcycle / bicycle (vehicle-type head)
Images come from Wikimedia Commons (free licences only); each entry keeps title, licence, author and source URL so
the app and the report can credit it. Thumbnails are WebP, longest side 128 px. Language, speech and mains hum get
no picture (by design). Nothing runs on the user's PC.
"""
from __future__ import annotations

import csv
import html
import io
import json
import os
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote, unquote

import requests
from PIL import Image

S = requests.Session()
S.headers["User-Agent"] = (f"EchoMap-image-pack-builder/1.0 (one-off open-licence thumbnail build; rate-limited; "
                           f"contact: {os.environ.get('WIKIMEDIA_CONTACT', 'not set')})")
SPARQL = "https://query.wikidata.org/sparql"
COMMONS = "https://commons.wikimedia.org/w/api.php"
SKIP_AS = re.compile(r"speech|speaking|conversation|narration|babbling|whisper|hum\b|mains|silence|noise|static|"
                     r"inside,|outside,|reverberation|echo|sound effect|sine wave|white noise|pink noise", re.I)
VEHICLE_TYPES = {"car": "Q1420", "bus": "Q5638", "truck": "Q43193", "tram": "Q5639", "motorcycle": "Q34493",
                 "bicycle": "Q11442"}
FREE = re.compile(r"^(cc0|public domain|pd|cc[ -]by(-sa)?[ -]?[0-9.]*|cc[ -]by[ -]sa[ -]?[0-9.]*|attribution|gfdl|fal|"
                  r"no restrictions)", re.I)
_lock = threading.Lock()


def sparql(q: str) -> list[dict]:
    for attempt in range(5):
        r = S.post(SPARQL, data={"query": q, "format": "json"}, timeout=120)
        if r.status_code in (429, 503):
            time.sleep(int(r.headers.get("Retry-After", 20)))
            continue
        r.raise_for_status()
        return r.json()["results"]["bindings"]
    return []


def file_of(url: str) -> str:
    return unquote(url.rsplit("/", 1)[-1]).replace("_", " ")


def lookup(values: list[str], prop: str, batch: int = 250) -> dict[str, str]:
    """value -> Commons file name (first image)."""
    out: dict[str, str] = {}
    for i in range(0, len(values), batch):
        vs = " ".join(json.dumps(v) for v in values[i:i + batch])
        rows = sparql(f"SELECT ?v ?img WHERE {{ VALUES ?v {{ {vs} }} ?item wdt:{prop} ?v . ?item wdt:P18 ?img . }}")
        for b in rows:
            out.setdefault(b["v"]["value"], file_of(b["img"]["value"]))
        print(f"  {prop}: {min(i + batch, len(values))}/{len(values)} queried, {len(out)} with image", flush=True)
        time.sleep(1.0)
    return out


def items_image(qids: dict[str, str]) -> dict[str, str]:
    vs = " ".join(f"wd:{q}" for q in qids.values())
    rows = sparql(f"SELECT ?item ?img WHERE {{ VALUES ?item {{ {vs} }} ?item wdt:P18 ?img . }}")
    by_q = {b["item"]["value"].rsplit("/", 1)[-1]: file_of(b["img"]["value"]) for b in rows}
    return {k: by_q[q] for k, q in qids.items() if q in by_q}


def licences(files: list[str]) -> dict[str, dict]:
    meta = {}
    for i in range(0, len(files), 50):
        titles = "|".join("File:" + f for f in files[i:i + 50])
        for attempt in range(5):
            r = S.get(COMMONS, params={"action": "query", "prop": "imageinfo", "iiprop": "extmetadata|url",
                                       "iiurlwidth": 250, "titles": titles, "format": "json"}, timeout=60)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 30)))
                continue
            r.raise_for_status()
            break
        for p in r.json().get("query", {}).get("pages", {}).values():
            ii = (p.get("imageinfo") or [{}])[0]
            em = ii.get("extmetadata") or {}
            lic = (em.get("LicenseShortName") or {}).get("value", "")
            artist = re.sub(r"<[^>]+>", "", html.unescape((em.get("Artist") or {}).get("value", ""))).strip()
            meta[p.get("title", "")[5:]] = {"licence": lic, "artist": artist[:120], "thumb": ii.get("thumburl"),
                                            "source": ii.get("descriptionurl")}
        if i % 1000 == 0:
            print(f"  licences {i}/{len(files)}", flush=True)
        time.sleep(0.5)
    return meta


def fetch_thumb(url: str) -> bytes | None:
    for attempt in range(5):
        try:
            r = S.get(url, timeout=60)
        except requests.RequestException:
            time.sleep(5)
            continue
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 30)))
            continue
        if r.status_code != 200:
            return None
        try:
            im = Image.open(io.BytesIO(r.content))
            im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
            im.thumbnail((128, 128))
            b = io.BytesIO()
            im.save(b, "WEBP", quality=72, method=6)
            return b.getvalue()
        except Exception:  # noqa: BLE001
            return None
    return None


def main():
    md, out = Path(sys.argv[1]), Path(sys.argv[2])
    keys: dict[str, str] = {}      # key -> commons file
    # AudioSet classes
    with open(md / "audioset_labels.csv", encoding="utf-8") as fh:
        asl = [r for r in csv.DictReader(fh) if not SKIP_AS.search(r["display_name"])]
    mids = {r["mid"]: r["display_name"] for r in asl}
    for mid, f in lookup(list(mids), "P646", batch=150).items():
        keys["as:" + mids[mid]] = f
    # vehicle types
    for k, f in items_image(VEHICLE_TYPES).items():
        keys["vt:" + k] = f
    # species
    names = set()
    with open(md / "birdnet_v3_labels.csv", encoding="utf-8-sig") as fh:
        names |= {r["sci_name"].strip() for r in csv.DictReader(fh, delimiter=";") if r.get("sci_name")}
    if len(sys.argv) > 3 and Path(sys.argv[3]).exists():
        cats = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))["categories"]
        names |= {c["name"].strip() for c in cats if c.get("name")}
    names = sorted(n for n in names if re.match(r"^[A-Z][a-z]+ [a-z\-]+( [a-z\-]+)?$", n))
    print(f"AudioSet classes with image: {sum(k.startswith('as:') for k in keys)}; species to query: {len(names)}", flush=True)
    for n, f in lookup(names, "P225").items():
        keys["sp:" + n.lower()] = f
    files = sorted(set(keys.values()))
    print(f"{len(keys)} keys -> {len(files)} distinct files; fetching licences", flush=True)
    meta = licences(files)
    files = [f for f in files if f in meta and meta[f]["thumb"] and FREE.match(meta[f]["licence"] or "")]
    print(f"{len(files)} files with a free licence; downloading thumbnails", flush=True)
    blobs: dict[str, bytes] = {}

    def job(f):
        b = fetch_thumb(meta[f]["thumb"])
        if b:
            with _lock:
                blobs[f] = b
                if len(blobs) % 500 == 0:
                    print(f"  thumbnails {len(blobs)}/{len(files)}", flush=True)
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(job, files))
    fid = {f: i for i, f in enumerate(sorted(blobs))}
    index = {}
    for k, f in keys.items():
        if f in fid:
            m = meta[f]
            index[k] = {"img": f"img/{fid[f]}.webp", "title": f, "licence": m["licence"], "artist": m["artist"],
                        "source": m["source"] or f"https://commons.wikimedia.org/wiki/File:{quote(f.replace(' ', '_'))}"}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for f, i in fid.items():
            z.writestr(f"img/{i}.webp", blobs[f])
        z.writestr("index.json", json.dumps({"version": 1, "source": "Wikimedia Commons via Wikidata (P18)",
                                             "entries": index}, ensure_ascii=False))
    counts = {p: sum(k.startswith(p) for k in index) for p in ("as:", "sp:", "vt:")}
    print(f"DONE {out} {out.stat().st_size / 1e6:.1f} MB, entries {counts}", flush=True)


if __name__ == "__main__":
    main()
