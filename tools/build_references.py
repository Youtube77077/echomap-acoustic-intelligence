"""Build EchoMap's reference signature library from openly licensed Wikimedia Commons recordings (CI only).

Usage: python tools/build_references.py <models_dir> <work_dir>
Output: echomap/data/references.json.gz (embeddings + fingerprints + metadata) and <work_dir>/clips/*
        (raw files, published as release asset 'references-v1' for provenance).
Only CC0 / public domain / CC BY / CC BY-SA files are used, and every entry keeps title, author, licence and URL.
"""
import base64
import gzip
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echomap.detect.references import fingerprint  # noqa: E402
from echomap.dsp import tonal  # noqa: E402
from echomap.ingest.media import decode_mono, resample  # noqa: E402
from echomap.models import sed  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"
S = requests.Session()
S.headers["User-Agent"] = "EchoMap-reference-builder/1.0 (private research tool; low volume)"
QUERIES = {
    "crossing signal": ["pedestrian crossing", "crosswalk signal", "Walk Sign", "audible pedestrian signal", "Akustikkennung Ampel",
                        "semaforo sonoro", "Toryanse", "aŭda piedira semaforo", "Övergångsställe"],
    "level crossing": ["level crossing", "Bahnübergang Läutewerk", "BÜ Schrankensignal", "railroad crossing bell",
                       "passaggio a livello", "Barrierenglocke"],
    "bells": ["change ringing", "church bells ringing", "Glockengeläut", "Kirchenglocken Geläut", "carillon"],
    "siren": ["siren", "Martinshorn", "police siren", "ambulance siren", "civil defense siren"],
    "announcement": ["station announcement", "train announcement", "metro announcement"],
    "tram": ["tram sound", "tram bell", "streetcar"],
    "train": ["train passing", "train horn", "steam locomotive whistle"],
}
OK_LIC = re.compile(r"^(CC0|Public domain|PD|CC BY(-SA)? [0-9.]+.*)$", re.I)
PRONUNCIATION = re.compile(r"^(LL-|[A-Z][a-z]{1,3}-)")


def fetch_file(url, tries=5):
    """Download one file politely: honour HTTP 429 / Retry-After and back off; reject HTML error pages."""
    for attempt in range(tries):
        r = S.get(url, timeout=120)
        if r.status_code == 200 and not r.headers.get("content-type", "").startswith("text/"):
            time.sleep(2.5)
            return r.content
        wait = float(r.headers.get("Retry-After", 0) or 0) or 10 * (attempt + 1)
        print("download status", r.status_code, "waiting", wait)
        time.sleep(min(wait, 120))
    return None


def search(q, limit=25):
    for attempt in range(4):
        try:
            return _search(q, limit)
        except Exception as e:  # noqa: BLE001 - Commons may throttle or return HTML errors
            print("search retry", q, attempt, e)
            time.sleep(10 * (attempt + 1))
    return []


def _search(q, limit=25):
    j = S.get(API, params={"maxlag": 5, "action": "query", "format": "json", "generator": "search", "gsrsearch": f"filetype:audio {q}",
                           "gsrnamespace": 6, "gsrlimit": limit, "prop": "coordinates|imageinfo",
                           "iiprop": "url|mime|size|extmetadata", "iiextmetadatafilter": "LicenseShortName|Artist|DateTimeOriginal"},
              timeout=60).json()
    return list((j.get("query") or {}).get("pages", {}).values())


def main():
    models, work = sys.argv[1], Path(sys.argv[2])
    clips = work / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    refs, seen = [], set()
    for cat, qs in QUERIES.items():
        n_cat = 0
        for q in qs:
            for p in search(q):
                title = p["title"][5:]
                ii = (p.get("imageinfo") or [{}])[0]
                em = ii.get("extmetadata", {})
                lic = em.get("LicenseShortName", {}).get("value", "")
                if title in seen or PRONUNCIATION.match(title) or not OK_LIC.match(lic.strip()):
                    continue
                if not (30_000 <= ii.get("size", 0) <= 15_000_000) or "audio" not in ii.get("mime", "") and "ogg" not in ii.get("mime", ""):
                    continue
                seen.add(title)
                dst = clips / re.sub(r"[^A-Za-z0-9._-]+", "_", title)[:120]
                try:
                    body = fetch_file(ii["url"])
                    if body is None:
                        print("skip (download refused/throttled)", title)
                        continue
                    dst.write_bytes(body)
                    x, sr, _ = decode_mono(dst, 32000)
                except Exception as e:  # noqa: BLE001
                    print("skip", title, e)
                    dst.unlink(missing_ok=True)
                    continue
                dur = len(x) / sr
                if dur < 0.5 or dur > 300:
                    dst.unlink(missing_ok=True)
                    continue
                r = sed.run(x, models)
                emb = r["embeddings"].mean(axis=0)
                emb /= np.linalg.norm(emb) + 1e-9
                top = np.argsort(-r["scores"].max(axis=0))[:3]
                labels = sed.load_labels(models)
                x8 = resample(x, 32000, 8000)
                tracks = tonal.tone_tracks(x, 32000)
                pats = tonal.periodic_tone_patterns(tracks)
                coords = (p.get("coordinates") or [{}])[0]
                artist = re.sub(r"<[^>]+>", "", em.get("Artist", {}).get("value", ""))[:120]
                refs.append({"id": f"R{len(refs) + 1:04d}", "title": title, "category": cat, "query": q,
                             "description": f"Wikimedia Commons recording found for '{q}'",
                             "source_url": f"https://commons.wikimedia.org/wiki/File:{title.replace(' ', '_')}",
                             "license": lic, "artist": artist, "lat": coords.get("lat"), "lon": coords.get("lon"),
                             "coordinate_precision": "Commons file coordinates (uploader supplied)" if coords else None,
                             "date": em.get("DateTimeOriginal", {}).get("value"), "duration_s": round(dur, 2), "sample_rate_decoded": 32000,
                             "audioset_top": [labels[i]["name"] for i in top],
                             "periodic_patterns": pats[:3], "clip_file": dst.name,
                             "embedding": base64.b64encode(emb.astype(np.float16).tobytes()).decode(),
                             "fingerprint": fingerprint(x8)[:6000]})
                n_cat += 1
                print(f"{cat:16s} {title[:70]:70s} {lic:14s} {dur:6.1f}s coords={bool(coords)}")
                time.sleep(0.3)
                if n_cat >= 30:
                    break
            if n_cat >= 30:
                break
    meta = {"built": time.strftime("%Y-%m-%d"), "n": len(refs), "source": "Wikimedia Commons (per-file licences recorded)",
            "categories": {c: sum(1 for r in refs if r["category"] == c) for c in QUERIES},
            "with_coordinates": sum(1 for r in refs if r["lat"] is not None),
            "embedding": "EfficientAT mn10_as 960-d, mean over 5 s windows, L2-normalised, float16",
            "fingerprint": "spectral-peak pair hashes at 8 kHz (512-pt FFT, hop 128)"}
    out = Path(__file__).resolve().parents[1] / "echomap" / "data" / "references.json.gz"
    with gzip.open(out, "wt", encoding="utf-8") as g:
        json.dump({"meta": meta, "refs": refs}, g, separators=(",", ":"))
    (work / "manifest.json").write_text(json.dumps({"meta": meta, "refs": [{k: v for k, v in r.items() if k not in ("embedding", "fingerprint")}
                                                                          for r in refs]}, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
