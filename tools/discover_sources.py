"""Cloud-side discovery of approved data sources (prints formats; stores nothing). CI only."""
import csv
import io
import json
import time

import requests

S = requests.Session()
S.headers["User-Agent"] = "EchoMap-research/1.0 (private research tool; low volume)"


def show(title, obj, n=1800):
    print(f"\n===== {title} =====")
    print(obj if isinstance(obj, str) else json.dumps(obj, indent=1, ensure_ascii=False)[:n])


# 1. iNaturalist observations with sounds
r = S.get("https://api.inaturalist.org/v1/observations", params={
    "sounds": "true", "quality_grade": "research", "per_page": 2, "sound_license": "cc0,cc-by,cc-by-sa",
    "iconic_taxa": "Aves", "order_by": "votes"}, timeout=60).json()
show("iNat total_results", r.get("total_results"))
if r.get("results"):
    o = r["results"][0]
    show("iNat obs keys", sorted(o.keys()))
    show("iNat obs location fields", {k: o.get(k) for k in ("location", "geojson", "positional_accuracy", "obscured",
                                                             "geoprivacy", "place_guess", "observed_on", "time_observed_at")})
    show("iNat sounds", o.get("sounds"))
    show("iNat taxon", {k: o["taxon"].get(k) for k in ("name", "preferred_common_name", "rank", "iconic_taxon_name")})
time.sleep(1)

# 2. Wikimedia Commons: audio search with coordinates
API = "https://commons.wikimedia.org/w/api.php"
for q in ["filetype:audio soundscape", "filetype:audio field recording", "filetype:audio pedestrian crossing",
          "filetype:audio crossing signal", "filetype:audio traffic light sound", "filetype:audio electric vehicle",
          "filetype:audio change ringing", "filetype:audio church bells ringing", "filetype:audio tram",
          "filetype:audio station announcement"]:
    try:
        j = S.get(API, params={"action": "query", "format": "json", "generator": "search", "gsrsearch": q,
                               "gsrnamespace": 6, "gsrlimit": 25, "prop": "coordinates|imageinfo",
                               "iiprop": "url|mime|size|extmetadata", "iiextmetadatafilter": "LicenseShortName|Artist|GPSLatitude"},
                  timeout=60).json()
        pages = list((j.get("query") or {}).get("pages", {}).values())
        rows = []
        for p in pages:
            ii = (p.get("imageinfo") or [{}])[0]
            em = ii.get("extmetadata", {})
            rows.append([p["title"][5:80], ii.get("mime"), round(ii.get("size", 0) / 1e6, 2),
                         (p.get("coordinates") or [{}])[0].get("lat"), (p.get("coordinates") or [{}])[0].get("lon"),
                         em.get("LicenseShortName", {}).get("value")])
        show(f"Commons '{q}' ({len(pages)} hits; with coords {sum(1 for r_ in rows if r_[3] is not None)})",
             "\n".join(" | ".join(str(c) for c in r_) for r_ in rows), 6000)
    except Exception as e:  # noqa: BLE001
        show(f"Commons '{q}' error", str(e))
    time.sleep(1)

# 3. FLEURS layout
for path in ["data/nl_nl", "data/nl_nl/audio"]:
    j = S.get(f"https://huggingface.co/api/datasets/google/fleurs/tree/main/{path}", timeout=60).json()
    show(f"FLEURS {path}", [(x.get("path"), x.get("size")) for x in j] if isinstance(j, list) else j)
t = S.get("https://huggingface.co/datasets/google/fleurs/resolve/main/data/nl_nl/test.tsv", timeout=60)
show("FLEURS nl_nl test.tsv status/first lines", f"{t.status_code}\n" + "\n".join(t.text.splitlines()[:3])[:1500])
j = S.get("https://huggingface.co/api/datasets/google/fleurs/tree/main/data", timeout=60).json()
show("FLEURS language dirs", [x["path"].split("/")[-1] for x in j] if isinstance(j, list) else j, 4000)

# 4. Dove's Guide CSVs
for name in ["dove.csv", "towers.csv", "bells.csv"]:
    for base in ["https://dove.cccbr.org.uk/downloads/", "https://dove.cccbr.org.uk/"]:
        try:
            r = S.get(base + name, timeout=60)
            if r.status_code == 200 and "," in r.text[:500]:
                rd = list(csv.reader(io.StringIO(r.text)))
                show(f"Dove {name} from {base} ({len(rd)} rows, {len(r.content)/1e6:.2f} MB)",
                     "\n".join(" | ".join(x[:40]) for x in rd[:4]), 6000)
                break
            show(f"Dove {name} {base}", f"status {r.status_code}, ctype {r.headers.get('content-type')}, head {r.text[:200]!r}")
        except Exception as e:  # noqa: BLE001
            show(f"Dove {name} {base} error", str(e))
r = S.get("https://dove.cccbr.org.uk/downloads", timeout=60)
import re
show("Dove downloads page links", sorted(set(re.findall(r'href="([^"]+\.(?:csv|txt|zip|json|xml))"', r.text))))
