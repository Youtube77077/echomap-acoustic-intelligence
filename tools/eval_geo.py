"""Geolocation evaluation on recordings with known coordinates (CI only).

Sources: iNaturalist research-grade observations with CC0/CC BY/CC BY-SA sounds (birds, 8 world regions),
plus geotagged Wikimedia Commons recordings from the reference manifest (non-bird soundscapes).
Each recording is analysed twice: date unknown, then with the observation date supplied.
Metrics: estimate rate, great-circle error of the best candidate centre, whether the truth lies inside the
best/any candidate circle (coverage, an uncertainty-calibration check), and whether the truth lies inside the
"most clues agree" region.
Caveat (data leakage): the BirdNET geomodel is trained on citizen-science occurrences that may include these
observations, so biogeographic results can be optimistic.
Usage: python tools/eval_geo.py <models_dir> <work_dir> <out_json> [per_region]
"""
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echomap import config  # noqa: E402
from echomap.geo.grid import haversine_km  # noqa: E402
from echomap.pipeline import Case  # noqa: E402

REGIONS = {"North America": (25, -125, 50, -70), "South America": (-35, -75, 5, -40), "Europe": (36, -10, 60, 30),
           "Africa": (-30, 15, 10, 40), "South Asia": (8, 70, 28, 90), "East Asia": (22, 100, 42, 140),
           "Australia": (-38, 115, -15, 153), "Middle East": (24, 34, 38, 55)}
S = requests.Session()
S.headers["User-Agent"] = "EchoMap-evaluation/1.0 (private research tool; low volume)"


def inat(per_region):
    out = []
    for name, (swlat, swlng, nelat, nelng) in REGIONS.items():
        r = S.get("https://api.inaturalist.org/v1/observations", params={
            "sounds": "true", "quality_grade": "research", "iconic_taxa": "Aves", "sound_license": "cc0,cc-by,cc-by-sa",
            "swlat": swlat, "swlng": swlng, "nelat": nelat, "nelng": nelng, "per_page": per_region * 3, "order_by": "random",
            "geoprivacy": "open", "acc_below": 1000}, timeout=60).json()
        n = 0
        for o in r.get("results", []):
            if o.get("obscured") or not o.get("sounds") or not o.get("location"):
                continue
            snd = next((s for s in o["sounds"] if s.get("file_url") and s.get("license_code")), None)
            if not snd:
                continue
            lat, lon = map(float, o["location"].split(","))
            out.append({"source": "iNaturalist", "region": name, "id": o["id"], "url": snd["file_url"], "license": snd["license_code"],
                        "attribution": snd.get("attribution"), "lat": lat, "lon": lon, "date": o.get("observed_on"),
                        "taxon": (o.get("taxon") or {}).get("name"), "common": (o.get("taxon") or {}).get("preferred_common_name")})
            n += 1
            if n >= per_region:
                break
        time.sleep(1.2)
    return out


def commons_geotagged():
    p = Path(__file__).resolve().parents[1] / "echomap" / "data" / "references.json.gz"
    if not p.exists():
        return []
    import gzip
    refs = json.load(gzip.open(p, "rt", encoding="utf-8"))["refs"]
    return [{"source": "Wikimedia Commons", "region": "commons", "id": r["id"], "title": r["title"], "category": r["category"],
             "url": None, "clip_file": r["clip_file"], "license": r["license"], "lat": r["lat"], "lon": r["lon"], "date": None}
            for r in refs if r.get("lat") is not None]


def evaluate(path, truth, models, cases_dir, date):
    s = config.Settings(models_dir=models, cases_dir=str(cases_dir), analysis_mode="standard")
    c = Case(str(path), s)
    c.prepare()
    if (c.meta.get("duration_s") or 0) > 300:
        return {"skipped": "longer than 5 min"}
    c.analyse()
    if c.state != "analysed":
        return {"error": c.error}
    ctx = {"date": {"mode": "exact", "value": date} if date else {"mode": "unknown"}, "location": {"mode": "unknown"}}
    c.contextualise(ctx)
    if c.state != "complete":
        return {"error": c.error}
    g = c.result["geo"]
    res = {"status": g.get("status"), "n_footprints": len(g.get("footprints", [])),
           "species": [r["common_name"] for r in c.result["inventory"] if r.get("scientific_name") and r["confidence"] >= 0.4][:8]}
    cands = g.get("candidates", [])
    if cands:
        d = [float(haversine_km(truth[0], truth[1], k["centre"]["lat"], k["centre"]["lon"])) for k in cands]
        res.update({"best_error_km": round(d[0], 1), "best_radius_km": cands[0]["uncertainty_radius_km"],
                    "inside_best": d[0] <= cands[0]["uncertainty_radius_km"],
                    "inside_any": any(di <= k["uncertainty_radius_km"] for di, k in zip(d, cands)),
                    "min_error_km": round(min(d), 1)})
    # agreement region check
    fps = [f for f in g.get("footprints", []) if f["used"]]
    if fps:
        row, col = int((90 - truth[0]) / 0.5), int((truth[1] + 180) / 0.5)
        idx = row * 720 + col
        cnt = sum(int(np.unpackbits(np.frombuffer(base64.b64decode(f["bits"]), np.uint8))[idx]) for f in fps)
        allc = np.zeros(720 * 360, np.int32)
        for f in fps:
            allc += np.unpackbits(np.frombuffer(base64.b64decode(f["bits"]), np.uint8))[:720 * 360]
        res.update({"clues_covering_truth": cnt, "clues_total": len(fps), "truth_in_max_agreement": bool(cnt == allc.max())})
    return res


def summarise(rows, key):
    rs = [r[key] for r in rows if key in r and "error" not in r[key] and "skipped" not in r[key]]
    est = [r for r in rs if r["status"] == "estimate" and "best_error_km" in r]
    errs = sorted(r["best_error_km"] for r in est)
    agree = [r for r in rs if "truth_in_max_agreement" in r]
    return {"n_analysed": len(rs), "n_estimates": len(est), "estimate_rate": round(len(est) / max(1, len(rs)), 3),
            "median_error_km": errs[len(errs) // 2] if errs else None,
            "p25_error_km": errs[len(errs) // 4] if errs else None, "p75_error_km": errs[3 * len(errs) // 4] if errs else None,
            "coverage_best": round(sum(r["inside_best"] for r in est) / max(1, len(est)), 3),
            "coverage_any": round(sum(r["inside_any"] for r in est) / max(1, len(est)), 3),
            "truth_in_max_agreement_rate": round(sum(r["truth_in_max_agreement"] for r in agree) / max(1, len(agree)), 3),
            "median_best_radius_km": sorted(r["best_radius_km"] for r in est)[len(est) // 2] if est else None}


def main():
    models, work, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    per = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    work.mkdir(parents=True, exist_ok=True)
    items = inat(per) + commons_geotagged()
    refclips = Path("/home/runner/refs/clips")
    rows = []
    for i, it in enumerate(items):
        if it["url"]:
            f = work / f"inat_{it['id']}{Path(it['url'].split('?')[0]).suffix or '.mp3'}"
            try:
                f.write_bytes(S.get(it["url"], timeout=120).content)
            except Exception as e:  # noqa: BLE001
                rows.append({**it, "download_error": str(e)})
                continue
            time.sleep(1.0)
        else:
            f = refclips / it["clip_file"]
            if not f.exists():
                continue
        row = {**it}
        try:
            row["date_unknown"] = evaluate(f, (it["lat"], it["lon"]), models, work / "cases", None)
            if it.get("date"):
                row["date_known"] = evaluate(f, (it["lat"], it["lon"]), models, work / "cases", it["date"])
        except Exception as e:  # noqa: BLE001
            row["date_unknown"] = {"error": f"{type(e).__name__}: {e}"}
        rows.append(row)
        print(i, it["source"], it["region"], it.get("common") or it.get("title"), json.dumps(row.get("date_unknown"))[:220], flush=True)
    summary = {"iNaturalist_date_unknown": summarise([r for r in rows if r["source"] == "iNaturalist"], "date_unknown"),
               "iNaturalist_date_known": summarise([r for r in rows if r["source"] == "iNaturalist"], "date_known"),
               "commons_date_unknown": summarise([r for r in rows if r["source"] == "Wikimedia Commons"], "date_unknown"),
               "by_region": {reg: summarise([r for r in rows if r["region"] == reg], "date_unknown") for reg in REGIONS}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"generated": time.strftime("%Y-%m-%d"), "method": __doc__.strip(), "summary": summary,
                               "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
