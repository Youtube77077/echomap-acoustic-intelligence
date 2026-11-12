"""Headless analysis of one file (development / testing). Usage: python tools/run_case.py <file> [country] [city] [date]"""
import faulthandler, json, os, sys, time
if os.environ.get("ECHOMAP_STACKDUMP"):
    faulthandler.dump_traceback_later(int(os.environ["ECHOMAP_STACKDUMP"]), repeat=True)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echomap import config
from echomap.pipeline import Case
from echomap.geo.gazetteer import search_city

f = sys.argv[1]
s = config.Settings.load()
c = Case(f, s)
t = time.time(); c.prepare(); c.analyse(); ta = time.time() - t
if c.state == "error":
    print("ERROR", c.error); print(c.history.steps[-1].get("params", {}).get("traceback", "")); sys.exit(1)
ctx = {"date": {"mode": "unknown"}, "location": {"mode": "unknown"}}
if len(sys.argv) > 2 and sys.argv[2] != "-":
    ctx["location"] = {"mode": "country", "country": sys.argv[2]}
if len(sys.argv) > 3 and sys.argv[3] != "-":
    city = search_city(sys.argv[3], sys.argv[2] if sys.argv[2] != "-" else None)[0]
    ctx["location"] = {"mode": "city", "country": city["country"], "city": city}
if len(sys.argv) > 4:
    ctx["date"] = {"mode": "exact", "value": sys.argv[4]}
t = time.time(); c.contextualise(ctx); tg = time.time() - t
if c.state == "error":
    print("ERROR", c.error); print(c.history.steps[-1].get("params", {}).get("traceback", "")); sys.exit(1)
r = c.result
for st in c.history.steps:
    print(f"  [{st.get('duration_s', '')}s] {st['step']}: {st['detail']}")
print(f"analysis {ta:.1f}s, geo {tg:.1f}s, records {len(r['inventory'])}, dir {c.dir}")
for x in r["inventory"]:
    print(f"  {x['id']} {x['start']:7.2f}-{x['end']:7.2f} {x['tier']:12s} {x['confidence']:.2f} [{x['category']}] {x['label']} | {x.get('specific') or ''}")
g = r["geo"]
print("GEO:", g.get("status"), g.get("summary"))
for cnd in g.get("candidates", []):
    print("  cand", cnd["rank"], cnd["name"], cnd["centre"], cnd["resolution"], cnd["uncertainty_radius_km"], cnd["strength"], "mass", cnd["relative_mass"])
print("SED top:", [(t["label"], t["max"]) for t in r["sed_raw"]["top"][:10]])
if "birdnet_raw" in r: print("BirdNET top:", [(t["label"], t["max"]) for t in r["birdnet_raw"]["top"][:6]])
print("SCENE:", [(t["scene"], t["score"]) for t in r["scene"]["top"]], r["scene"]["indoor_outdoor"]["estimate"])
print("MAINS:", r["signal"]["mains"].get("detected"), r["signal"]["mains"].get("nominal_hz"))
print("TIME:", r["time"]["estimated"].get("season_from_species", {}).get("interpretation"))
