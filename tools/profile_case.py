"""Profile one headless analysis (CI only). Usage: python tools/profile_case.py <file> <mode> [out_json]

Prints per-step timings, the slowest functions (cProfile) and what the user would see: transcripts,
languages, described detections and the geographic summary.
"""
import cProfile
import faulthandler
import json
import os
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echomap import config  # noqa: E402
from echomap.pipeline import Case  # noqa: E402

if os.environ.get("ECHOMAP_STACKDUMP"):
    faulthandler.dump_traceback_later(int(os.environ["ECHOMAP_STACKDUMP"]), repeat=True)

f, mode = sys.argv[1], sys.argv[2]
s = config.Settings.load()
s.analysis_mode = mode
c = Case(f, s)
pr = cProfile.Profile()
t0 = time.time()
pr.enable()
c.prepare()
c.analyse()
ta = time.time() - t0
if c.state != "error" and c.state != "complete":
    c.contextualise({"date": {"mode": "unknown"}, "location": {"mode": "unknown"}})
pr.disable()
tt = time.time() - t0
for st in c.history.steps:
    if st.get("duration_s") is not None and st["duration_s"] >= 0.2:
        print(f"  [{st['duration_s']:7.2f}s] {st['step']}: {st['detail'][:90]}")
print(f"STATE {c.state} error={c.error}")
dur = (c.result.get("analysis_copy") or {}).get("duration_s", 0)
print(f"TIMING mode={mode} audio={dur:.1f}s analysis={ta:.1f}s total={tt:.1f}s ratio={tt / max(dur, 1):.2f}x")
pstats.Stats(pr).sort_stats("cumulative").print_stats(25)
pstats.Stats(pr).sort_stats("tottime").print_stats(15)
r = c.result
tr = (r.get("transcripts") or {}).get("segments") or []
print("TRANSCRIPTS:", len(tr))
for t in tr[:12]:
    print(f"  {t['start']:6.1f}-{t['end']:6.1f} [{t['language']}/{t['quality']}] {t['text'][:120]}")
langs = [(x["label"], x.get("language_code"), x["confidence"]) for x in r.get("inventory", []) if x.get("language_code")]
print("LANGUAGES:", langs[:8])
for x in r.get("inventory", [])[:25]:
    print(f"  {x['start']:6.1f}-{x['end']:6.1f} {x['confidence']:.2f} [{x['category']}] {x['label']} | "
          f"{x.get('specific') or ''} | veh={x.get('evidence', {}).get('vehicle_type')}")
g = r.get("geo") or {}
print("GEO:", g.get("status"), "|", g.get("summary"))
for cnd in (g.get("candidates") or [])[:5]:
    print("  cand", cnd.get("rank"), cnd.get("name"), cnd.get("uncertainty_radius_km"), cnd.get("relative_mass"))
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps({"mode": mode, "file": Path(f).name, "audio_s": dur, "analysis_s": ta,
                                             "total_s": tt, "state": c.state, "error": c.error,
                                             "transcripts": len(tr), "languages": langs[:8],
                                             "geo_status": g.get("status"), "geo_summary": g.get("summary")}))
