"""Tune the geographic engine on saved evaluation cases (CI only). Usage: python tools/geo_tune.py <models> <geo_cases.json> <out.json>

geo_cases.json comes from tools/eval_geo.py (detections + truth per real field recording; no audio). For every setting
in a small grid, the engine is re-run on all cases and scored. Cases are split by index parity into two halves so a
setting that only wins on one half (over-fitting) is visible. Score: median error of the best candidate over ALL cases
(a case without an estimate counts as 5,000 km), plus the estimate rate and the radius calibration factor that would
give 68% coverage.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echomap.geo import engine as E  # noqa: E402
from echomap.geo.grid import haversine_km  # noqa: E402

GRID = {"GROUP_CLIP": [6.0, 10.0], "GROUP_DISCOUNT_EXP": [0.5, 0.25], "SPECIES_MIN_CONF": [0.4, 0.3],
        "MIN_INFORMATION_RATIO": [2.0, 1.5]}
NO_EST_KM = 5000.0


def run(cases, models, params):
    for k, v in params.items():
        setattr(E, k, v)
    eng = run.eng.setdefault(models, E.GeoEngine(models))
    out = []
    for c in cases:
        ctx = {"date": {"mode": "exact", "value": c["date"]} if c.get("date") else {"mode": "unknown"},
               "location": {"mode": "unknown"}}
        recs = [dict(r) for r in c["records"]]
        try:
            g = eng.run(recs, c["mains"], ctx, {})
        except Exception as e:  # noqa: BLE001
            out.append({"err": str(e)})
            continue
        cand = (g.get("candidates") or [None])[0]
        if cand:
            d = float(haversine_km(c["truth"][0], c["truth"][1], cand["centre"]["lat"], cand["centre"]["lon"]))
            out.append({"est": True, "err_km": d, "radius": cand["uncertainty_radius_km"]})
        else:
            out.append({"est": False, "err_km": NO_EST_KM, "radius": None})
    return out


run.eng = {}


def score(res):
    errs = [r["err_km"] for r in res if "err_km" in r]
    est = [r for r in res if r.get("est")]
    ratio = sorted(r["err_km"] / max(r["radius"], 1) for r in est)
    return {"median_err_all_km": round(float(np.median(errs)), 1) if errs else None,
            "median_err_estimates_km": round(float(np.median([r["err_km"] for r in est])), 1) if est else None,
            "estimate_rate": round(len(est) / max(1, len(res)), 3),
            "within_500km": round(sum(r["err_km"] <= 500 for r in res) / max(1, len(res)), 3),
            "within_250km": round(sum(r["err_km"] <= 250 for r in res) / max(1, len(res)), 3),
            "radius_factor_for_68pct": round(ratio[int(0.68 * (len(ratio) - 1))], 2) if ratio else None}


def main():
    models, cases_path, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    base = {k: getattr(E, k) for k in GRID}
    t0 = time.time()
    rows = []
    for combo in itertools.product(*GRID.values()):
        params = dict(zip(GRID, combo))
        res = run(cases, models, params)
        rows.append({"params": params, "all": score(res), "half_a": score(res[0::2]), "half_b": score(res[1::2])})
    for k, v in base.items():
        setattr(E, k, v)
    rows.sort(key=lambda r: (r["all"]["median_err_all_km"], -r["all"]["within_500km"]))
    cur = next(r for r in rows if r["params"] == base)
    best = rows[0]
    robust = [r for r in rows if r["half_a"]["median_err_all_km"] <= cur["half_a"]["median_err_all_km"]
              and r["half_b"]["median_err_all_km"] <= cur["half_b"]["median_err_all_km"]]
    report = {"cases": len(cases), "seconds": round(time.time() - t0), "current": cur, "best": best,
              "best_robust": robust[0] if robust else cur, "top10": rows[:10]}
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("cases", "current", "best", "best_robust")}, indent=1))


if __name__ == "__main__":
    main()
