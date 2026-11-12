"""Bell-ring matching against Dove's Guide for Church Bell Ringers (CC BY-SA 4.0, dove.cccbr.org.uk).

The match compares the observed bells (nominal frequencies, lowest = tenor) with each listed ring.
When Dove has per-bell nominals, the observed bells are aligned from the tenor upward (bands often ring
the back bells of a larger ring). Otherwise only the tenor nominal is used. Tolerance is σ = 25 cents,
which allows for recording, temperature and tuning differences. A match means "compatible with", never
"identified". Many rings share a tenor note, and a recording made at the wrong sample rate shifts every pitch.
"""
from __future__ import annotations

import gzip
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data" / "dove_rings.json.gz"
SIGMA_CENTS = 25.0


@lru_cache(maxsize=1)
def rings() -> dict:
    if not DATA.exists():
        return {"rings": [], "meta": {}}
    with gzip.open(DATA, "rt", encoding="utf-8") as f:
        return json.load(f)


def match(analysis: dict, top: int = 15) -> dict:
    data = rings()
    obs = [b["nominal_hz"] for b in analysis.get("bells", [])]
    if not data["rings"]:
        return {"available": False, "reason": "Dove's Guide data not bundled."}
    if len(obs) < 1:
        return {"available": True, "candidates": [], "reason": "No bells reconstructed from the recording."}
    obs = np.array(sorted(obs))
    n = len(obs)
    out = []
    for r in data["rings"]:
        if r.get("bells", 0) < n or not r.get("tenor_hz"):
            continue
        noms = r.get("nominals")
        if noms and len(noms) >= n:
            ref = np.array(sorted(noms)[:n])
            err = float(np.sqrt(np.mean((1200 * np.log2(obs / ref)) ** 2)))
            basis = f"{n} bell nominals"
        else:
            err = abs(1200 * float(np.log2(obs[0] / r["tenor_hz"])))
            basis = "tenor nominal only"
        if err > 4 * SIGMA_CENTS:
            continue
        s = float(np.exp(-0.5 * (err / SIGMA_CENTS) ** 2)) * (1.0 if r["bells"] == n else 0.5)
        out.append({**{k: r.get(k) for k in ("ring_id", "tower_id", "place", "dedication", "county", "country", "iso",
                                             "lat", "lon", "bells", "tenor_hz", "tenor_note")},
                    "rms_error_cents": round(err, 1), "basis": basis, "score": round(s, 4)})
    out.sort(key=lambda c: -c["score"])
    return {"available": True, "observed_bells": int(n), "observed_nominals_hz": [round(float(v), 2) for v in obs],
            "n_compatible": len(out), "candidates": out[:top], "all_compatible": out[:400],
            "dataset": data.get("meta", {}),
            "note": ("Compatible rings only: many towers share bell counts and tenor notes, and pitch errors from "
                     "recording speed or distance shift the comparison. Dove's Guide covers mainly full-circle rings "
                     "(mostly UK). Continental swinging bells are largely absent.")}
