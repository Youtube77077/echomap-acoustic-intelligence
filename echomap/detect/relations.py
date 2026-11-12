"""Event relationship graph and sequence hypotheses.

Relations are temporal facts (before/after, overlap, simultaneous, repetition). Sequence patterns
are *hypotheses* consistent with common real-world sequences. Co-occurrence never establishes causation.
"""
from __future__ import annotations

import numpy as np

LONG = 0.9  # fraction of the recording above which a record is treated as a background measurement


def _cat(r, *cats):
    return r["category"] in cats


def _env_shape(rms_db: np.ndarray, hop: float, t0: float, t1: float) -> dict | None:
    a, b = int(t0 / hop), int(t1 / hop)
    seg = rms_db[a:b]
    if len(seg) < 8:
        return None
    k = int(np.argmax(seg))
    rise = float(seg[k] - seg[: max(1, k)].min()) if k > 0 else 0.0
    fall = float(seg[k] - seg[k:].min()) if k < len(seg) - 1 else 0.0
    return {"peak_t": round(t0 + k * hop, 2), "rise_db": round(rise, 1), "fall_db": round(fall, 1),
            "rise_then_fall": rise >= 6 and fall >= 6 and 0.15 < k / len(seg) < 0.85}


def build(records: list[dict], duration: float, rms_db: list[float], rms_hop: float) -> dict:
    ev = [r for r in records if r["duration"] < LONG * duration]
    rms = np.array(rms_db, np.float32) if rms_db else np.zeros(0)
    edges = []
    for i, a in enumerate(ev):
        for b in ev[i + 1:]:
            ov = min(a["end"], b["end"]) - max(a["start"], b["start"])
            if abs(a["start"] - b["start"]) < 0.5 and ov > 0:
                rel = "simultaneous onset"
            elif ov > 0:
                rel = "overlapping"
            elif 0 <= b["start"] - a["end"] <= 3.0:
                rel = "followed by"
            else:
                continue
            edges.append({"from": a["id"], "to": b["id"], "relation": rel, "from_label": a["label"], "to_label": b["label"]})
            if len(edges) >= 400:
                break
    # repetition
    reps = []
    by_label: dict[str, list[dict]] = {}
    for r in ev:
        by_label.setdefault(r.get("specific") if r.get("scientific_name") else r["label"], []).append(r)
    for lab, rs in by_label.items():
        if len(rs) >= 3:
            starts = np.array(sorted(x["start"] for x in rs))
            iv = np.diff(starts)
            reps.append({"label": lab, "count": len(rs), "mean_interval_s": round(float(iv.mean()), 2),
                         "interval_cv": round(float(iv.std() / (iv.mean() + 1e-9)), 2),
                         "regular": bool(iv.std() / (iv.mean() + 1e-9) < 0.2), "records": [x["id"] for x in rs][:30]})
    # sequence hypotheses
    hyps = []
    rails = [r for r in records if _cat(r, "railway")]
    warns = [r for r in records if (_cat(r, "electronic_signals", "bells_alarms", "road_infrastructure") and
                                    ("Repeating" in r["label"] or "Periodic" in r["label"] or "bell" in r["label"].lower() or "ticks" in r["label"]))]
    for t in rails:
        for w in warns:
            if w["start"] <= t["start"] + 2 and w["end"] >= t["start"] - 30 and w["start"] >= t["start"] - 60:
                hyps.append({"pattern": "railway level-crossing sequence", "records": [w["id"], t["id"]],
                             "explanation": f"A repeating warning signal ({w['label']}) starts before or at the train sound "
                                            f"({t['label']}), as at an active level crossing.",
                             "status": "hypothesis (temporal consistency only)"})
                break
    for r in records:
        if r["duration"] >= 3 and r["duration"] < LONG * duration and len(rms):
            shp = _env_shape(rms, rms_hop, r["start"], r["end"])
            if not shp or not shp["rise_then_fall"]:
                continue
            if _cat(r, "vehicles", "engines_machinery", "public_transport", "railway", "road_infrastructure"):
                kind = "aircraft fly-over" if "aircraft" in r["label"].lower() or "Helicopter" in r["label"] else "pass-by"
                hyps.append({"pattern": kind, "records": [r["id"]], "envelope": shp,
                             "explanation": f"Level rises {shp['rise_db']} dB to a peak at {shp['peak_t']} s and falls "
                                            f"{shp['fall_db']} dB, consistent with a source approaching and receding.",
                             "status": "hypothesis"})
            elif "siren" in r["label"].lower():
                hyps.append({"pattern": "siren approaching and receding", "records": [r["id"]], "envelope": shp,
                             "explanation": "Rise-peak-fall level envelope on a siren.", "status": "hypothesis"})
    chimes = [r for r in records if any(k in r["label"] for k in ("Chime", "Ding-dong", "Beep", "Doorbell", "Repeating tone"))]
    speech = [r for r in records if r["category"] == "language"]
    for c in chimes:
        s = next((s for s in speech if 0 <= s["start"] - c["end"] <= 2.5), None)
        if s:
            hyps.append({"pattern": "possible public-address announcement", "records": [c["id"], s["id"]],
                         "explanation": "An attention chime is followed within 2.5 s by speech.", "status": "hypothesis"})
    rain = [r for r in records if "Rain" in r["label"]]
    water = [r for r in records if r["category"] == "water_geological"]
    for a in rain:
        w = next((w for w in water if w["start"] >= a["start"]), None)
        if w:
            hyps.append({"pattern": "rain followed by water flow", "records": [a["id"], w["id"]],
                         "explanation": "Rain detected, then flowing or dripping water.", "status": "hypothesis"})
    for h in hyps:
        for rid in h["records"]:
            for r in records:
                if r["id"] == rid:
                    r.setdefault("related_events", []).append(h["pattern"])
    return {"edges": edges, "repetitions": reps, "hypotheses": hyps,
            "note": "Relations are temporal. Sequence patterns are hypotheses; co-occurrence is not causation."}
