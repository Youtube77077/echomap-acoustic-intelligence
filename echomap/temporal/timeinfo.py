"""Recording-time analysis: keeps file times, metadata times, user times and estimates strictly separate."""
from __future__ import annotations

import numpy as np

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def analyse(meta: dict, ctx: dict, records: list[dict], species_profiles: list[dict]) -> dict:
    out = {
        "file_system_times": meta.get("filesystem_times"),
        "embedded_metadata_times": meta.get("embedded_times") or {},
        "embedded_note": ("Container/stream tags written by the recording device or later software. They can be wrong, "
                          "reset or edited, so they are claims, not verification."),
        "user_provided": (ctx or {}).get("date") or {"mode": "unknown"},
        "user_timezone": (ctx or {}).get("timezone") or "unknown",
        "estimated": {},
        "verified_external": {"status": "none",
                              "note": "No external time verification was performed (ENF reference databases and public "
                                      "transport schedules are not bundled)."},
    }
    clues = []
    for r in records:
        if r.get("temporal_relevance") and r["temporal_relevance"] != "None.":
            clues.append({"record": r["id"], "label": r["label"], "clue": r["temporal_relevance"]})
    out["estimated"]["clues"] = clues
    # seasonal compatibility from species occurrence at the best geographic candidate
    profiles = [p for p in species_profiles if p.get("seasonality_at_best_candidate")]
    if profiles:
        conf = {r["scientific_name"]: r["confidence"] for r in records if r.get("scientific_name")}
        score = np.zeros(12)
        for p in profiles:
            m = np.array(p["seasonality_at_best_candidate"]["monthly_max_occurrence"])
            c = conf.get(p["scientific_name"], 0.3)
            score += np.log(c * np.maximum(m, 0.02) + (1 - c) * m.mean() + 1e-9) - np.log(m.mean() + 1e-9)
        score /= np.sqrt(len(profiles))
        rel = np.exp(score - score.max())
        best = [MONTHS[i] for i in np.argsort(-rel) if rel[i] >= 0.6]
        flat = rel.min() > 0.6
        out["estimated"]["season_from_species"] = {
            "relative_compatibility_by_month": {MONTHS[i]: round(float(rel[i]), 3) for i in range(12)},
            "most_compatible_months": best,
            "interpretation": ("Species mix is compatible with any month (resident species), so there is no seasonal "
                               "information." if flat else
                               f"Species mix is most compatible with: {', '.join(best)} (at the best geographic candidate)."),
            "caveat": "Depends on the location estimate and on range-model seasonality. Not a date estimate."}
    else:
        out["estimated"]["season_from_species"] = {"interpretation": "No species with seasonal range data, or no location "
                                                                    "candidate, so no seasonal estimate."}
    out["summary"] = ("No recording time can be established from the audio alone." if not clues else
                      f"{len(clues)} heuristic time clue(s) found. None is verified.")
    return out
