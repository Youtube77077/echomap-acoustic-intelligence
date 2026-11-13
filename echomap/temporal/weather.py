"""Weather-related observations: acoustic (direct) vs external records (optional), kept separate."""
from __future__ import annotations

WEATHER_LABELS = {"Rain", "Raindrop", "Rain on surface", "Thunder", "Thunderstorm", "Wind", "Wind noise (microphone)",
                  "Rustling leaves", "Waves, surf"}


def acoustic_observations(records: list[dict], wind: dict) -> dict:
    obs = []
    for r in records:
        names = set(r.get("evidence", {}).get("ontology_parents_detected", [])) | {r["label"]}
        names |= {m["label"] for m in r.get("evidence", {}).get("model_outputs", []) if isinstance(m, dict) and "label" in m}
        hit = names & WEATHER_LABELS
        if hit or r["label"].startswith("Wind gust"):
            obs.append({"record": r["id"], "label": r["label"], "start": r["start"], "end": r["end"],
                        "confidence": r["confidence"], "tier": r["tier"]})
    rain = any("Rain" in o["label"] for o in obs)
    thunder = any("Thunder" in o["label"] for o in obs)
    intensity = None
    rr = [o for o in obs if "Rain" in o["label"]]
    if rr:   # heuristic: how sure, how much of the time, and thunder
        total = max((r["end"] for r in records), default=1.0) or 1.0
        frac = min(1.0, sum(o["end"] - o["start"] for o in rr) / total)
        conf = max(o["confidence"] for o in rr)
        intensity = ("heavy" if thunder or (conf >= 0.6 and frac >= 0.6) else
                     "light" if conf < 0.35 or frac < 0.2 else "moderate")
    gusts = len(wind.get("gusts", []))
    return {"observations": obs,
            "summary": {"rain_detected": rain, "rain_intensity_estimate": intensity, "thunder_detected": thunder,
                        "wind_gust_events": gusts,
                        "lf_wind_noise_fraction": wind.get("fraction_frames_lf_dominant")},
            "not_determinable": "Wind speed, wind direction and temperature cannot be measured from ordinary audio. "
                                "Rain intensity is relative only.",
            "type": "direct acoustic observation"}


def compare_external(acoustic: dict, ext: dict, hour: int | None) -> dict:
    h = ext.get("hourly", {})
    if not h.get("time"):
        return {"status": "no data"}
    idx = list(range(len(h["time"]))) if hour is None else [min(len(h["time"]) - 1, max(0, hour))]
    prec = [h["precipitation"][i] or 0 for i in idx]
    gust = [h["wind_gusts_10m"][i] or 0 for i in idx]
    rain_ext = max(prec) > 0.1
    s = acoustic["summary"]
    notes = []
    if s["rain_detected"] and not rain_ext:
        notes.append("Acoustic rain detected but the archive shows no precipitation: weak contradiction (could be a "
                     "local shower, sprinklers or a misclassification).")
    if s["rain_detected"] and rain_ext:
        notes.append("Acoustic rain is consistent with archive precipitation.")
    if not s["rain_detected"] and rain_ext:
        notes.append("Archive shows precipitation but none detected acoustically (indoors, or rain at another hour).")
    return {"precipitation_mm": prec, "wind_gusts_kmh": gust, "temperature_c": [h["temperature_2m"][i] for i in idx],
            "consistency": notes or ["No conflict found."], "source": ext.get("source"), "type": "external record",
            "note": ext.get("note")}
