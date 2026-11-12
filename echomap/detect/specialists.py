"""Specialised analyses that add detail or extra records: languages, engines/vehicles,
infrastructure signals, sirens, bells, wind, mains hum, reverberation, unclassified energy."""
from __future__ import annotations

import numpy as np

from .inventory import Inventory, overlap, runs, tier_for
from .taxonomy import SPEECH, class_table

VEHICLE_NAMES = {"Car", "Bus", "Truck", "Motorcycle", "Motor vehicle (road)", "Vehicle", "Car passing by", "Race car, auto racing",
                 "Engine", "Light engine (high frequency)", "Medium engine (mid frequency)", "Heavy engine (low frequency)",
                 "Idling", "Accelerating, revving, vroom", "Engine starting", "Engine knocking", "Boat, Water vehicle",
                 "Motorboat, speedboat", "Train", "Rail transport", "Subway, metro, underground", "Traffic noise, roadway noise"}
ROAD_ENGINE = {"Car", "Bus", "Truck", "Motorcycle", "Motor vehicle (road)", "Car passing by", "Race car, auto racing",
               "Engine", "Light engine (high frequency)", "Medium engine (mid frequency)", "Heavy engine (low frequency)",
               "Idling", "Accelerating, revving, vroom", "Engine starting", "Engine knocking", "Vehicle"}
NOT_SUPPORTED_VEHICLE = ("Manufacturer, model, generation and exhaust configuration: NOT SUPPORTED. No validated public "
                         "model or reference dataset identifies these from audio. Electric vs combustion and "
                         "diesel vs petrol are not classes of the AudioSet model either.")


# ----------------------------------------------------------------------------- languages
def speech_regions(sed: dict, min_score: float = 0.3, min_len: float = 1.0) -> list[tuple[float, float, float]]:
    idx = [c["idx"] for c in class_table() if c["name"] in SPEECH]
    s = sed["frame_scores"][:, idx].max(axis=1)
    hop = sed["hop_s"]
    out = []
    for a, b in runs(s >= min_score, gap=3):
        t0, t1 = a * hop, b * hop
        if t1 - t0 >= min_len:
            out.append((t0, t1, float(s[a:b].max())))
    return out


def add_languages(inv: Inventory, regions, x16k: np.ndarray, langid_fn) -> list[dict]:
    recs = []
    for t0, t1, sp in regions:
        chunks = []
        t = t0
        while t < t1 - 0.5:
            e = min(t1, t + 30.0)
            chunks.append(x16k[int(t * 16000):int(e * 16000)])
            t = e
        res = [langid_fn(c) for c in chunks]
        agg: dict[str, dict] = {}
        for r in res:
            for d in r:
                a = agg.setdefault(d["code"], {"code": d["code"], "name": d["name"], "prob": 0.0})
                a["prob"] += d["prob"] / len(res)
        ranked = sorted(agg.values(), key=lambda d: -d["prob"])
        top = ranked[0]
        mixed = len(ranked) > 1 and ranked[1]["prob"] >= 0.25
        speech_rec = inv.find_overlapping(t0, t1, lambda r: r["category"] == "language" and "_audioset_names" in r, 0.3)
        conf = top["prob"]
        rec = inv.add(
            category="language", label=f"Spoken language: {top['name']}", specific=f"{top['name']} ({top['code']})",
            language_code=top["code"], language_name=top["name"],
            candidates=[{"label": f"{d['name']} ({d['code']})", "score": round(d["prob"], 4)} for d in ranked[1:4]],
            start=t0, end=t1, confidence=conf, tier=tier_for(conf, bool(speech_rec)),
            model="Whisper base (LID-only graph, int8)", corroborated_by=[f"{r['id']} {r['label']}" for r in speech_rec],
            overlapping_languages=("Possible: second language has ≥ 25% of the probability mass (overlapping or "
                                   "code-switched speech, or uncertainty)" if mixed else "No evidence of a second language"),
            evidence={"model_outputs": [{"model": "Whisper base LID", "chunk": i, "top": r} for i, r in enumerate(res)],
                      "speech_score_max": round(sp, 3)},
            limitations=("Language ID only, so no words are transcribed and speakers are not identified. Accuracy drops "
                         "on short, distant, overlapping or accented speech and between related languages "
                         "(e.g. Dutch and German). Speech from radios, TVs or phones counts as speech."),
            signature="Speech region",
        )
        for r in speech_rec:
            rec["evidence"].setdefault("associated_audioset_events", []).append(
                {"label": r["label"], "score": r["confidence"], "start": r["start"], "end": r["end"]})
        inv.remove(speech_rec)
        recs.append(rec)
    return recs


# ----------------------------------------------------------------------------- vehicles / engines
def add_engine_analysis(inv: Inventory, x4k: np.ndarray, f0_fn) -> list[dict]:
    new = []
    for r in list(inv.records):
        if r.get("_audioset_names") is None or r["category"] not in ("vehicles", "engines_machinery", "public_transport",
                                                                     "railway", "road_infrastructure"):
            continue
        if not (set(r["_audioset_names"]) & VEHICLE_NAMES):
            continue
        road_engine = r["label"] in ROAD_ENGINE
        r["identification_detail"] = NOT_SUPPORTED_VEHICLE
        if r["duration"] < 1.0:
            continue
        eng = f0_fn(x4k, 4000, r["start"], r["end"])
        r["evidence"]["engine_harmonics"] = eng
        r.setdefault("directly_measured", {})
        if eng.get("available"):
            r["directly_measured"]["firing_frequency_hz"] = eng["f0_median_hz"]
            r["directly_measured"]["states"] = eng["states"]
            r["signature"] = f"Harmonic series, f0 ≈ {eng['f0_median_hz']} Hz"
            if "corroborated_by" in r:
                r["corroborated_by"].append("Harmonic-series measurement (engine-like periodicity)")
                r["tier"] = tier_for(r["confidence"], True)
            merged_acc = []
            for s in eng["states"] if (road_engine and eng.get("continuity", 0) >= 0.8) else []:
                if s["state"] != "accelerating":
                    continue
                if merged_acc and s["start"] <= merged_acc[-1]["end"] + 0.1:
                    merged_acc[-1]["end"] = s["end"]
                    merged_acc[-1]["slope_pct_per_s"] = max(merged_acc[-1]["slope_pct_per_s"], s["slope_pct_per_s"])
                else:
                    merged_acc.append(dict(s))
            for s in merged_acc:
                if s["end"] - s["start"] >= 1.5:
                    new.append(inv.add(
                        category="engines_machinery", label="Engine acceleration event",
                        specific=f"f0 rising {s['slope_pct_per_s']}%/s", start=s["start"], end=s["end"],
                        confidence=min(0.95, r["confidence"]), tier="possible", kind="derived",
                        model="Harmonic f0 tracking (EchoMap DSP)", parent=r["id"],
                        evidence={"from": r["id"], "segment": s},
                        limitations="Derived from f0 slope. Doppler shift from a passing vehicle also raises f0.",
                        signature="Rising firing frequency"))
            for g in (eng.get("possible_gear_changes_s", []) if (road_engine and eng.get("continuity", 0) >= 0.9) else []):
                new.append(inv.add(category="engines_machinery", label="Possible gear change", start=g - 0.2, end=g + 0.2,
                                   confidence=0.3, tier="possible", kind="derived", parent=r["id"],
                                   model="Harmonic f0 tracking (EchoMap DSP)",
                                   evidence={"from": r["id"], "time_s": g},
                                   limitations="Heuristic: sudden f0 drop between rising segments.",
                                   signature="f0 drop > 15 %"))
        else:
            r["directly_measured"]["firing_frequency_hz"] = None
            r["directly_measured"]["note"] = eng.get("reason")
    return new


# ----------------------------------------------------------------------------- infrastructure / signals
def classify_periodic(p: dict) -> tuple[str, list[dict], str]:
    per, dur = p["period_s"], p.get("pulse_duration_s", 0.05)
    cands = []
    if p["kind"] == "click_train" and 0.08 <= per <= 0.25:
        label = "Rapid repeating ticks"
        cands = [{"label": "Pedestrian crossing 'walk' ticker (several countries use rapid ticks)", "score": None},
                 {"label": "Vehicle turn indicator relay / mechanical ticking", "score": None}]
        geo = "Weak. Rapid tickers are used by pedestrian signals in several countries, but the pattern alone does not identify one."
    elif 0.85 <= per <= 1.15 and dur <= 0.2:
        label = "Repeating signal at ~1 s interval"
        cands = [{"label": "Accessible pedestrian signal locator tone (US MUTCD: ≤0.15 s tone, 1 s interval; similar elsewhere)", "score": None},
                 {"label": "Reversing alarm / equipment beeper", "score": None},
                 {"label": "Clock ticking", "score": None}]
        geo = "Weak. A 1 s locator pattern matches the US MUTCD spec but is not unique to the US."
    elif 0.3 <= per < 0.85:
        label = "Repeating tone/beep"
        cands = [{"label": "Reversing alarm / vehicle beeper", "score": None},
                 {"label": "Level-crossing bell or warning tone", "score": None},
                 {"label": "Equipment or door warning beeper", "score": None}]
        geo = "Not distinctive without a reference recording."
    else:
        label = "Periodic signal"
        cands = [{"label": "Mechanical or electronic periodic source", "score": None}]
        geo = "Not distinctive."
    return label, cands, geo


def add_periodic(inv: Inventory, patterns: list[dict], profile_fn) -> list[dict]:
    recs = []
    SIRENISH = {"Siren", "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)", "Emergency vehicle",
                "Civil defense siren", "Music", "Musical instrument"}
    for p in patterns:
        host = inv.find_overlapping(p["start"], p["end"], lambda r: bool(set(r.get("_audioset_names", [])) & SIRENISH), 0.5)
        if host:
            for h in host:
                h["evidence"].setdefault("periodic_components", []).append(p)
            continue  # part of a siren or music, not a separate signal
        label, cands, geo = classify_periodic(p)
        birds = inv.find_overlapping(p["start"], p["end"], lambda r: r["category"] == "birds_wildlife", 0.3)
        strong_animal = [b for b in birds if b["confidence"] >= 0.5]
        if strong_animal:
            for b in strong_animal:
                b["evidence"].setdefault("periodic_components", []).append(p)
            continue  # stridulation or repeated song notes of a detected animal
        if birds:
            cands = [{"label": "Repetitive animal call element (overlaps weak animal detection)", "score": None}] + cands
        sed = inv.find_overlapping(p["start"], p["end"], lambda r: any(
            n in ("Reversing beeps", "Beep, bleep", "Tick", "Tick-tock", "Clock", "Alarm", "Buzzer")
            for n in r.get("_audioset_names", [])), 0.2)
        regular = p["period_cv"] <= 0.04 and p["count"] >= 8
        conf = 0.6 if regular and not birds else 0.3
        desc = (f"{p['count']} pulses, period {p['period_s']} s (CV {p['period_cv']})" +
                (f", tone {p['f_hz']} Hz, pulse {p['pulse_duration_s']} s" if "f_hz" in p else ", broadband clicks"))
        recs.append(inv.add(
            category="road_infrastructure" if "pedestrian" in cands[0]["label"].lower() else "electronic_signals",
            label=label, specific=desc, candidates=cands, start=p["start"], end=p["end"], confidence=conf,
            tier=tier_for(conf, bool(sed), measurement=False), kind="measurement", model="Periodicity analysis (EchoMap DSP)",
            frequency=profile_fn(p["start"], p["end"]), evidence={"pattern": p}, signature=desc,
            corroborated_by=[f"{r['id']} {r['label']}" for r in sed], geographic_relevance=geo,
            limitations="The period and pitch are measured. The source type is an interpretation of the pattern, and "
                        "candidates are listed without scores because no validated classifier exists."))
    return recs


def add_sirens(inv: Inventory, x32k, siren_fn) -> None:
    for r in inv.records:
        names = set(r.get("_audioset_names", []))
        if names & {"Siren", "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)",
                    "Emergency vehicle", "Civil defense siren"}:
            pat = siren_fn(x32k, 32000, r["start"], r["end"])
            r["evidence"]["siren_pattern"] = pat
            r["specific"] = f"Pitch pattern: {pat.get('pattern')}"
            if "two-tone" in pat.get("pattern", ""):
                r["geographic_relevance"] = ("Weak regional cue: two-tone (hi-lo) emergency sirens are common in much of "
                                             "Europe; wail/yelp sweeps dominate in North America and elsewhere. Many "
                                             "countries use both.")
            elif "sweep" in pat.get("pattern", ""):
                r["geographic_relevance"] = "Weak: wail/yelp sweep patterns are used in many countries."


def add_bell_counts(inv: Inventory, onset_times: list[float]) -> None:
    for r in inv.records:
        names = set(r.get("_audioset_names", []))
        if names & {"Church bell", "Change ringing (campanology)", "Bell"}:
            ts = [t for t in onset_times if r["start"] - 0.2 <= t <= r["end"] + 0.2]
            r["evidence"]["strike_onsets"] = ts[:200]
            if "Change ringing (campanology)" in names:
                r["geographic_relevance"] = ("Moderate: full-circle change ringing is concentrated in England (most "
                                             "towers in Dove's Guide), with smaller numbers in other countries.")
            if 2 <= len(ts) <= 13:
                iv = np.diff(ts)
                if len(iv) and iv.mean() >= 1.2 and iv.std() / iv.mean() < 0.15:
                    r["temporal_relevance"] = (f"Possible clock strike count: {len(ts)} regular strikes, which could "
                                               f"mark {len(ts)} o'clock (or {len(ts) + 12 if len(ts) < 12 else len(ts)}:00). "
                                               "Heuristic: partial strikes or recording cut-offs change the count.")


def add_mains(inv: Inventory, mains: dict, duration: float) -> None:
    if not mains.get("detected"):
        return
    fam = mains["families"][mains["nominal_hz"]]
    strong = sum(1 for h in fam["harmonics"] if h["snr_db"] >= 10) >= 3 and duration >= 10.0
    desc = ", ".join(f"{h['freq_hz']} Hz ({h['snr_db']} dB)" for h in fam["harmonics"])
    inv.add(category="electronic_signals", label=f"Mains hum ({mains['nominal_hz']} Hz electrical network)",
            specific=desc, start=0.0, end=duration, confidence=min(0.99, 0.5 + fam["score"] / 60),
            tier="confirmed" if strong and mains.get("mains_consistent") else "probable", kind="measurement",
            model="Narrowband harmonic analysis (EchoMap DSP)", frequency={"peak_hz": mains["nominal_hz"]},
            evidence={"mains": {k: v for k, v in mains.items() if k != "enf_trace"}},
            signature=f"Harmonic series at {mains['nominal_hz']} Hz multiples",
            geographic_relevance=f"Indicates a {mains['nominal_hz']} Hz power grid (independent infrastructure evidence).",
            temporal_relevance=("An ENF trace was extracted. Matching it against grid frequency logs could verify "
                                "the recording time, but no reference database is bundled."),
            limitations=mains.get("confidence_note", ""))


def add_wind(inv: Inventory, wind: dict) -> None:
    wind_sed = [r for r in inv.records if set(r.get("_audioset_names", [])) & {"Wind", "Wind noise (microphone)"}]
    for g in wind["gusts"]:
        corr = [r for r in wind_sed if overlap(g["start"], g["end"], r["start"], r["end"]) > 0]
        conf = min(0.9, 0.3 + g["peak_db_above_baseline"] / 30)
        inv.add(category="weather", label="Wind gust (low-frequency microphone wind noise)",
                specific=f"+{g['peak_db_above_baseline']} dB above low-frequency baseline", start=g["start"], end=g["end"],
                confidence=conf, tier=tier_for(conf, bool(corr)), kind="measurement",
                model="Low-frequency envelope analysis (EchoMap DSP)", corroborated_by=[f"{r['id']} {r['label']}" for r in corr],
                evidence={"gust": g}, signature="LF energy burst 20-150 Hz",
                limitations=wind["note"])


def add_reverb(inv: Inventory, rt: dict, duration: float) -> None:
    if rt.get("rt60_s") is None:
        return
    inv.add(category="buildings_acoustics", label="Reverberation estimate", specific=f"RT60 ≈ {rt['rt60_s']} s ({rt['interpretation']})",
            start=0.0, end=duration, confidence=min(0.9, 0.3 + rt["n_decays"] / 40), tier="probable" if rt["n_decays"] >= 8 else "possible",
            kind="measurement", model="Blind decay analysis (EchoMap DSP)", evidence={"reverb": rt},
            signature=f"Median of {rt['n_decays']} free decays", limitations=rt["method"] + " Coarse indicator only.")


def add_unclassified(inv: Inventory, segments: list[dict], profile_fn, sed: dict | None = None, labels=None) -> None:
    for s in segments:
        if any(overlap(s["start"], s["end"], r["start"], r["end"]) > 0.5 * (s["end"] - s["start"])
               for r in inv.records if r["end"] - r["start"] < inv.duration * 0.9):
            continue
        hint = None
        if sed is not None and labels is not None:
            a = int(s["start"] / sed["hop_s"])
            b = max(a + 1, int(np.ceil(s["end"] / sed["hop_s"])))
            seg = sed["frame_scores"][a:b].max(axis=0)
            top = np.argsort(-seg)[:3]
            hint = [{"label": labels[i], "score": round(float(seg[i]), 3)} for i in top if seg[i] >= 0.03]
        inv.add(category="unknown", label="Unclassified sound event", start=s["start"], end=s["end"],
                specific=("Closest model labels (below threshold): " +
                          ", ".join(f"{h['label']} {h['score']:.2f}" for h in hint)) if hint else None,
                candidates=hint or [],
                confidence=0.0, tier="unclassified", kind="measurement", model="Energy segmentation (EchoMap DSP)",
                frequency=profile_fn(s["start"], s["end"]), evidence={"segment": s},
                signature=f"{s['peak_db_above_floor']} dB above noise floor",
                limitations="Energy above the noise floor that no model classified above the listing threshold.")
